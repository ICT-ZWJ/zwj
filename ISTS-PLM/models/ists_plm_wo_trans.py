import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import TimeEmbedding, ValueEmbedding


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        hidden = self.fc2(F.relu(self.fc1(x)))
        return self.dropout(hidden)


class DistortionAwareEmbedding(nn.Module):
    def __init__(self, d_model, device=None, dropout=0.1, use_te=True):
        super().__init__()
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        self.distortion_embedding = nn.Linear(1, d_model)
        self.use_te = use_te
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark, distortion):
        # tt/x/x_mark: [B, L, D], distortion: [B, L, 1]
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1))  # [B, L, D, d_model]
        x_int = torch.cat([x.unsqueeze(dim=-1), x_mark.unsqueeze(dim=-1)], dim=-1)
        value_emb = self.value_embedding(x_int)  # [B, L, D, d_model]

        if self.use_te:
            x_emb = x_mark.unsqueeze(dim=-1) * time_emb + value_emb
        else:
            x_emb = value_emb

        dist_emb = self.distortion_embedding(distortion)  # [B, L, d_model]
        dist_emb = dist_emb.unsqueeze(dim=2).expand_as(x_emb)

        x_emb = x_emb + dist_emb
        B, L, D, E = x_emb.shape
        x_emb = x_emb.permute(0, 2, 1, 3).reshape(B * D, L, E)
        return self.dropout(x_emb)


class istsplm_spd_forecast(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.d_model = opt.d_model
        self.spd_dim = getattr(opt, "spd_dim", 4)
        self.alpha = getattr(opt, "dist_alpha", 1.0)
        self.beta = getattr(opt, "dist_beta", 1.0)
        self.eps = 1e-6
        self.jitter = 1e-4

        self.enc_embedding = DistortionAwareEmbedding(
            d_model=self.d_model,
            device=opt.device,
            dropout=opt.dropout,
            use_te=True,
        )
        self.num_layer = getattr(opt, "num_layer", 1)
        self.temporal = nn.Sequential(
            *[
                MultiLayerPerceptron(self.d_model, self.d_model, dropout=opt.dropout)
                for _ in range(self.num_layer)
            ]
        )
        self.spd_proj = nn.Linear(self.d_model, self.spd_dim * self.spd_dim)
        self.spd_readout = nn.Linear(self.spd_dim * self.spd_dim, self.d_model)

        self.var_attn = nn.MultiheadAttention(self.d_model, num_heads=1, batch_first=True)
        self.var_ln = nn.LayerNorm(self.d_model)

        self.predict_decoder = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        ).to(opt.device)

    def _compute_distortion(self, tt, mask):
        # tt: [B, L] or [B, L, D], mask: [B, L, D]
        B, L, D = mask.shape
        if L == 0:
            return torch.zeros((B, 0, 1), device=mask.device)
        missing_rate = 1.0 - (mask.sum(dim=-1, keepdim=True) / (D + self.eps))

        if tt.dim() == 2:
            time_scalar = tt.unsqueeze(dim=-1)
        else:
            denom = mask.sum(dim=-1, keepdim=True) + self.eps
            time_scalar = (tt * mask).sum(dim=-1, keepdim=True) / denom

        delta = torch.zeros_like(time_scalar)
        if L > 1:
            delta[:, 1:, :] = torch.abs(time_scalar[:, 1:, :] - time_scalar[:, :-1, :])
            max_delta = delta.max(dim=1, keepdim=True).values + self.eps
        else:
            max_delta = torch.ones_like(delta[:, :1, :]) + self.eps
        delta_norm = delta / max_delta

        distortion = self.alpha * missing_rate + self.beta * delta_norm
        return torch.clamp(distortion, min=0.0, max=1.0)

    def _spd_from_features(self, features):
        # features: [B*D, L, d_model] -> [B*D, L, m, m]
        BDL, L, _ = features.shape
        m = self.spd_dim
        raw = self.spd_proj(features).view(BDL, L, m, m)
        raw = torch.tanh(raw)
        sym = 0.5 * (raw + raw.transpose(-1, -2))
        eye = torch.eye(m, device=features.device).view(1, 1, m, m)
        spd = sym @ sym.transpose(-1, -2) + self.jitter * eye
        return spd

    def _spd_logm(self, spd):
        # spd: [..., m, m]
        eigvals, eigvecs = torch.linalg.eigh(spd)
        eigvals = torch.clamp(eigvals, min=self.jitter, max=1e4)
        logvals = torch.log(eigvals)
        out = eigvecs @ torch.diag_embed(logvals) @ eigvecs.transpose(-1, -2)
        return torch.nan_to_num(out)

    def _spd_expm(self, tangent):
        eigvals, eigvecs = torch.linalg.eigh(tangent)
        eigvals = torch.clamp(eigvals, min=-20.0, max=20.0)
        expvals = torch.exp(eigvals)
        out = eigvecs @ torch.diag_embed(expvals) @ eigvecs.transpose(-1, -2)
        return torch.nan_to_num(out)

    def _weighted_frechet_mean(self, spd_seq, weights):
        # spd_seq: [B*D, L, m, m], weights: [B*D, L]
        BDL, L, m, _ = spd_seq.shape
        weights = weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
        logm = self._spd_logm(spd_seq)
        mean_log = (weights * logm).sum(dim=1)
        return self._spd_expm(mean_log)

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        # observed_data/mask: [B, L, D], observed_tp: [B, L] or [B, L, D]
        B, L, D = observed_data.shape

        if observed_tp.dim() == 2:
            observed_tp = observed_tp.unsqueeze(dim=-1).repeat(1, 1, D)

        if L == 0:
            if time_steps_to_predict.dim() == 2:
                Lp = time_steps_to_predict.size(1)
            else:
                Lp = time_steps_to_predict.size(1)
            return torch.zeros((1, B, Lp, D), device=observed_data.device)

        distortion = self._compute_distortion(observed_tp, observed_mask)
        emb = self.enc_embedding(observed_tp, observed_data, observed_mask, distortion)

        seq_out = self.temporal(emb)  # [B*D, L, d_model]
        spd_seq = self._spd_from_features(seq_out)

        # weights: slow (low distortion) dominates, apply per-variable mask
        w_time = (1.0 - distortion).repeat_interleave(D, dim=0).squeeze(dim=-1)
        mask_var = observed_mask.permute(0, 2, 1).reshape(B * D, L)
        weights = w_time * mask_var
        weight_sum = weights.sum(dim=1, keepdim=True)
        zero_mask = weight_sum <= self.eps
        if zero_mask.any():
            weights = torch.where(zero_mask, mask_var, weights)
            weight_sum = weights.sum(dim=1, keepdim=True) + self.eps
        weights = weights / (weight_sum + self.eps)

        spd_mean = self._weighted_frechet_mean(spd_seq, weights)
        spd_log = self._spd_logm(spd_mean)
        spd_feat = torch.nan_to_num(spd_log).reshape(B * D, -1)
        var_feat = self.spd_readout(spd_feat).view(B, D, -1)

        attn_out, _ = self.var_attn(var_feat, var_feat, var_feat)
        var_feat = self.var_ln(var_feat + attn_out)

        # forecasting head
        if time_steps_to_predict.dim() == 2:
            time_pred = time_steps_to_predict.unsqueeze(dim=-1).repeat(1, 1, D)
        else:
            time_pred = time_steps_to_predict
        B, Lp, _ = time_pred.shape
        time_pred = time_pred.unsqueeze(dim=-1)

        h = var_feat.unsqueeze(dim=1).repeat(1, Lp, 1, 1)
        h = torch.cat([h, time_pred], dim=-1)
        output = self.predict_decoder(h).unsqueeze(dim=0).squeeze(dim=-1)
        return output  # [1, B, Lp, D]
