import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.embed import TimeEmbedding, ValueEmbedding

# ==========================================
# Components from tPatchGNN
# ==========================================

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x (B, F, N, M)
        # A (B, M, N, N)
        x = torch.einsum('bfnm,bmnv->bfvm', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # x (B, F, N, M)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        return F.relu(h)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (Batch, Seq_len, D_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

# ==========================================
# Components from ISTS-PLM
# ==========================================

class DistortionAwareEmbedding(nn.Module):
    def __init__(self, d_model, device=None, dropout=0.1, use_te=True):
        super().__init__()
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)
        self.distortion_embedding = nn.Linear(1, d_model)
        self.use_te = use_te
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tt, x, x_mark, distortion):
        # tt, x, x_mark: [B, L, D]
        # distortion: [B, L, 1] or [B, L, D]
        
        # Ensure time embedding input is correct size
        if tt.dim() == 2:
            tt_emb_in = tt.unsqueeze(-1)
        else:
            tt_emb_in = tt.unsqueeze(-1) # [B, L, D, 1] - TimeEmbedding expects specific shape

        # Adjust for TimeEmbedding implementation details
        # Assuming TimeEmbedding takes (Batch, Length, Dim, 1) or similar. 
        # Here we simplify: expand tt to match x shape logic
        
        # Based on ISTS code:
        # time_emb = self.time_embedding(tt.unsqueeze(dim=-1))
        # Note: In ISTS, tt might be [B, L]. If [B, L, D], unsqueeze makes it [B, L, D, 1].
        
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1)) # [B, L, D, d_model]
        
        x_int = torch.cat([x.unsqueeze(dim=-1), x_mark.unsqueeze(dim=-1)], dim=-1)
        value_emb = self.value_embedding(x_int)

        if self.use_te:
            x_emb = x_mark.unsqueeze(dim=-1) * time_emb + value_emb
        else:
            x_emb = value_emb

        dist_emb = self.distortion_embedding(distortion)
        if dist_emb.dim() == 3: # [B, L, d_model]
             dist_emb = dist_emb.unsqueeze(2).expand_as(x_emb)
        elif dist_emb.dim() == 4: # [B, L, D, d_model]
             pass

        x_emb = x_emb + dist_emb
        B_size, L_size, D_size, E_size = x_emb.shape
        # Flatten Batch and Var dim for processing: [B*D, L, E]
        x_emb = x_emb.permute(0, 2, 1, 3).reshape(B_size * D_size, L_size, E_size)
        return self.dropout(x_emb)

# ==========================================
# Merged Model: ISTS-PLM with tPatchGNN Backbone
# ==========================================

class ISTS_tPatch(nn.Module):
    def __init__(self, args):
        super(ISTS_tPatch, self).__init__()
        
        # Args extraction
        self.device = args.device
        self.d_model = args.hid_dim  # consistent with tPatchGNN hid_dim
        self.spd_dim = getattr(args, "spd_dim", 4)
        self.alpha = getattr(args, "dist_alpha", 1.0)
        self.beta = getattr(args, "dist_beta", 1.0)
        self.density_window = int(getattr(args, "density_window", 5))
        self.eps = 1e-6
        self.jitter = 1e-4
        
        # tPatchGNN specific args
        self.n_layer = args.nlayer
        self.nhead = args.nhead
        self.tf_layer = args.tf_layer
        self.dropout = args.dropout
        self.N = args.ndim # Number of variables/nodes
        
        # 1. Embedding Layer (from ISTS)
        self.enc_embedding = DistortionAwareEmbedding(
            d_model=self.d_model,
            device=self.device,
            dropout=self.dropout,
            use_te=True
        )

        # 2. Backbone: tPatchGNN Components (Modified for Sequence-level processing)
        
        ## TTCN Replacement
        # As requested, this replaces the simple MLP in ISTS.
        # We act on the sequence level, so this is essentially a pointwise feature refinement.
        self.ttcn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True)
        )

        ## Transformer (Temporal modeling over Sequence Length L)
        self.ADD_PE = PositionalEncoding(self.d_model)
        self.transformer_encoder = nn.ModuleList()
        for _ in range(self.n_layer):
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, num_layers=self.tf_layer))

        ## GNN (Spatial modeling - Sequence Level)
        # We will treat Time Steps L as the 'M' dimension in tPatchGNN's GCN logic
        # This allows time-varying graph structures.
        self.nodevec_dim = args.node_dim
        self.nodevec1 = nn.Parameter(torch.randn(self.N, self.nodevec_dim).to(self.device), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(self.nodevec_dim, self.N).to(self.device), requires_grad=True)
        
        self.nodevec_linear1 = nn.ModuleList()
        self.nodevec_linear2 = nn.ModuleList()
        self.nodevec_gate1 = nn.ModuleList()
        self.nodevec_gate2 = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        self.supports_len = 1 
        
        for _ in range(self.n_layer):
            self.nodevec_linear1.append(nn.Linear(self.d_model, self.nodevec_dim))
            self.nodevec_linear2.append(nn.Linear(self.d_model, self.nodevec_dim))
            self.nodevec_gate1.append(nn.Sequential(
                nn.Linear(self.d_model + self.nodevec_dim, 1),
                nn.Tanh(), nn.ReLU()
            ))
            self.nodevec_gate2.append(nn.Sequential(
                nn.Linear(self.d_model + self.nodevec_dim, 1),
                nn.Tanh(), nn.ReLU()
            ))
            self.gconv.append(gcn(self.d_model, self.d_model, self.dropout, support_len=self.supports_len, order=args.hop))

        # 3. Aggregation Components (Euclidean Main Path)
        # SPD components removed for ablation study.

        # 4. Post-Aggregation Variable Relationship Modeling (Replaces var_attn with GCN)
        # Separate GCN parameters for post-aggregation processing
        # Here M=1 (aggregated features), N=ndim
        self.post_agg_nodevec1 = nn.Parameter(torch.randn(self.N, self.nodevec_dim).to(self.device), requires_grad=True)
        self.post_agg_nodevec2 = nn.Parameter(torch.randn(self.nodevec_dim, self.N).to(self.device), requires_grad=True)
        
        self.post_agg_nodevec_linear1 = nn.Linear(self.d_model, self.nodevec_dim)
        self.post_agg_nodevec_linear2 = nn.Linear(self.d_model, self.nodevec_dim)
        self.post_agg_nodevec_gate1 = nn.Sequential(
                nn.Linear(self.d_model + self.nodevec_dim, 1),
                nn.Tanh(), nn.ReLU()
            )
        self.post_agg_nodevec_gate2 = nn.Sequential(
                nn.Linear(self.d_model + self.nodevec_dim, 1),
                nn.Tanh(), nn.ReLU()
            )
        self.post_agg_gcn = gcn(self.d_model, self.d_model, self.dropout, support_len=1, order=args.hop)
        self.var_ln = nn.LayerNorm(self.d_model)

        # 5. Decoder
        self.predict_decoder = nn.Sequential(
            nn.Linear(self.d_model + 1, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 1),
        ).to(self.device)

    # --- Helper methods from ISTS ---
    def _compute_distortion(self, tt, mask):
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

    def _compute_interval_irregularity(self, tt, mask):
        B, L, D = mask.shape
        if L == 0: return torch.ones((B, D), device=mask.device)
        mask_bool = mask > 0
        prev_time = torch.zeros((B, D), device=mask.device)
        prev_seen = torch.zeros((B, D), dtype=torch.bool, device=mask.device)
        log_sum = torch.zeros((B, D), device=mask.device)
        log_sq_sum = torch.zeros((B, D), device=mask.device)
        count = torch.zeros((B, D), device=mask.device)
        for t in range(L):
            cur_mask = mask_bool[:, t, :]
            cur_time = tt[:, t, :]
            has_prev = prev_seen & cur_mask
            if has_prev.any():
                delta = (cur_time - prev_time).abs()
                log_interval = torch.log(delta + self.eps)
                log_sum = log_sum + torch.where(has_prev, log_interval, torch.zeros_like(log_interval))
                log_sq_sum = log_sq_sum + torch.where(has_prev, log_interval * log_interval, torch.zeros_like(log_interval))
                count = count + has_prev.float()
            prev_time = torch.where(cur_mask, cur_time, prev_time)
            prev_seen = prev_seen | cur_mask
        count_safe = torch.clamp(count, min=1.0)
        mean = log_sum / count_safe
        var = log_sq_sum / count_safe - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        irregularity = torch.tanh(std)
        irregularity = torch.where(count < 1.0, torch.ones_like(irregularity), irregularity)
        return irregularity

    def _compute_local_density_fluct(self, mask):
        B, L, D = mask.shape
        if L == 0: return torch.zeros((B, 0, D), device=mask.device)
        w = max(self.density_window, 0)
        mask_float = (mask > 0).float()
        mask_conv = mask_float.permute(0, 2, 1).reshape(B * D, 1, L)
        kernel = torch.ones((1, 1, 2 * w + 1), device=mask.device)
        density = F.conv1d(mask_conv, kernel, padding=w)
        window_size = F.conv1d(torch.ones_like(density), kernel, padding=w)
        density_sum = F.conv1d(density, kernel, padding=w)
        density_sq_sum = F.conv1d(density * density, kernel, padding=w)
        mean = density_sum / (window_size + self.eps)
        var = density_sq_sum / (window_size + self.eps) - mean * mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        density_std = torch.tanh(std)
        return density_std.view(B, D, L).permute(0, 2, 1)

    def _compute_weight_distortion(self, tt, mask):
        B, L, D = mask.shape
        if L == 0: return torch.zeros((B, 0, D), device=mask.device)
        irregularity = self._compute_interval_irregularity(tt, mask).unsqueeze(dim=1).expand(B, L, D)
        density_fluct = self._compute_local_density_fluct(mask)
        distortion = 0.5 * irregularity + 0.5 * density_fluct
        return torch.clamp(distortion, min=0.0, max=1.0)

    # --- SPD Helpers (Auxiliary Path) ---
    def _spd_from_features(self, features):
        BDL, L, _ = features.shape
        m = self.spd_dim
        raw = self.spd_proj(features).view(BDL, L, m, m)
        raw = torch.tanh(raw)
        sym = 0.5 * (raw + raw.transpose(-1, -2))
        eye = torch.eye(m, device=features.device).view(1, 1, m, m)
        # Jitter for stability
        spd = sym @ sym.transpose(-1, -2) + (self.jitter + 1e-3) * eye
        return spd

    def _spd_logm(self, spd):
        spd_d = spd.double()
        spd_d = 0.5 * (spd_d + spd_d.transpose(-1, -2))
        try:
            eigvals, eigvecs = torch.linalg.eigh(spd_d)
        except torch._C._LinAlgError:
            try:
                # Stronger Jitter Fallback
                spd_d = spd_d + 1e-2 * torch.eye(spd_d.shape[-1], device=spd.device, dtype=torch.float64)
                eigvals, eigvecs = torch.linalg.eigh(spd_d)
            except torch._C._LinAlgError:
                # Return None to indicate failure, will be skipped in loss
                return None
            
        eigvals = torch.clamp(eigvals, min=self.jitter, max=1e4)
        logvals = torch.log(eigvals)
        out = eigvecs @ torch.diag_embed(logvals) @ eigvecs.transpose(-1, -2)
        return torch.nan_to_num(out.float())

    def _weighted_frechet_mean(self, spd_seq, weights):
        # spd_seq: [B*D, L, m, m]
        # weights: [B*D, L]
        weights = weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
        logm = self._spd_logm(spd_seq)
        
        if logm is None: # Fallback if logm failed
            return None
            
        mean_log = (weights * logm).sum(dim=1)
        # Return mean in Log domain directly for loss calculation
        return mean_log # [B*D, m, m]

    # --- Backbone Logic (Sequence Level) ---
    
    def run_backbone(self, x_emb):
        # x_emb: [B*D, L, d_model]
        BD, L, FeatDim = x_emb.shape
        B = BD // self.N
        
        # Reshape to [B, N, L, F] to separate variables
        x = x_emb.view(B, self.N, L, FeatDim)
        
        # 1. Intra-variable modeling (TTCN Replacement)
        # Apply transformation to each time step independently
        # Input: [B, N, L, F] -> Flatten: [B*N*L, F]
        x = self.ttcn(x)
        
        # 2. Inter-variable & Temporal Modeling
        # In tPatchGNN, we iterate over layers
        for layer in range(self.n_layer):
            if layer > 0:
                x_last = x.clone()
            
            # 2.1 Temporal: Transformer
            # Shape: [B*N, L, F] for Transformer
            x_temporal = x.view(B * self.N, L, FeatDim)
            x_temporal = self.ADD_PE(x_temporal)
            x_temporal = self.transformer_encoder[layer](x_temporal)
            x = x_temporal.view(B, self.N, L, FeatDim)
            
            # 2.2 Spatial: GNN
            # We treat L (Time) as the 'M' (Patch/Snapshot) dimension in GNN
            # x shape for GNN logic should be [B, N, L, F]
            # tPatchGNN logic: 
            # nodevec1 shape exp: [B, L, N, D]
            # nodevec2 shape exp: [B, L, D, N]
            
            # Expand nodevecs: repeat for L timesteps
            nodevec1 = self.nodevec1.view(1, 1, self.N, self.nodevec_dim).repeat(B, L, 1, 1)
            nodevec2 = self.nodevec2.view(1, 1, self.nodevec_dim, self.N).repeat(B, L, 1, 1)
            
            # x for gate: [B, N, L, F] -> permute to [B, L, N, F] (Batch, M=L, N, F)
            x_perm = x.permute(0, 2, 1, 3) 
            
            x_gate1 = self.nodevec_gate1[layer](torch.cat([x_perm, nodevec1], dim=-1))
            x_gate2 = self.nodevec_gate2[layer](torch.cat([x_perm, nodevec2.permute(0,1,3,2)], dim=-1))
            
            x_p1 = x_gate1 * self.nodevec_linear1[layer](x_perm)
            x_p2 = x_gate2 * self.nodevec_linear2[layer](x_perm)
            
            nodevec1 = nodevec1 + x_p1
            nodevec2 = nodevec2 + x_p2.permute(0, 1, 3, 2) # [B, L, D, N]
            
            adp = F.softmax(F.relu(torch.matmul(nodevec1, nodevec2)), dim=-1) # [B, L, N, N]
            supports = [adp]
            
            # GNN Forward
            # gconv expects x: (B, F, N, M). Here M is L.
            # Our x is [B, N, L, F].
            # Permute to [B, F, N, L]
            x_gcn_in = x.permute(0, 3, 1, 2)
            x_gcn_out = self.gconv[layer](x_gcn_in, supports) # [B, F, N, L]
            
            # Permute back to [B, N, L, F]
            x = x_gcn_out.permute(0, 2, 3, 1) 
            
            if layer > 0:
                x = x + x_last

            # Output: [B, N, L, F]
        # Reshape back to [B*N, L, F] for SPD processing
        x_out = x.reshape(B * self.N, L, FeatDim)
        
        return x_out

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):
        B, L, D = observed_data.shape
        self.batch_size = B
        
        if observed_tp.dim() == 2:
            observed_tp = observed_tp.unsqueeze(dim=-1).repeat(1, 1, D)

        if L == 0:
             # Basic zero return handling
            if time_steps_to_predict.dim() == 2: Lp = time_steps_to_predict.size(1)
            else: Lp = time_steps_to_predict.size(1)
            return torch.zeros((1, B, Lp, D), device=observed_data.device)

        # 1. Embedding
        distortion = self._compute_distortion(observed_tp, observed_mask)
        emb = self.enc_embedding(observed_tp, observed_data, observed_mask, distortion)
        # emb: [B*D, L, d_model]

        # 2. Backbone Processing (Sequence Level)
        seq_out = self.run_backbone(emb) # [B*D, L, d_model]
        
        # 3. Feature Aggregation
        
        # Calculate weights based on distortion
        weight_distortion = self._compute_weight_distortion(observed_tp, observed_mask)
        w_time = (1.0 - weight_distortion).permute(0, 2, 1).reshape(B * D, L)
        mask_var = observed_mask.permute(0, 2, 1).reshape(B * D, L)
        weights = w_time * mask_var
        weight_sum = weights.sum(dim=1, keepdim=True)
        zero_mask = weight_sum <= self.eps
        if zero_mask.any():
            weights = torch.where(zero_mask, mask_var, weights)
            weight_sum = weights.sum(dim=1, keepdim=True) + self.eps
        weights = weights / (weight_sum + self.eps) # [B*D, L]

        # A. Main Path: Weighted Mean in Euclidean Space
        agg_feat = torch.sum(seq_out * weights.unsqueeze(-1), dim=1) # [B*D, d_model]
        
        # B. Auxiliary Path: SPD Constraint Removed for Ablation

        var_feat = agg_feat.view(B, D, -1) # [B, D, d_model]

        # 4. Post-Aggregation Variable Modeling (GCN)
        # var_feat: [B, D, F] -> Treat D as N, F as F, M=1 (snapshot)
        # GCN expects x: [B, F, N, M] -> [B, F, D, 1]
        x_post = var_feat.permute(0, 2, 1).unsqueeze(-1)
        
        # Calculate dynamic adjacency for this snapshot
        # nodevecs: [N, D] -> expand to [B, 1, N, D]
        n1 = self.post_agg_nodevec1.view(1, 1, self.N, self.nodevec_dim).repeat(B, 1, 1, 1)
        n2 = self.post_agg_nodevec2.view(1, 1, self.nodevec_dim, self.N).repeat(B, 1, 1, 1)
        
        # x for gate: [B, D, F] -> [B, 1, D, F] (M=1)
        x_perm = var_feat.unsqueeze(1) 
        
        g1 = self.post_agg_nodevec_gate1(torch.cat([x_perm, n1], dim=-1))
        g2 = self.post_agg_nodevec_gate2(torch.cat([x_perm, n2.permute(0,1,3,2)], dim=-1))
        
        p1 = g1 * self.post_agg_nodevec_linear1(x_perm)
        p2 = g2 * self.post_agg_nodevec_linear2(x_perm)
        
        n1 = n1 + p1
        n2 = n2 + p2.permute(0, 1, 3, 2)
        
        adp = F.softmax(F.relu(torch.matmul(n1, n2)), dim=-1) # [B, 1, N, N]
        
        x_gcn_out = self.post_agg_gcn(x_post, [adp]) # [B, F, N, 1]
        
        var_feat = x_gcn_out.squeeze(-1).permute(0, 2, 1) # [B, N, F]
        var_feat = self.var_ln(var_feat)

        # 5. Prediction
        if time_steps_to_predict.dim() == 2:
            time_pred = time_steps_to_predict.unsqueeze(dim=-1).repeat(1, 1, D)
        else:
            time_pred = time_steps_to_predict
        
        B, Lp, _ = time_pred.shape
        time_pred = time_pred.unsqueeze(dim=-1)

        h = var_feat.unsqueeze(dim=1).repeat(1, Lp, 1, 1)
        h = torch.cat([h, time_pred], dim=-1)
        output = self.predict_decoder(h).unsqueeze(dim=0).squeeze(dim=-1)
        
        return output