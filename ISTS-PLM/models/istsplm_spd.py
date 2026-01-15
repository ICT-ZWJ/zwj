import torch  # PyTorch基础库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数接口
from models.embed import TimeEmbedding, ValueEmbedding  # 时间与数值嵌入


class MultiLayerPerceptron(nn.Module):  # MLP module
    def __init__(self, input_dim, hidden_dim, dropout=0.0):  # 初始化
        super().__init__()  # 调用父类构造
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层线性
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二层线性
        self.dropout = nn.Dropout(p=dropout)  # dropout层

    def forward(self, x):  # 前向传播
        hidden = self.fc2(F.relu(self.fc1(x)))  # 线性->ReLU->线性
        return self.dropout(hidden)  # 应用dropout并返回


class DistortionAwareEmbedding(nn.Module):  # 失真感知嵌入
    def __init__(self, d_model, device=None, dropout=0.1, use_te=True):  # 初始化
        super().__init__()  # 父类初始化
        self.time_embedding = TimeEmbedding(d_model=d_model).to(device)  # 时间嵌入
        self.value_embedding = ValueEmbedding(c_in=2, d_model=d_model).to(device)  # 数值/掩码嵌入
        self.distortion_embedding = nn.Linear(1, d_model)  # 失真线性映射
        self.use_te = use_te  # 是否使用时间嵌入
        self.dropout = nn.Dropout(p=dropout)  # dropout层

    def forward(self, tt, x, x_mark, distortion):  # 前向
        # tt/x/x_mark: [B, L, D], distortion: [B, L, 1]  # 输入形状说明
        time_emb = self.time_embedding(tt.unsqueeze(dim=-1))  # 时间嵌入 [B, L, D, d_model]
        x_int = torch.cat([x.unsqueeze(dim=-1), x_mark.unsqueeze(dim=-1)], dim=-1)  # 拼接数值与掩码
        value_emb = self.value_embedding(x_int)  # 数值嵌入 [B, L, D, d_model]

        if self.use_te:  # 如果使用时间嵌入
            x_emb = x_mark.unsqueeze(dim=-1) * time_emb + value_emb  # 掩码加权时间嵌入并叠加数值嵌入
        else:  # 否则仅用数值嵌入
            x_emb = value_emb  # 直接使用数值嵌入

        dist_emb = self.distortion_embedding(distortion)  # 失真嵌入 [B, L, d_model]
        dist_emb = dist_emb.unsqueeze(dim=2).expand_as(x_emb)  # 扩展到与x_emb相同形状

        x_emb = x_emb + dist_emb  # 融合失真嵌入
        B, L, D, E = x_emb.shape  # 取出维度
        x_emb = x_emb.permute(0, 2, 1, 3).reshape(B * D, L, E)  # 变形为[B*D, L, E]
        return self.dropout(x_emb)  # dropout并返回


class istsplm_spd_forecast(nn.Module):  # 主模型类
    def __init__(self, opt):  # 初始化
        super().__init__()  # 父类初始化
        self.d_model = opt.d_model  # 隐藏维度
        self.spd_dim = getattr(opt, "spd_dim", 4)  # SPD矩阵维度
        self.alpha = getattr(opt, "dist_alpha", 1.0)  # 失真权重alpha
        self.beta = getattr(opt, "dist_beta", 1.0)  # 失真权重beta
        self.density_window = int(getattr(opt, "density_window", 500))  # local density window radius
        self.eps = 1e-6  # 数值稳定常数
        self.jitter = 1e-4  # SPD稳定抖动项

        self.enc_embedding = DistortionAwareEmbedding(  # 构建嵌入模块
            d_model=self.d_model,  # 嵌入维度
            device=opt.device,  # 设备
            dropout=opt.dropout,  # dropout
            use_te=True,  # 使用时间嵌入
        )  # 嵌入模块结束
        self.num_layer = getattr(opt, "num_layer", 6)  # 时序MLP层数
        self.temporal = nn.Sequential(  # 时序特征提取
            *[  # 解包列表
                MultiLayerPerceptron(self.d_model, self.d_model, dropout=opt.dropout)  # 单个MLP
                for _ in range(self.num_layer)  # 重复num_layer次
            ]  # 列表结束
        )  # Sequential结束
        self.spd_proj = nn.Linear(self.d_model, self.spd_dim * self.spd_dim)  # 投影到SPD矩阵元素
        self.spd_readout = nn.Linear(self.spd_dim * self.spd_dim, self.d_model)  # SPD特征回读

        self.var_attn = nn.MultiheadAttention(self.d_model, num_heads=1, batch_first=True)  # 变量注意力
        self.var_ln = nn.LayerNorm(self.d_model)  # 变量层归一化

        self.predict_decoder = nn.Sequential(  # 预测解码器
            nn.Linear(self.d_model + 1, self.d_model),  # 输入拼接时间标量
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Linear(self.d_model, self.d_model),  # 线性层
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Linear(self.d_model, 1),  # 输出单变量预测
        ).to(opt.device)  # 放到设备

    def _compute_distortion(self, tt, mask):  # 计算失真
        # tt: [B, L] or [B, L, D], mask: [B, L, D]  # 输入形状
        B, L, D = mask.shape  # 取出维度
        if L == 0:  # 无时间步
            return torch.zeros((B, 0, 1), device=mask.device)  # 返回空失真
        missing_rate = 1.0 - (mask.sum(dim=-1, keepdim=True) / (D + self.eps))  # 缺失率

        if tt.dim() == 2:  # 如果时间是[B, L]
            time_scalar = tt.unsqueeze(dim=-1)  # 扩展到[B, L, 1]
        else:  # 否则时间为[B, L, D]
            denom = mask.sum(dim=-1, keepdim=True) + self.eps  # 有效观测计数
            time_scalar = (tt * mask).sum(dim=-1, keepdim=True) / denom  # 加权平均时间

        delta = torch.zeros_like(time_scalar)  # 初始化时间间隔
        if L > 1:  # 多于1个时间步
            delta[:, 1:, :] = torch.abs(time_scalar[:, 1:, :] - time_scalar[:, :-1, :])  # 相邻差值
            max_delta = delta.max(dim=1, keepdim=True).values + self.eps  # 最大间隔
        else:  # 只有1个时间步
            max_delta = torch.ones_like(delta[:, :1, :]) + self.eps  # 避免除零
        delta_norm = delta / max_delta  # 归一化间隔

        distortion = self.alpha * missing_rate + self.beta * delta_norm  # 组合失真
        return torch.clamp(distortion, min=0.0, max=1.0)  # 限制到[0,1]

    def _compute_interval_irregularity(self, tt, mask):  # interval irregularity per variable
        B, L, D = mask.shape
        if L == 0:
            return torch.ones((B, D), device=mask.device)
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

    def _compute_local_density_fluct(self, mask):  # local density fluctuation
        B, L, D = mask.shape
        if L == 0:
            return torch.zeros((B, 0, D), device=mask.device)
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

    def _compute_weight_distortion(self, tt, mask):  # distortion for Frechet weights
        B, L, D = mask.shape
        if L == 0:
            return torch.zeros((B, 0, D), device=mask.device)
        irregularity = self._compute_interval_irregularity(tt, mask).unsqueeze(dim=1).expand(B, L, D)
        density_fluct = self._compute_local_density_fluct(mask)
        distortion = 0.5 * irregularity + 0.5 * density_fluct
        return torch.clamp(distortion, min=0.0, max=1.0)

    def _spd_from_features(self, features):  # 从特征构造SPD矩阵
        # features: [B*D, L, d_model] -> [B*D, L, m, m]  # 形状说明
        BDL, L, _ = features.shape  # 展开维度
        m = self.spd_dim  # SPD矩阵维度
        raw = self.spd_proj(features).view(BDL, L, m, m)  # 投影并reshape
        raw = torch.tanh(raw)  # 限幅
        sym = 0.5 * (raw + raw.transpose(-1, -2))  # 对称化
        eye = torch.eye(m, device=features.device).view(1, 1, m, m)  # 单位矩阵
        spd = sym @ sym.transpose(-1, -2) + self.jitter * eye  # 构造SPD
        return spd  # 返回SPD序列

    def _spd_logm(self, spd):  # SPD矩阵对数映射
        # spd: [..., m, m]  # 输入形状
        eigvals, eigvecs = torch.linalg.eigh(spd)  # 特征分解
        eigvals = torch.clamp(eigvals, min=self.jitter, max=1e4)  # 截断特征值
        logvals = torch.log(eigvals)  # 取对数
        out = eigvecs @ torch.diag_embed(logvals) @ eigvecs.transpose(-1, -2)  # 重构logm
        return torch.nan_to_num(out)  # 清理NaN/Inf

    def _spd_expm(self, tangent):  # SPD矩阵指数映射
        eigvals, eigvecs = torch.linalg.eigh(tangent)  # 特征分解
        eigvals = torch.clamp(eigvals, min=-20.0, max=20.0)  # 控制指数范围
        expvals = torch.exp(eigvals)  # 指数
        out = eigvecs @ torch.diag_embed(expvals) @ eigvecs.transpose(-1, -2)  # 重构expm
        return torch.nan_to_num(out)  # 清理NaN/Inf

    def _weighted_frechet_mean(self, spd_seq, weights):  # 加权Frechet均值
        # spd_seq: [B*D, L, m, m], weights: [B*D, L]  # 输入形状
        BDL, L, m, _ = spd_seq.shape  # 取维度
        weights = weights.unsqueeze(dim=-1).unsqueeze(dim=-1)  # 变为可广播
        logm = self._spd_logm(spd_seq)  # 映射到切空间
        mean_log = (weights * logm).sum(dim=1)  # 加权求和
        return self._spd_expm(mean_log)  # 映射回SPD流形

    def forecasting(self, time_steps_to_predict, observed_data, observed_tp, observed_mask):  # 预测函数
        # observed_data/mask: [B, L, D], observed_tp: [B, L] or [B, L, D]  # 输入形状
        B, L, D = observed_data.shape  # 维度

        if observed_tp.dim() == 2:  # 如果时间缺少变量维
            observed_tp = observed_tp.unsqueeze(dim=-1).repeat(1, 1, D)  # 扩展到[B, L, D]

        if L == 0:  # 无历史长度
            if time_steps_to_predict.dim() == 2:  # 预测时间为[B, Lp]
                Lp = time_steps_to_predict.size(1)  # 预测长度
            else:  # 预测时间为[B, Lp, D]
                Lp = time_steps_to_predict.size(1)  # 预测长度
            return torch.zeros((1, B, Lp, D), device=observed_data.device)  # 返回零预测

        distortion = self._compute_distortion(observed_tp, observed_mask)  # 计算失真
        emb = self.enc_embedding(observed_tp, observed_data, observed_mask, distortion)  # 计算嵌入

        seq_out = self.temporal(emb)  # [B*D, L, d_model]  # 时序特征
        spd_seq = self._spd_from_features(seq_out)  # 构建SPD序列

        # weights: slow (low distortion) dominates, apply per-variable mask  # 权重说明
        weight_distortion = self._compute_weight_distortion(observed_tp, observed_mask)
        w_time = (1.0 - weight_distortion).permute(0, 2, 1).reshape(B * D, L)
        mask_var = observed_mask.permute(0, 2, 1).reshape(B * D, L)  # 变量掩码
        weights = w_time * mask_var  # 合并权重
        weight_sum = weights.sum(dim=1, keepdim=True)  # 权重和
        zero_mask = weight_sum <= self.eps  # 检查全零
        if zero_mask.any():  # 如果存在全零
            weights = torch.where(zero_mask, mask_var, weights)  # 退化为掩码
            weight_sum = weights.sum(dim=1, keepdim=True) + self.eps  # 重新求和
        weights = weights / (weight_sum + self.eps)  # 归一化权重

        spd_mean = self._weighted_frechet_mean(spd_seq, weights)  # 加权均值SPD
        spd_log = self._spd_logm(spd_mean)  # log映射
        spd_feat = torch.nan_to_num(spd_log).reshape(B * D, -1)  # 展平特征
        var_feat = self.spd_readout(spd_feat).view(B, D, -1)  # 回读到d_model

        attn_out, _ = self.var_attn(var_feat, var_feat, var_feat)  # 变量注意力
        var_feat = self.var_ln(var_feat + attn_out)  # 残差+归一化

        # forecasting head  # 预测头
        if time_steps_to_predict.dim() == 2:  # 预测时间为[B, Lp]
            time_pred = time_steps_to_predict.unsqueeze(dim=-1).repeat(1, 1, D)  # 扩展到[B, Lp, D]
        else:  # 已经是[B, Lp, D]
            time_pred = time_steps_to_predict  # 直接使用
        B, Lp, _ = time_pred.shape  # 预测维度
        time_pred = time_pred.unsqueeze(dim=-1)  # 增加标量维

        h = var_feat.unsqueeze(dim=1).repeat(1, Lp, 1, 1)  # 广播变量特征到时间
        h = torch.cat([h, time_pred], dim=-1)  # 拼接时间标量
        output = self.predict_decoder(h).unsqueeze(dim=0).squeeze(dim=-1)  # 解码输出
        return output  # [1, B, Lp, D]  # 返回预测
