import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# Per-frame MLP. 每帧独立映射,不依赖序列长度 / 邻居,完全因果 + 长度无关。
class DanceBridgeMLP(nn.Module):
    def __init__(self, input_dim, output_dim=2048, hidden_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x 任意 shape (..., input_dim) 都可以,因为是 per-position
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, input_dim)
        return self.fc2(self.act(self.fc1(x)))


# Music bridge 保留 v4 风格 MoE,但 n_bins=2(只前 2 个频段)
class Expert(nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048, num_heads=8, hidden_dim=2048):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.transformer_attn = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
        self.mamba = Mamba(d_model=output_dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        x = self.proj(x)
        attn_out, _ = self.transformer_attn(x, x, x)
        if attn_out.dim() == 2:
            attn_out = attn_out.unsqueeze(1)
        mamba_out = self.mamba(self.norm(attn_out))
        return mamba_out + attn_out


class MoEMusicBridgeN2(nn.Module):
    def __init__(self, input_dim, dance_cond_dim=264, output_dim=2048, num_heads=8, hidden_dim=2048, n_bins=2):
        super().__init__()
        self.n_bins = n_bins
        self.fs = 30.0
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.specialized_experts = nn.ModuleList([
            Expert(input_dim=dance_cond_dim, output_dim=output_dim, num_heads=num_heads, hidden_dim=hidden_dim)
            for _ in range(n_bins)
        ])
        self.universal_expert = Expert(input_dim=output_dim, output_dim=output_dim, num_heads=num_heads, hidden_dim=hidden_dim)
        self.fusion_proj = nn.Linear(output_dim * (n_bins + 1), output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        dance_cond = x[:, :, -264:]
        x_projected = self.input_proj(x)
        universal_out = self.universal_expert(x_projected)

        bands = bandpass_decompose(dance_cond, self.fs, self.n_bins)
        specialized_outs = []
        for i, band in enumerate(bands):
            band = band.squeeze(0) if band.dim() == 3 and band.size(0) == 1 else band
            if band.size(0) > x.size(1):
                band = band[:x.size(1)]
            elif band.size(0) < x.size(1):
                band = F.pad(band, (0, 0, 0, x.size(1) - band.size(0)))
            if band.dim() == 2:
                band = band.unsqueeze(0)
            spec_out = self.specialized_experts[i](band)
            specialized_outs.append(spec_out)
        moe_out = self.fusion_proj(torch.cat([universal_out] + specialized_outs, dim=-1))
        return moe_out


def split_freq_by_bins_rfft(T: int, fs: float = 30.0, n_bins: int = 2, device=None):
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs)
    max_freq = fs / 2.0
    freq_bins = torch.linspace(0, max_freq, n_bins + 1)
    masks = []
    for i in range(n_bins):
        left = freq_bins[i]
        right = freq_bins[i + 1]
        if i < n_bins - 1:
            mask = (freqs >= left) & (freqs < right)
        else:
            mask = (freqs >= left) & (freqs <= right)
        masks.append(mask.to(device) if device is not None else mask)
    return masks


def bandpass_decompose(x, fs: float = 30.0, n_bins: int = 2):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, T, D = x.shape
    device = x.device
    masks = split_freq_by_bins_rfft(T, fs=fs, n_bins=n_bins, device=device)
    X_freq = torch.fft.rfft(x, dim=1)
    out_bands = []
    for m in masks:
        masked = X_freq * m.view(1, -1, 1)
        out = torch.fft.irfft(masked, n=T, dim=1)
        out_bands.append(out)
    return out_bands
