"""Transformer encoder used by the released RetrievalNet checkpoint."""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


class StylizationBlock(nn.Module):
    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(time_embed_dim, 2 * latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, hidden, embedding):
        scale, shift = torch.chunk(self.emb_layers(embedding).unsqueeze(1), 2, dim=2)
        hidden = self.norm(hidden) * (1 + scale) + shift
        return self.out_layers(hidden)


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        dropout: float,
        time_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        emb: Optional[torch.Tensor] = None,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        batch, frames, channels = x.shape
        heads = self.num_heads
        normalized = self.norm(x)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)
        if src_mask is not None:
            key = key + (1 - src_mask) * -1000000
            value = value * src_mask
        query = F.softmax(query.view(batch, frames, heads, -1), dim=-1)
        key = F.softmax(key.view(batch, frames, heads, -1), dim=1)
        value = value.view(batch, frames, heads, -1)
        attention = torch.einsum("bnhd,bnhl->bhdl", key, value)
        output = torch.einsum("bnhd,bhdl->bnhl", query, attention).reshape(
            batch, frames, channels
        )
        if self.time_embed_dim is None:
            return x + output
        return x + self.proj_out(output, emb)


class FFN(nn.Module):
    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        output = self.linear2(
            self.dropout(self.activation(self.linear1(self.norm(x))))
        )
        return x + output


class EncoderLayer(nn.Module):
    def __init__(self, latent_dim=256, num_heads=4, ff_size=1024, dropout=0):
        super().__init__()
        self.sa_block = EfficientSelfAttention(latent_dim, num_heads, dropout)
        self.ffn = FFN(latent_dim, ff_size, dropout)

    def forward(self, **kwargs):
        hidden = self.sa_block(**kwargs)
        kwargs.update({"x": hidden})
        return self.ffn(**kwargs)


class TransEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_feats = cfg.input_feats
        self.max_seq_len = cfg.max_seq_len
        self.latent_dim = cfg.latent_dim
        self.proj_in = nn.Linear(cfg.input_feats, cfg.latent_dim)
        self.embedding = nn.Parameter(torch.randn(cfg.max_seq_len, cfg.latent_dim))
        self.output_var = cfg.output_var
        self.blocks = nn.ModuleList(
            [
                EncoderLayer(
                    latent_dim=cfg.latent_dim,
                    num_heads=cfg.num_heads,
                    ff_size=cfg.ff_size,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        output_dim = cfg.latent_dim * 2 if cfg.output_var else cfg.latent_dim
        projection = nn.Linear(cfg.latent_dim, output_dim)
        self.proj_out = zero_module(projection) if cfg.zero_output else projection
        self.cls_token = (
            nn.Parameter(torch.randn(cfg.latent_dim)) if cfg.cls_token else None
        )

    def forward(self, motion, motion_mask):
        batch, frames = motion.shape[:2]
        if self.input_feats == 132:
            motion = motion.reshape(batch, frames, 2, -1)[..., :66].reshape(
                batch, frames, -1
            )
        elif self.input_feats == 516:
            motion = motion.reshape(batch, frames, 2, -1)[..., :-4].reshape(
                batch, frames, -1
            )
        elif self.input_feats not in (121, 10, 66, 55, 264):
            raise ValueError(f"Unsupported input feature size: {self.input_feats}")

        hidden = self.proj_in(motion)
        hidden = hidden + self.embedding.view(1, self.max_seq_len, -1)[:, :frames, :]
        if self.cls_token is not None:
            cls_token = self.cls_token.view(1, 1, -1).repeat(batch, 1, 1)
            hidden = torch.cat((cls_token, hidden), dim=1)
            motion_mask = torch.cat(
                (torch.ones(batch, 1).type_as(motion), motion_mask), dim=1
            )
        for block in self.blocks:
            hidden = block(x=hidden, src_mask=motion_mask.view(batch, -1, 1))
        hidden = hidden[:, 0, :] if self.cls_token is not None else hidden.mean(dim=1)
        output = self.proj_out(hidden)
        mean = output[:, : self.latent_dim].contiguous()
        if self.output_var:
            return mean, output[:, self.latent_dim :].contiguous()
        return mean
