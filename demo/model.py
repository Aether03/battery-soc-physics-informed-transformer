import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """Explicit Q/K/V/Output projections (not nn.MultiheadAttention's fused
    in_proj) to give an exact 1:1 parameter correspondence with MATLAB's
    selfAttentionLayer, which stores QueryWeights/KeyWeights/ValueWeights/
    OutputWeights as four separate learnable arrays."""

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch, seq, dim]
        b, t, d = x.shape
        q = self.query(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # [batch, heads, seq, head_dim]
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out(out)


def sinusoidal_positional_encoding(num_channels, seq_len, scale, device, dtype):
    """Reimplements PositionalEncodingLayer.m exactly: per-channel-pair
    sin/cos, scaled, added (not concatenated)."""
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    enc = torch.zeros(num_channels, seq_len, device=device, dtype=dtype)
    for i in range(0, num_channels, 2):
        exponent = i / num_channels
        div = 10000 ** exponent
        enc[i, :] = torch.sin(pos / div)
        if i + 1 < num_channels:
            enc[i + 1, :] = torch.cos(pos / div)
    return enc * scale  # [channels, seq]


class HAPITTransformer(nn.Module):
    """Port of HAPIT8v6.m's layer graph (see PyTorch_port/Documents/
    05_Model_Architecture_and_Weight_Schema.md for the source-derived spec).
    Forward pass takes x: [batch, num_features, seq_len] and returns
    [batch, 2, seq_len] (raw SOC correction, raw OCV correction) — matching
    the MATLAB net's 'net_out' layer exactly, before any post-processing."""

    def __init__(self, num_features=5, num_filters=48, attention_dim=64,
                 num_heads=4, ffn_dim=128, dropout=0.05, pos_enc_scale=0.1):
        super().__init__()
        self.pos_enc_scale = pos_enc_scale
        self.attention_dim = attention_dim

        self.conv_k3 = nn.Conv1d(num_features, num_filters, kernel_size=3, padding="same")
        self.conv_k5 = nn.Conv1d(num_features, num_filters, kernel_size=5, padding="same")
        self.conv_k9 = nn.Conv1d(num_features, num_filters, kernel_size=9, padding="same")
        self.ln_cnn = nn.LayerNorm(num_filters * 3)
        self.drop_cnn = nn.Dropout(dropout)
        self.proj = nn.Linear(num_filters * 3, attention_dim)

        self.self_att_1 = SelfAttentionBlock(attention_dim, num_heads)
        self.drop_att_1 = nn.Dropout(dropout)
        self.ln_att_1 = nn.LayerNorm(attention_dim)
        self.ffn1_1 = nn.Linear(attention_dim, ffn_dim)
        self.ffn2_1 = nn.Linear(ffn_dim, attention_dim)
        self.drop_ffn_1 = nn.Dropout(dropout)
        self.ln_ffn_1 = nn.LayerNorm(attention_dim)

        self.self_att_2 = SelfAttentionBlock(attention_dim, num_heads)
        self.drop_att_2 = nn.Dropout(dropout)
        self.ln_att_2 = nn.LayerNorm(attention_dim)
        self.ffn1_2 = nn.Linear(attention_dim, ffn_dim)
        self.ffn2_2 = nn.Linear(ffn_dim, attention_dim)
        self.drop_ffn_2 = nn.Dropout(dropout)
        self.ln_ffn_2 = nn.LayerNorm(attention_dim)

        self.net_out = nn.Linear(attention_dim, 2)

    def forward(self, x):
        # x: [batch, num_features, seq_len]
        c3 = self.conv_k3(x)
        c5 = self.conv_k5(x)
        c9 = self.conv_k9(x)
        cat = torch.cat([c3, c5, c9], dim=1)  # [batch, 144, seq_len]

        h = cat.transpose(1, 2)  # [batch, seq_len, 144] -- LayerNorm/Linear operate on last dim
        h = self.ln_cnn(h)
        h = F.relu(h)
        h = self.drop_cnn(h)
        h = self.proj(h)  # [batch, seq_len, attention_dim]

        seq_len = h.shape[1]
        pos = sinusoidal_positional_encoding(
            self.attention_dim, seq_len, self.pos_enc_scale, h.device, h.dtype
        )
        h = h + pos.transpose(0, 1)  # broadcast [seq_len, dim] over batch

        # Block 1 (post-norm: add residual, then normalize)
        att_out = self.drop_att_1(self.self_att_1(h))
        h = self.ln_att_1(att_out + h)
        ffn_out = self.drop_ffn_1(self.ffn2_1(F.relu(self.ffn1_1(h))))
        h = self.ln_ffn_1(ffn_out + h)

        # Block 2
        att_out = self.drop_att_2(self.self_att_2(h))
        h = self.ln_att_2(att_out + h)
        ffn_out = self.drop_ffn_2(self.ffn2_2(F.relu(self.ffn1_2(h))))
        h = self.ln_ffn_2(ffn_out + h)

        out = self.net_out(h)  # [batch, seq_len, 2]
        return out.transpose(1, 2)  # [batch, 2, seq_len] -- matches MATLAB's CBT-derived output layout
