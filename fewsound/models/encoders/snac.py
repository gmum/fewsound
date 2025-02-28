# https://github.com/hubertsiuzdak/snac/blob/main/snac/layers.py
# license in legal/

import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import torch
from einops import rearrange
from torch import nn

from fewsound.models.encoders.types import AudioEncoder

def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class LocalMHA(nn.Module):
    def __init__(self, dim=1024, window_size=32, dim_head=64, use_rotary_pos_emb=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        if use_rotary_pos_emb:
            self.rel_pos = SinusoidalEmbeddings(dim_head, scale_base=window_size // 2)
        else:
            self.rel_pos = None
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        residual = x
        x = self.norm(x.transpose(1, 2))
        windows = T // self.window_size
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads), (q, k, v))
        if self.rel_pos is not None:
            pos_emb, scale = self.rel_pos(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale)
        out = scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h w n d -> b (w n) (h d)")
        out = self.to_out(out)
        return out.transpose(1, 2) + residual


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # xpos related
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and scale_base is None), "scale base must be defined if using xpos"
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b ... r d", r=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale**-1
    if scale.ndim == 2:
        scale = scale[-q_len:, :]
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k


class Snac(AudioEncoder):
    def __init__(
        self,
        d_model=64,
        strides=[3, 3, 7, 7],
        depthwise=False,
        attn_window_size=32,
    ):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
        groups = d_model if depthwise else 1
        layers += [
            WNConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]
        self.block = nn.Sequential(*layers)
        self.ultimate_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=4,
                stride=4,
            ),
            nn.LeakyReLU(0.2)
        )

        self.ultimate_layer = nn.Linear(4096, 3296)

    def forward(self, x):
        x = x.unsqueeze(1)
        embedding = self.block(x)
        embedding = embedding.flatten(1, 2).unsqueeze(1)
        embedding = self.ultimate_conv(embedding).squeeze(1)
        embedding = self.ultimate_layer(embedding)
        return embedding

    def output_width(self, input_length):
        return 3296


class EncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

