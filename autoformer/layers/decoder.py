import torch.nn as nn
from torch import Tensor

from typing import Literal, Iterable

from .decomposition import SeriesDecomposition


class DecoderLayer(nn.Module):
    def __init__(self, self_attn: nn.Module, cross_attn: nn.Module, d_model: int, c_out: int, *,
                 d_ff: int | None = None,
                 moving_avg=25,
                 dropout=.1,
                 activation: Literal['gelu'] | Literal['relu'] = 'relu'):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.self_attn = self_attn
        self.cross_attn = cross_attn

        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)

        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(
            d_model, c_out, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)

        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x: Tensor, cross: Tensor):
        x = x + self.dropout(self.self_attn(x, x, x, needs_weight=False))
        x, trend1 = self.decomp1(x)

        x = x + self.dropout(self.cross_attn(x, cross,
                             cross, needs_weight=False))
        x, trend2 = self.decomp2(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(
            residual_trend.permute(0, 2, 1)).transpose(1, 2)

        return x, residual_trend


class Decoder(nn.Module):
    def __init__(self, layers: Iterable[nn.Module], *,
                 norm_layer: nn.Module | None = None,
                 projection: nn.Module | None = None):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: Tensor, cross: Tensor, trend: Tensor):
        for layer in self.layers:
            x, residual_trend = layer(x, cross)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, trend
