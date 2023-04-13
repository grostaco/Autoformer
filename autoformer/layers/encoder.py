import torch.nn as nn
from torch import Tensor

from typing import Literal, Iterable

from .decomposition import SeriesDecomposition


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, attn_layer: nn.Module, *,
                 d_ff: int | None = None,
                 moving_avg=25,
                 dropout=.1,
                 activation: Literal['gelu'] | Literal['relu'] = 'relu'):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.attn_layer = attn_layer
        self.conv1 = nn.Conv1d(d_model, d_ff, 1, bias=False)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1, bias=False)

        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x: Tensor):
        y, weights = self.attn_layer(x, x, x)

        x = x + self.dropout(y)
        x, _ = self.decomp1(x)

        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        res, _ = self.decomp2(x + y)

        return res, weights


class Encoder(nn.Module):
    def __init__(self, attn_layers: Iterable[nn.Module], *,
                 norm_layer: nn.Module | None = None):
        super().__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: Tensor):
        weights = []
        for attn_layer in self.attn_layers:
            x, weight = attn_layer(x)
            weights.append(weight)

        if self.norm is not None:
            x = self.norm(x)

        return x, weights
