import torch
import torch.nn as nn
from torch import Tensor

import math
from enum import IntEnum
from typing import Literal


class FixedEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()

        w = torch.zeros(c_in, d_model, dtype=torch.float)
        position = torch.arange(0, c_in, dtype=torch.float).unsqueeze(-1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float)
                    * -(math.log(10000.) / d_model)).exp()

        w[:, ::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model, _weight=w, _freeze=True)

    def forward(self, X: Tensor):
        # .detach() might not be necessary
        return self.emb(X).detach()


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()

        self.conv = nn.Conv1d(c_in, d_model, kernel_size=3,
                              padding=1, padding_mode='circular', bias=False)
        self.apply(TokenEmbedding.init_weights)

    @staticmethod
    def init_weights(module: nn.Module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(
                module.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, X: Tensor):
        X = self.conv(X.permute(0, 2, 1)).transpose(1, 2)
        return X


class TemporalSize(IntEnum):
    HOUR = 24
    WEEKDAY = 7
    DAY = 32
    MONTH = 13


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int, *,
                 embed_type: Literal['fixed'] | Literal['learnable'] = 'fixed'):
        super().__init__()

        Embedding = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        self.hour_embed = Embedding(TemporalSize.HOUR, d_model)
        self.weekday_embed = Embedding(TemporalSize.WEEKDAY, d_model)
        self.day_embed = Embedding(TemporalSize.DAY, d_model)
        self.month_embed = Embedding(TemporalSize.MONTH, d_model)

    def forward(self, mark: Tensor):
        mark = mark.type(torch.long)

        hour_embed = self.hour_embed(mark[:, :, 3])
        weekday_embed = self.weekday_embed(mark[:, :, 2])
        day_embed = self.day_embed(mark[:, :, 1])
        month_embed = self.month_embed(mark[:, :, 0])

        return hour_embed + weekday_embed + day_embed + month_embed


class DataEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int, *,
                 embed_type: Literal['fixed'] | Literal['learnable'] = 'fixed',
                 dropout=.1):
        super().__init__()

        self.token_emb = TokenEmbedding(c_in, d_model)
        self.temporal_emb = TemporalEmbedding(d_model, embed_type=embed_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, X_mark: Tensor):
        X = self.token_emb(X) + self.temporal_emb(X_mark)
        return self.dropout(X)
