import torch
from torch import Tensor
import torch.nn as nn


class SeasonalLayerNorm(nn.Module):
    def __init__(self, c_in: int):
        super().__init__()

        self.ln = nn.LayerNorm(c_in)

    def forward(self, x: Tensor):
        x = self.ln(x)
        bias = torch.mean(x, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)

        return x - bias
