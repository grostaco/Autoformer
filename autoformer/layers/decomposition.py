from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()

        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size, stride)

    def forward(self, x: Tensor):
        padding = (self.kernel_size - 1) // 2

        x = F.pad(x, (0, 0, padding, padding), mode='replicate')
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()

        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x: Tensor):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean

        return res, moving_mean
