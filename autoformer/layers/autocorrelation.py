import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import math


class AutoCorrelation(nn.Module):
    def __init__(self, *,
                 factor: float = 1.,
                 dropout=.1):
        super().__init__()

        self.factor = factor
        self.dropout = nn.Dropout(dropout)

    def time_delay_agg_training(self, values: Tensor, corr: Tensor):
        N, H, E_v, S = values.shape

        top_k = int(self.factor * math.log(S))
        mean = torch.mean(corr, dim=(1, 2))
        _, indices = torch.topk(torch.mean(mean, dim=0), top_k)

        weights = mean[:, indices[:top_k]]

        tmp_corr = torch.softmax(weights, dim=-1)

        delays_agg = torch.zeros_like(values, dtype=torch.float)

        for i in range(top_k):
            pattern = torch.roll(values, -indices[i].item(), dims=-1)
            delays_agg = delays_agg + pattern * tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1).repeat(1, H, E_v, S)
        return self.dropout(delays_agg)

    def time_delay_agg_inference(self, values: Tensor, corr: Tensor):
        N, H, E_v, S = values.shape

        init_indices = torch.arange(S).repeat(N, H, E_v, 1).to(values.device)

        top_k = int(self.factor * math.log(S))
        mean = torch.mean(corr, dim=(1, 2))
        weights, delay = torch.topk(mean, top_k)

        tmp_corr = torch.softmax(weights, dim=-1)

        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values, dtype=torch.float)

        for i in range(top_k):
            tmp_delay = init_indices + delay[:, i].repeat(1, H, E_v, S)
            pattern = torch.gather(tmp_values, -1, tmp_delay)

            delays_agg += pattern * tmp_corr[:, i].repeat(1, H, E_v, S)

        return delays_agg

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, *,
                need_weights=True):
        """Forward propagation call for `AutoCorrelation`

        Parameters
        ----------
        queries : Tensor
            Query embeddings of shape `(N, L, H, E_q)`, where `L` is the target sequence length,
            `N` is the batch size, `H` is the number of heads, and `E_q` is the query embedding dimension.
        keys : Tensor
            Key embeddings of shape `(N, S, H, E_k)`, where `S` is the source sequence length,
            `N` is the batch size, `H` is the number of heads, and `E_k` is the key embedding dimension.
        values : Tensor
            Value embeddings of shape `(N, S, H, E_v)`, where `S` is the source sequence length,
            `N` is the batch size, `H` is the number of heads, and `E_v` is the value embedding dimension.
        """
        _, L, _, _ = queries.shape
        _, S, _, _ = values.shape

        if L > S:
            pad = (0, 0, 0, 0, 0, L - S)
            keys = F.pad(keys, pad, 'constant', 0)
            values = F.pad(values, pad, 'constant', 0)

        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1), dim=-1)
        res = q_fft * torch.conj(k_fft)

        corr = torch.fft.irfft(res, n=L, dim=-1)

        # (N, S, H, E_v)
        # (N, H, E_v, S)
        # (N, S, H, E_v)
        if self.training:
            v = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1), corr).permute(0, 3, 1, 2)
        else:
            v = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1), corr).permute(0, 3, 1, 2)

        if need_weights:
            return (v, corr.permute(0, 3, 1, 2))
        return v


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 attn_pool: nn.Module):
        super().__init__()

        self.num_heads = num_heads

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_pool = attn_pool

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, need_weights=True):
        N, L, E_q = queries.shape
        _, S, E_k = keys.shape
        H = self.num_heads

        queries = self.query_proj(queries).view(N, L, H, -1)
        keys = self.key_proj(keys).view(N, S, H, -1)
        values = self.value_proj(values).view(N, S, H, -1)

        if need_weights:
            out, attn = self.attn_pool(
                queries, keys, values, need_weights=need_weights)
        else:
            out = self.attn_pool(queries, keys, values,
                                 need_weights=need_weights)

        out = out.view(N, L, -1)

        if need_weights:
            return self.out_proj(out), attn  # type: ignore
        return self.out_proj(out)
