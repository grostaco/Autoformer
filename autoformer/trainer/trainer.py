import torch
import torch.nn as nn

from typing import Any


class Trainer:
    def __init__(self, model: nn.Module, config: Any) -> None:
        self.config = config
        self.model = model

        self.device = self._get_device()

    def _get_device(self):
        device = torch.device('cuda' if self.config.use_gpu else 'cpu')
        return device

    def step(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(
            batch_y[:, -self.config.pred_len:, :], dtype=torch.float)
        dec_inp = torch.cat(
            [batch_y[:, :self.config.label_len, :], dec_inp], dim=1)

        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return outputs.cpu().detach().numpy()
