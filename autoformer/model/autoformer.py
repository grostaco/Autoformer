import torch
import torch.nn as nn

from autoformer.layers.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from autoformer.layers.decoder import Decoder, DecoderLayer
from autoformer.layers.encoder import Encoder, EncoderLayer
from autoformer.layers.norm import SeasonalLayerNorm
from autoformer.layers.embedding import DataEmbedding
from autoformer.layers.decomposition import SeriesDecomposition


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        kernel_size = configs.moving_avg
        self.decomp = SeriesDecomposition(kernel_size)

        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, embed_type=configs.embed, dropout=configs.dropout)
        self.dec_embedding = DataEmbedding(
            configs.dec_in, configs.d_model, embed_type=configs.embed, dropout=configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    AutoCorrelationLayer(
                        configs.d_model,
                        configs.n_heads,
                        AutoCorrelation(factor=configs.factor,
                                        dropout=configs.dropout),
                    ),
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=SeasonalLayerNorm(configs.d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        configs.d_model,
                        configs.n_heads,
                        AutoCorrelation(
                            factor=configs.factor,
                            dropout=configs.dropout
                        )
                    ),
                    AutoCorrelationLayer(
                        configs.d_model,
                        configs.n_heads,
                        AutoCorrelation(
                            factor=configs.factor,
                            dropout=configs.dropout
                        )
                    ),
                    d_model=configs.d_model,
                    c_out=configs.c_out,
                    d_ff=configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.d_layers)
            ],
            norm_layer=SeasonalLayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                            x_dec.shape[2]], device=x_enc.device, dtype=torch.float)

        seasonal_init, trend_init = self.decomp(x_enc)

        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_emb)

        dec_emb = self.dec_embedding(seasonal_init, x_mark_dec)

        seasonal_part, trend_part = self.decoder(
            dec_emb, enc_out, trend=trend_init)

        dec_out = seasonal_part + trend_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        return dec_out[:, -self.pred_len:, :]
