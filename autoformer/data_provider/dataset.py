import torch
import os
import torch.utils.data
import pandas as pd
import numpy as np

from torch import Tensor

from typing import Literal
from sklearn.preprocessing import StandardScaler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path: str | os.PathLike,
                 *,
                 target: None | str = None,
                 lengths: tuple[int, int, int] = (24 * 4 * 4, 24 * 4, 24 * 4),
                 data_type: Literal['train'] | Literal['test'] | Literal['validate'] = 'train',
                 feature_type: Literal['univariate'] | Literal['multivariate'] = 'multivariate',
                 freq: str = 'h',
                 scale: bool = True,
                 split: tuple[float, float, float | None] = (.7, .2, .1)
                 ) -> None:
        super().__init__()

        self.path = path

        self.target = target
        self.seq_len, self.label_len, self.pred_len = lengths
        self.train_split, self.test_split, _ = split

        self.data_type = data_type

        self.feature_type = feature_type
        self.freq = freq
        self.scale = scale
        if self.scale:
            self.scaler = StandardScaler()

        self.split = split
        self._read()

    def _read(self):
        df = pd.read_csv(self.path)

        # sanity check: ensure that the dataframe follows the column format
        # [date] [... features ...] [target]

        columns = list(df.columns)
        columns.remove('date')

        if self.target is not None:
            columns.remove(self.target)
            df = df[['date'] + columns + [self.target]]
        else:
            # Infer the last column as the target
            df = df[['date'] + columns]

        match self.feature_type:
            case 'multivariate':
                df_data = df.iloc[:, 1:]
            case 'univariate':
                if self.target is None:
                    df_data = df.iloc[:, [-1]]
                else:
                    df_data = df[[self.target]]
            case t:
                raise ValueError(
                    f'Unknown feature type `{t}`, available ones are [univariate, multivariate]')

        # Splitting for train-test-validation
        n = len(df)
        n_train = int(n * self.train_split)
        n_test = int(n * self.test_split)
        n_val = n - n_train - n_test

        # pylance does not realize that data_type is a union of literal
        self.data_type: Literal['train'] | Literal['test'] | Literal['validate']
        match self.data_type:
            case 'train':
                begin = 0
                end = n_train
            case 'test':
                begin = n_train - self.seq_len
                end = n - n_val
            case 'validate':
                begin = n - n_test - self.seq_len
                end = n
            case t:
                raise ValueError(
                    f'Unknown data type `{t}`, available ones are [train, test, validate]')

        self.features = df[begin:end]

        if self.scale:
            train_data = df_data[:n_train]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data.values

        df_stamp = df[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        for date_type in ['month', 'day', 'hour']:
            df_stamp[date_type] = df_stamp['date'].apply(
                lambda x: getattr(x, date_type), True)
        df_stamp['weekday'] = df_stamp['date'].apply(
            lambda x: x.weekday(), True)

        self.data_stamp = df_stamp[['month', 'day', 'weekday', 'hour']].values
        self.data = data[begin:end]  # type: ignore

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        src = self.data[s_begin:s_end]
        tgt = self.data[r_begin:r_end]

        src_mark = self.data_stamp[s_begin:s_end]
        tgt_mark = self.data_stamp[r_begin:r_end]

        return src, tgt, src_mark, tgt_mark

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray | Tensor):
        return self.scaler.inverse_transform(data)
