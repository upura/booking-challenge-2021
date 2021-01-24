from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import torch


def load_train_test():
    train_set = pd.read_csv('../input/booking/booking_train_set.csv',
                            usecols=[
                                'user_id',
                                'checkin',
                                'checkout',
                                'city_id',
                                'device_class',
                                'affiliate_id',
                                'booker_country',
                                'hotel_country',
                                'utrip_id'
                            ]).sort_values(by=['utrip_id', 'checkin'])
    test_set = pd.read_csv('../input/booking/booking_test_set.csv',
                           usecols=[
                               'user_id',
                               'checkin',
                               'checkout',
                               'device_class',
                               'affiliate_id',
                               'booker_country',
                               'utrip_id',
                               'row_num',       # test only
                               'total_rows',    # test only
                               'city_id',
                               'hotel_country'
                           ]).sort_values(by=['utrip_id', 'row_num'])
    return pd.concat([train_set, test_set], sort=False)


class BookingDataset:
    def __init__(self, X, is_train=True, categorical_cols=[], numerical_cols=[]):
        self.X = X
        self.is_train = is_train
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x_seq = self.X['city_id'].values[i]
        x_cat = self.X[self.categorical_cols].iloc[i].values
        x_num = self.X['duration'].values[i]
        return (
            x_seq[:-1],
            x_seq[1:],
            x_cat,
            x_num[1:]
        ) if self.is_train else (
            x_seq,
            x_cat,
            x_num[1:]
        )


class MyCollator(object):
    def __init__(self, mode="train", percentile=100):
        self.mode = mode
        self.percentile = percentile

    def __call__(self, batch):
        if self.mode in ("train", "valid"):
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            cats = [item[2] for item in batch]
            nums = [item[3] for item in batch]
        else:
            data = [item[0] for item in batch]
            cats = [item[1] for item in batch]
            nums = [item[2] for item in batch]
        lens = [len(x) for x in data]
        max_len = np.percentile(lens, self.percentile)
        if self.mode in ("train"):
            data = sequence.pad_sequences(data, maxlen=int(max_len))
        data = torch.tensor(data, dtype=torch.long)
        cats = torch.tensor(cats, dtype=torch.long)
        nums = sequence.pad_sequences(nums, maxlen=int(max_len))
        nums = torch.tensor(nums, dtype=torch.float)
        if self.mode in ("train", "valid"):
            target = sequence.pad_sequences(target, maxlen=int(max_len))
            target = torch.tensor(target, dtype=torch.long)
        return (
            data,
            target,
            cats,
            nums
        ) if self.mode in ("train", "valid") else (
            data,
            cats,
            nums
        )
