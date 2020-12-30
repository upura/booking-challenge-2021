from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import torch


def load_train_test():
    train_set = pd.read_csv('../input/booking_train_set.csv',
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
    test_set = pd.read_csv('../input/sample_test_set.csv',
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
                           ]).sort_values(by=['utrip_id', 'checkin'])
    return pd.concat([train_set, test_set], sort=False)


class BookingDataset:
    def __init__(self, X, is_train=True):
        self.X = X
        self.is_train = is_train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x_seq = self.X['city_id'].values[i]
        x_cat = self.X.drop(['utrip_id', 'city_id'], axis=1).iloc[i].values
        return (
            x_seq[:-1],
            x_seq[1:],
            x_cat
        ) if self.is_train else (
            x_seq,
            x_cat
        )


class MyCollator(object):
    def __init__(self, is_train=True, percentile=100):
        self.is_train = is_train
        self.percentile = percentile

    def __call__(self, batch):
        if self.is_train:
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
            cats = [item[2] for item in batch]
        else:
            data = [item[0] for item in batch]
            cats = [item[1] for item in batch]
        lens = [len(x) for x in data]
        max_len = np.percentile(lens, self.percentile)
        data = sequence.pad_sequences(data, maxlen=int(max_len))
        data = torch.tensor(data, dtype=torch.long)
        cats = torch.tensor(cats, dtype=torch.long)
        if self.is_train:
            target = sequence.pad_sequences(target, maxlen=int(max_len))
            target = torch.tensor(target, dtype=torch.long)
        return (
            data,
            target,
            cats
        ) if self.is_train else (
            data,
            cats
        )
