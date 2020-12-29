import os

import pandas as pd
from sklearn import preprocessing
import xfeat

from ayniy.utils import Data


def load_train_test() -> pd.DataFrame:
    filepath = "../input/feather/train_test.ftr"
    if not os.path.exists(filepath):
        # Convert dataset into feather format.
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
        xfeat.utils.compress_df(pd.concat([train_set, test_set], sort=False)).reset_index(
            drop=True
        ).to_feather(filepath)

    return pd.read_feather(filepath)


categorical_cols = [
    'user_id',
    'device_class',
    'affiliate_id',
    'booker_country',
    'hotel_country'
]
target_col = 'city_id'


if __name__ == '__main__':

    train_test = load_train_test()

    # label encoding
    for c in categorical_cols:
        le = preprocessing.LabelEncoder()
        train_test[c] = le.fit_transform(train_test[c].astype(str).fillna('unk').values)

    # target limitation
    y_train = train_test[train_test['row_num'].isnull()].groupby('utrip_id')['city_id'].last().reset_index(drop=True)
    train_test[train_test['row_num'].isnull()].groupby('utrip_id')['city_id'].last().reset_index().to_csv('../input/y_train.csv', index=False)

    top_cities = y_train.value_counts().index[:1000]
    print(y_train.value_counts().head(10))
    y_train.loc[~y_train.isin(top_cities)] = -1
    le = preprocessing.LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train))

    # feature engineering
    X_train = train_test[train_test['row_num'].isnull()].groupby('utrip_id')[categorical_cols].max()
    X_test = train_test[~train_test['row_num'].isnull()].groupby('utrip_id')[categorical_cols].max()
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    print(y_train.unique())
    print(X_train.shape)
    print(X_test.shape)

    # distinations = train_test[train_test['row_num'].isnull()].groupby('utrip_id')['city_id'].unique()

    fe_name = 'fe000'
    Data.dump(X_train, f'../input/pickle/X_train_{fe_name}.pkl')
    Data.dump(y_train, f'../input/pickle/y_train_{fe_name}.pkl')
    Data.dump(X_test, f'../input/pickle/X_test_{fe_name}.pkl')
