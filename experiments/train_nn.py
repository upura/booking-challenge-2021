import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader

from src.datasets import load_train_test, BookingDataset, MyCollator
from src.models import BookingNN
from src.utils import seed_everything
from src.runner import CustomRunner


if __name__ == '__main__':

    run_name = 'nn000'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    categorical_cols = [
        'user_id',
        # 'device_class',
        # 'affiliate_id',
        'booker_country',
        # 'hotel_country'
    ]

    train_test = load_train_test()
    cat_dims = [int(train_test[col].nunique()) for col in categorical_cols]
    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    target_le = preprocessing.LabelEncoder()
    train_test['city_id'] = target_le.fit_transform(train_test['city_id'])

    for c in categorical_cols:
        le = preprocessing.LabelEncoder()
        train_test[c] = le.fit_transform(train_test[c].astype(str).fillna('unk').values)

    train = train_test[train_test['row_num'].isnull()]
    test = train_test[~train_test['row_num'].isnull()]

    train_trips = train[train['city_id'] != train['city_id'].shift(1)].groupby('utrip_id')['city_id'].apply(lambda x: x.values).reset_index()
    test_trips = test[test['city_id'] != test['city_id'].shift(1)].query('city_id!=0').groupby('utrip_id')['city_id'].apply(lambda x: x.values).reset_index()

    X_train = train[train['city_id'] != train['city_id'].shift(1)].groupby('utrip_id')[categorical_cols].last().reset_index()
    X_test = test[test['city_id'] != test['city_id'].shift(1)].query('city_id!=0').groupby('utrip_id')[categorical_cols].last().reset_index()

    X_train['city_id'] = train_trips['city_id']
    X_test['city_id'] = test_trips['city_id']

    X_train['n_trips'] = X_train['city_id'].map(lambda x: len(x))
    X_train = X_train.query('n_trips > 2').sort_values('n_trips').reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=False)

    test_dataset = BookingDataset(X_test, is_train=False)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1)

    del train_test, train, test, X_test
    gc.collect()

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train,
                                                        pd.cut(X_train['n_trips'], 5, labels=False))):
        if fold_id in (0,):
            X_tr = X_train.loc[tr_idx, :]
            X_val = X_train.loc[va_idx, :]

            train_dataset = BookingDataset(X=X_tr)
            valid_dataset = BookingDataset(X=X_val)

            collate = MyCollator(percentile=100)
            train_loader = DataLoader(train_dataset,
                                      shuffle=False,
                                      batch_size=256,
                                      collate_fn=collate)
            valid_loader = DataLoader(valid_dataset,
                                      shuffle=False,
                                      batch_size=1)

            loaders = {'train': train_loader, 'valid': valid_loader}
            runner = CustomRunner(device=device)

            model = BookingNN(len(target_le.classes_))
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
            logdir = f'logdir_{run_name}/fold{fold_id}'
            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                loaders=loaders,
                logdir=logdir,
                num_epochs=10,
                verbose=True,
            )
