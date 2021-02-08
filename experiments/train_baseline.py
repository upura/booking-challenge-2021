import gc
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import torch

from src.datasets import load_train_test
from src.datasets import BookingDatasetBaseline as BookingDataset
from src.datasets import MyCollatorBaseline as MyCollator
from src.models import BookingNNBaseline as BookingNN
from src.utils import seed_everything
from src.runner import CustomRunnerBaseline as CustomRunner


CATEGORICAL_COLS = [
    "booker_country",
    "device_class",
    "affiliate_id",
]
NUMERICAL_COLS = [
]


if __name__ == '__main__':

    run_name = 'baseline'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_test = load_train_test()
    train_test["past_city_id"] = (
        train_test.groupby("utrip_id")["city_id"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    target_le = preprocessing.LabelEncoder()
    train_test["city_id"] = target_le.fit_transform(
        train_test["city_id"]
    )
    train_test["past_city_id"] = target_le.transform(
        train_test["past_city_id"]
    )

    cat_le = {}
    for c in CATEGORICAL_COLS:
        le = preprocessing.LabelEncoder()
        train_test[c] = le.fit_transform(
            train_test[c].fillna("UNK").astype(str).values
        )
        cat_le[c] = le

    train = train_test[train_test['row_num'].isnull()]
    test = train_test[~train_test['row_num'].isnull()]

    X_train, X_test = [], []
    for c in ["city_id", "past_city_id"] + CATEGORICAL_COLS + NUMERICAL_COLS:
        X_train.append(train[train['city_id'] != train['city_id'].shift(1)].groupby("utrip_id")[c].apply(list))
        X_test.append(test[test['city_id'] != test['city_id'].shift(1)].groupby("utrip_id")[c].apply(list))
    X_train = pd.concat(X_train, axis=1)
    X_test = pd.concat(X_test, axis=1)

    X_train['n_trips'] = X_train['city_id'].map(lambda x: len(x))
    X_train = X_train.query('n_trips > 2').sort_values('n_trips').reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=False)

    test_dataset = BookingDataset(X_test, is_train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=MyCollator(is_train=False),
        shuffle=False,
    )
    del train_test, train, test, X_test
    gc.collect()

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train,
                                                        pd.cut(X_train['n_trips'], 5, labels=False))):
        if fold_id in (0, 1, 2, 3, 4):
            X_tr = X_train.loc[tr_idx, :]
            X_val = X_train.loc[va_idx, :]

            train_dataset = BookingDataset(X_tr, is_train=True)
            valid_dataset = BookingDataset(X_val, is_train=True)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=256,
                num_workers=os.cpu_count(),
                pin_memory=True,
                collate_fn=MyCollator(is_train=True),
                shuffle=True,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=1,
                num_workers=os.cpu_count(),
                pin_memory=True,
                collate_fn=MyCollator(is_train=True),
                shuffle=False,
            )

            runner = CustomRunner(device=device)
            model = BookingNN(
                n_city_id=len(target_le.classes_),
                n_booker_country=len(cat_le["booker_country"].classes_),
                n_device_class=len(cat_le["device_class"].classes_),
                n_affiliate_id=len(cat_le["affiliate_id"].classes_),
            )
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
            logdir = f'logdir_{run_name}/fold{fold_id}'

            loaders = {'train': train_loader}
            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                loaders=loaders,
                logdir=logdir,
                num_epochs=11,
                verbose=True,
            )

            loaders = {'train': train_loader, 'valid': valid_loader}
            runner.train(
                model=model,
                resume=f'{logdir}/checkpoints/best_full.pth',
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                loaders=loaders,
                logdir=logdir,
                main_metric="accuracy04",
                minimize_metric=False,
                num_epochs=14,
                verbose=True,
            )
