import gc
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import torch

from src.datasets import load_train_test
from src.datasets import BookingDatasetAug as BookingDataset
from src.datasets import MyCollatorAug as MyCollator
from src.models import BookingNNAug as BookingNN
from src.utils import seed_everything
from src.runner import CustomRunnerAug as CustomRunner


CATEGORICAL_COLS = [
    # "booker_country",
    "device_class",
    "affiliate_id",
    "month_checkin",
    "past_hotel_country",
]
NUMERICAL_COLS = [
    "days_stay",
    "num_checkin",
    "days_move",
    "num_visit_drop_duplicates",
    "num_visit",
    "num_visit_same_city",
    "num_stay_consecutively"
]


if __name__ == '__main__':

    run_name = 'nn005'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_test = load_train_test()
    train_test["past_city_id"] = (
        train_test.groupby("utrip_id")["city_id"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )
    train_test["past_hotel_country"] = (
        train_test.groupby("utrip_id")["hotel_country"]
        .shift(1)
        .fillna("UNK")
        .astype(str)
    )
    target_le = preprocessing.LabelEncoder()
    train_test["city_id"] = target_le.fit_transform(
        train_test["city_id"]
    )
    train_test["past_city_id"] = target_le.transform(
        train_test["past_city_id"]
    )

    hotel_le = preprocessing.LabelEncoder()
    train_test["hotel_country"] = hotel_le.fit_transform(
        train_test["hotel_country"].fillna("UNK")
    )
    train_test["past_hotel_country"] = hotel_le.transform(
        train_test["past_hotel_country"]
    )
    train_test["checkin"] = pd.to_datetime(train_test["checkin"])
    train_test["checkout"] = pd.to_datetime(train_test["checkout"])

    train_test["month_checkin"] = train_test["checkin"].dt.month
    train_test["days_stay"] = (
        train_test["checkout"] - train_test["checkin"]
    ).dt.days.apply(lambda x: np.log10(x))
    train_test["num_checkin"] = (
        train_test.groupby("utrip_id")["checkin"]
        .rank()
        .apply(lambda x: np.log10(x))
    )
    train_test["past_checkout"] = train_test.groupby("utrip_id")[
        "checkout"
    ].shift(1)
    train_test["days_move"] = (
        (train_test["checkin"] - train_test["past_checkout"])
        .dt.days.fillna(0)
        .apply(lambda x: np.log1p(x))
    )

    num_visit_drop_duplicates = (
        train_test.query("city_id != 0")[["user_id", "city_id"]]
        .drop_duplicates()
        .groupby("city_id")
        .size()
        .apply(lambda x: np.log1p(x)).reset_index()
    )
    num_visit_drop_duplicates.columns = ["past_city_id", "num_visit_drop_duplicates"]
    num_visit = train_test.query("city_id != 0")[["user_id", "city_id"]].groupby("city_id").size().apply(lambda x: np.log1p(x)).reset_index()
    num_visit.columns = ["past_city_id", "num_visit"]
    num_visit_same_city = (
        train_test[train_test['city_id'] == train_test['city_id'].shift(1)]
        .groupby("city_id")
        .size()
        .apply(lambda x: np.log1p(x))
        .reset_index()
    )
    num_visit_same_city.columns = ["past_city_id", "num_visit_same_city"]
    train_test = pd.merge(train_test, num_visit_drop_duplicates, on="past_city_id", how="left")
    train_test = pd.merge(train_test, num_visit, on="past_city_id", how="left")
    train_test = pd.merge(train_test, num_visit_same_city, on="past_city_id", how="left")
    train_test["num_visit_drop_duplicates"].fillna(0, inplace=True)
    train_test["num_visit"].fillna(0, inplace=True)
    train_test["num_visit_same_city"].fillna(0, inplace=True)
    train_test["num_stay_consecutively"] = (
        train_test.groupby(["utrip_id", "past_city_id"])["past_city_id"]
        .rank(method="first")
        .fillna(1)
        .apply(lambda x: np.log1p(x))
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
    for c in ["city_id", "hotel_country", "past_city_id"] + CATEGORICAL_COLS + NUMERICAL_COLS:
        X_train.append(train[train['city_id'] != train['city_id'].shift(1)].groupby("utrip_id")[c].apply(list))
        X_test.append(test[test['city_id'] != test['city_id'].shift(1)].groupby("utrip_id")[c].apply(list))
    X_train = pd.concat(X_train, axis=1)
    X_test = pd.concat(X_test, axis=1)
    X_train['utrip_id'] = train[train['city_id'] != train['city_id'].shift(1)].groupby("utrip_id")['utrip_id'].count().index
    X_test['utrip_id'] = test[test['city_id'] != test['city_id'].shift(1)].groupby("utrip_id")['utrip_id'].count().index

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
    del train_test, train, test
    gc.collect()

    oof_preds = np.zeros((len(X_train), len(target_le.classes_)), dtype=np.float32)
    test_preds = np.zeros((len(X_test), len(target_le.classes_)), dtype=np.float32)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train,
                                                        pd.cut(X_train['n_trips'], 5, labels=False))):
        if fold_id in (0, 1, 2, 3, 4):
            logdir = f'logdir_{run_name}/fold{fold_id}'

            X_tr = X_train.loc[tr_idx, :]
            X_val = X_train.loc[va_idx, :]
            np.save(f"{logdir}/y_val_utrip_id{fold_id}", X_val["utrip_id"].values)

            X_aug = X_tr.copy()
            for c in ["city_id", "hotel_country", "past_city_id"] + CATEGORICAL_COLS + NUMERICAL_COLS:
                X_aug[c] = X_aug[c].map(lambda x: x[::-1])
            X_tr = pd.concat([X_tr, X_aug], axis=0).sort_values('n_trips')

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
                n_hotel_id=len(hotel_le.classes_),
                # n_booker_country=len(cat_le["booker_country"].classes_),
                n_device_class=len(cat_le["device_class"].classes_),
                n_affiliate_id=len(cat_le["affiliate_id"].classes_),
                n_month_checkin=len(cat_le["month_checkin"].classes_),
                n_hotel_country=len(cat_le["past_hotel_country"].classes_)
            )
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

            oof_preds[va_idx, :] = np.array(
                list(
                    map(
                        lambda x: x[0].cpu().numpy()[-1, :],
                        runner.predict_loader(
                            loader=valid_loader,
                            resume=f"{logdir}/checkpoints/best.pth",
                            model=model,
                        ),
                    )
                )
            )
            np.save(f"{logdir}/y_val_pred_fold{fold_id}", oof_preds[va_idx, :])

            test_preds_ = np.array(
                list(
                    map(
                        lambda x: x[0].cpu().numpy()[-1, :],
                        runner.predict_loader(
                            loader=test_loader,
                            resume=f"{logdir}/checkpoints/best.pth",
                            model=model,
                        ),
                    )
                )
            )
            test_preds += test_preds_ / cv.n_splits
            np.save(f"{logdir}/y_test_pred_fold{fold_id}", test_preds_)

    np.save(f"{logdir}/y_oof_pred", oof_preds)
    np.save(f"{logdir}/y_test_pred", test_preds)
    np.save(f"{logdir}/y_test_utrip_id", X_test["utrip_id"].values)
    np.save(f"{logdir}/y_oof_utrip_id", X_train["utrip_id"].values)
