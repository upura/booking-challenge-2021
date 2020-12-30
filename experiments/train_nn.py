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


def calc_score(y_train, y_pred):
    score = 0
    for y, p in zip(y_train, y_pred):
        correct = y in p
        score += int(correct)
    score /= len(y_train)
    return score


if __name__ == '__main__':

    run_name = 'nn000'
    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    categorical_cols = [
        'user_id',
        'device_class',
        'affiliate_id',
        'booker_country',
        'hotel_country'
    ]

    train_test = load_train_test()

    target_le = preprocessing.LabelEncoder()
    train_test['city_id'] = target_le.fit_transform(train_test['city_id'])

    for c in categorical_cols:
        le = preprocessing.LabelEncoder()
        train_test[c] = le.fit_transform(train_test[c].astype(str).fillna('unk').values)

    train = train_test[train_test['row_num'].isnull()]
    test = train_test[~train_test['row_num'].isnull()]

    train_trips = train.groupby('utrip_id')['city_id'].unique().reset_index()
    test_trips = test.query('city_id!=0').groupby('utrip_id')['city_id'].unique().reset_index()

    X_train = train.groupby('utrip_id')[categorical_cols].last().reset_index()
    X_test = test.query('city_id!=0').groupby('utrip_id')[categorical_cols].last().reset_index()

    X_train['city_id'] = train_trips['city_id']
    X_test['city_id'] = test_trips['city_id']

    X_train['n_trips'] = X_train['city_id'].map(lambda x: len(x))
    X_train = X_train.query('n_trips > 2').sort_values('n_trips').reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    cv_scores = []

    test_dataset = BookingDataset(X_test, is_train=False)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1)

    for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_train,
                                                        pd.cut(X_train['n_trips'], 5, labels=False))):
        if fold_id == 0:
            X_tr = X_train.loc[tr_idx, :]
            X_val = X_train.loc[va_idx, :]

            train_dataset = BookingDataset(X=X_tr)
            valid_dataset = BookingDataset(X=X_val)

            collate = MyCollator(percentile=100)
            train_loader = DataLoader(train_dataset,
                                      shuffle=False,
                                      batch_size=128,
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
                num_epochs=1,
                verbose=True,
            )

            oof_train = np.array(list(map(lambda x: x.cpu().numpy()[-1, :],
                                          runner.predict_loader(
                                              loader=valid_loader,
                                              resume=f'{logdir}/checkpoints/best.pth',
                                              model=model,),)))

            print(oof_train.shape)
            np.save(f'oof_train_fold{fold_id}', oof_train)

            y_train = X_train['city_id'].map(lambda x: x[-1])
            _oof = np.argsort(oof_train, axis=1)[:, -4:]
            print('acc@4', calc_score(y_train, _oof))

            pred = np.array(list(map(lambda x: x.cpu().numpy()[-1, :],
                                     runner.predict_loader(
                                         loader=test_loader,
                                         resume=f'{logdir}/checkpoints/best.pth',
                                         model=model,),)))
            print(pred.shape)
            np.save(f'y_pred_fold{fold_id}', pred)
