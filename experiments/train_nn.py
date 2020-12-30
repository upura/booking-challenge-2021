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
    test_trips = test.groupby('utrip_id')['city_id'].unique().reset_index()

    X_train = train.groupby('utrip_id')[categorical_cols].first().reset_index()
    X_test = test.groupby('utrip_id')[categorical_cols].first().reset_index()

    X_train['city_id'] = train_trips['city_id']
    X_test['city_id'] = test_trips['city_id']

    X_train['n_trips'] = X_train['city_id'].map(lambda x: len(x))
    X_test['n_trips'] = X_test['city_id'].map(lambda x: len(x))
    X_train = X_train.query('n_trips > 2').sort_values('n_trips').reset_index(drop=True)
    X_test = X_test.sort_values('n_trips').reset_index(drop=True)

    cv = StratifiedKFold(n_splits=5, shuffle=False)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    cv_scores = []

    test_dataset = BookingDataset(X_test, is_train=False)
    collate = MyCollator(is_train=False, percentile=100)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=32,
                             collate_fn=collate)

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
                                      batch_size=32,
                                      collate_fn=collate)
            collate = MyCollator(percentile=100)
            valid_loader = DataLoader(valid_dataset,
                                      shuffle=False,
                                      batch_size=32,
                                      collate_fn=collate)

            loaders = {'train': train_loader, 'valid': valid_loader}
            runner = CustomRunner(device=device)

            model = BookingNN(len(target_le.classes_))
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
            logdir = f'../output/logdir_{run_name}/fold{fold_id}'
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

            oof_train = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                                runner.predict_loader(
                                                loader=valid_loader,
                                                resume=f'{logdir}/checkpoints/best.pth',
                                                model=model,),)))

            print(oof_train.shape)
            print(oof_train)
            np.save('oof_train', oof_train)

            # oof_preds[va_idx] = pred
            # y_pred_oof = (pred > 0.5).astype(int)
            # score = accuracy_score(y_val, y_pred_oof)
            # cv_scores.append(score)
            # print('score', score)

            pred = np.concatenate(list(map(lambda x: x.cpu().numpy(),
                                           runner.predict_loader(
                                           loader=test_loader,
                                           resume=f'{logdir}/checkpoints/best.pth',
                                           model=model,),)))
            print(pred.shape)
            print(pred)
            np.save('y_pred', pred)

            # test_preds += pred / 5
