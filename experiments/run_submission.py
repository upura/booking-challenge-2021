import gc

import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import load_train_test, BookingDataset
from src.models import BookingNN
from src.utils import seed_everything
from src.runner import CustomRunner


if __name__ == '__main__':

    seed_everything(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
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

    test = train_test[~train_test['row_num'].isnull()]
    test_trips = test[test['city_id'] != test['city_id'].shift(1)].query('city_id!=0').groupby('utrip_id')['city_id'].apply(lambda x: x.values).reset_index()

    X_test = test[test['city_id'] != test['city_id'].shift(1)].query('city_id!=0').groupby('utrip_id')[categorical_cols].last().reset_index()
    X_test['city_id'] = test_trips['city_id']
    X_test = X_test.reset_index(drop=True)

    test_dataset = BookingDataset(X_test, is_train=False)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1)

    del train_test, test, test_trips
    gc.collect()

    model_paths = [
        '../input/booking-bi-lstm-ep1/logdir_nn000',
    ]
    for mp in model_paths:
        for fold_id in (0,):
            runner = CustomRunner(device=device)
            model = BookingNN(len(target_le.classes_))
            pred = []
            for prediction in tqdm(runner.predict_loader(loader=test_loader,
                                                         resume=f'{mp}/fold{fold_id}/checkpoints/best.pth',
                                                         model=model,)):
                pred.append(target_le.inverse_transform(np.argsort(prediction.cpu().numpy()[-1, :])[-4:]))
            pred = np.array(pred)
            np.save(f"y_pred{mp.replace('/', '_').replace('.', '')}_fold{fold_id}", pred)

    submission = pd.concat([
        X_test['utrip_id'],
        pd.DataFrame(pred, columns=['city_id_1', 'city_id_2', 'city_id_3', 'city_id_4'])
    ], axis=1)
    print(submission.head())
    submission.to_csv('submission.csv', index=False)
