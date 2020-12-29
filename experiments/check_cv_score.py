import numpy as np
from scipy.stats import rankdata

from ayniy.utils import Data


def calc_score(y_train, y_pred):
    score = 0
    for y, p in zip(y_train, y_pred):
        correct = y in p
        score += int(correct)
    score /= len(y_train)
    return score


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


run_ids = [
    'run000'
]
run_name = 'weight000'


if __name__ == '__main__':
    y_train = Data.load('../input/pickle/y_train_fe000.pkl')
    data = [load_from_run_id(ri, to_rank=False) for ri in run_ids]
    y_pred = np.argsort(data[0][0], axis=1)[:, -4:]
    print(calc_score(y_train, y_pred))
