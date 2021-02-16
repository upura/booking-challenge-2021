# Booking.com Data Challenge 6th Place Solution

This repository contains 6th place solution codes for [Booking.com Data Challenge](https://www.bookingchallenge.com/), which is a challenge with a task of predicting travellers' next destination.
We trained four types of Long short-term memory (LSTM) model, and archived the final score: 0.5399 by weighted averaging of these predictions.
There are some differences in these models in feature engineering, multi-task learning, and data augmentation.
Our experiments showed that the diversity of the models boosted the final result.

Our solution is described in the submitted paper, and our code is available at [teammate's repository](https://github.com/hakubishin3/booking-challenge-2021) and here. In this repository, our baseline model and the models named LSTM 1-3 are avaliable.

- Baseline
- LSTM 1: BookingNN
- LSTM 2: BookingNNMtl (LSTM 1 + Multi-task learning)
- LSTM 3: BookingNNAug (LSTM 2 + Data augmentation)

| model | fold0 | fold1 | fold2 | fold3 | fold4 | average |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Baseline | 0.4773 | 0.4388 | 0.4420 | 0.4731 | 0.4837 | 0.4629 |
| LSTM 1 | 0.5083 | 0.4673 | 0.4728 | 0.4999 | 0.5154 | 0.49274 |
| LSTM 2 | 0.5086 | 0.4681 | 0.4747 | 0.5012 | 0.5162 | 0.49376 | 
| LSTM 3 | 0.5029 | 0.4575 | 0.4661 | 0.4967 | 0.5082 | 0.48628 |

## BookingNN

```bash
python train_nn.py
```

![image](docs/booking_nn.png)

- LSTM with categorical and numerical features.
- Pretrained weights of `city_id` and `hotel_country` are calculated by Word2Vec.
- Some numerical features are created like `days_stay`, `num_visit_same_city`, and so on.
- `days_stay` at the prediction point can be used and useful in this competition.

```python
from src.datasets import load_train_test, BookingDataset, MyCollator
from src.models import BookingNN
from src.utils import seed_everything
from src.runner import CustomRunner
```

## BookingNNMtl

```bash
python train_nn_mtl.py
```

![image](docs/booking_nn_mtl.png)

- Multi-task learning version BookingNN.
- Predict not only `city_id` but also `hotel_country`.

```python
from src.datasets import load_train_test
from src.datasets import BookingDatasetMtl as BookingDataset
from src.datasets import MyCollatorMtl as MyCollator
from src.models import BookingNNMtl as BookingNN
from src.utils import seed_everything
from src.runner import CustomRunnerMtl as CustomRunner
```

## BookingNNAug

Each sequence of trips can be flipped. Be sure to remove `booker_country` in order to keep consistency.

```bash
python train_nn_aug.py
```

```python
from src.datasets import load_train_test
from src.datasets import BookingDatasetAug as BookingDataset
from src.datasets import MyCollatorAug as MyCollator
from src.models import BookingNNAug as BookingNN
from src.utils import seed_everything
from src.runner import CustomRunnerAug as CustomRunner
```

## Graph features

We believe that graph related features are important because it can lead to reconstruct geographical information. Each sequence of trips are just a fragment of it.

The following figure is a scatter plot of `city_id` vectors calculated by Word2Vec. The number of dimension is compressed by umap. Embedding vectors are used as weights in the model.

```bash
python fe_w2v.py
```

![image](docs/scatter_city.png)

## Training tips

- Google Colab with GPU (runner.ipynb).
- Catalyst for PyTorch model training.
- Stratified split by the length of each trip.
- Use model with best validation score from epoch 11 to 14.
- You need to add `PYTHONPATH` as follows.

```python
import sys
sys.path.append(YOUR_PROJECT_ROOT)
```
