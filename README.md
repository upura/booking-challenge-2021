# Booking.com Data Challenge
https://www.bookingchallenge.com/

There are three types of model.

1. BookingNN
1. BookingNN with Multi-task learning
1. BookingNN with Data augmentation

## BookingNN

```bash
python train_nn.py
```

![bookingnn](docs/booking_nn.png)

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

![bookingnn](docs/booking_nn_mtl.png)

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
