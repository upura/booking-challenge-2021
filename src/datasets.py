from keras.preprocessing import sequence
import pandas as pd
import torch


def load_train_test():
    train_set = pd.read_csv('../input/booking/booking_train_set.csv',
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
    test_set = pd.read_csv('../input/booking/booking_test_set.csv',
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
                           ]).sort_values(by=['utrip_id', 'row_num'])
    return pd.concat([train_set, test_set], sort=False)


class BookingDatasetBaseline(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True) -> None:
        super().__init__
        self.is_train = is_train
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        city_id_tensor = self.df["past_city_id"].values[index]
        booker_country_tensor = self.df["booker_country"].values[index]
        device_class_tensor = self.df["device_class"].values[index]
        affiliate_id_tensor = self.df["affiliate_id"].values[index]
        target_tensor = self.df["city_id"].values[index][
            -1
        ]  # extrast last value of sequence

        if self.is_train:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                target_tensor,
            )
        else:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
            )


class MyCollatorBaseline(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        city_id_tensor = [item[0] for item in batch]
        booker_country_tensor = [item[1] for item in batch]
        device_class_tensor = [item[2] for item in batch]
        affiliate_id_tensor = [item[3] for item in batch]
        if self.is_train:
            targets = [item[-1] for item in batch]

        def _pad_sequences(data, maxlen: int, dtype=torch.long) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=dtype)

        lens = [len(s) for s in city_id_tensor]
        city_id_tensor = _pad_sequences(city_id_tensor, max(lens))
        booker_country_tensor = _pad_sequences(booker_country_tensor, max(lens))
        device_class_tensor = _pad_sequences(device_class_tensor, max(lens))
        affiliate_id_tensor = _pad_sequences(affiliate_id_tensor, max(lens))
        if self.is_train:
            targets = torch.tensor(targets, dtype=torch.long)
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                targets,
            )

        return (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
        )


class BookingDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True) -> None:
        super().__init__
        self.is_train = is_train
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        city_id_tensor = self.df["past_city_id"].values[index]
        booker_country_tensor = self.df["booker_country"].values[index]
        device_class_tensor = self.df["device_class"].values[index]
        affiliate_id_tensor = self.df["affiliate_id"].values[index]
        target_tensor = self.df["city_id"].values[index][
            -1
        ]  # extrast last value of sequence
        month_checkin_tensor = self.df["month_checkin"].values[index]
        num_checkin_tensor = self.df["num_checkin"].values[index]
        days_stay_tensor = self.df["days_stay"].values[index]
        days_move_tensor = self.df["days_move"].values[index]
        hotel_country_tensor = self.df["past_hotel_country"].values[index]
        num_visit_drop_duplicates_tensor = self.df["num_visit_drop_duplicates"].values[index]
        num_visit_tensor = self.df["num_visit"].values[index]
        num_visit_same_city_tensor = self.df["num_visit_same_city"].values[index]
        num_stay_consecutively_tensor = self.df["num_stay_consecutively"].values[index]

        if self.is_train:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
                target_tensor,
            )
        else:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
            )


class MyCollator(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        city_id_tensor = [item[0] for item in batch]
        booker_country_tensor = [item[1] for item in batch]
        device_class_tensor = [item[2] for item in batch]
        affiliate_id_tensor = [item[3] for item in batch]
        month_checkin_tensor = [item[4] for item in batch]
        num_checkin_tensor = [item[5] for item in batch]
        days_stay_tensor = [item[6] for item in batch]
        days_move_tensor = [item[7] for item in batch]
        hotel_country_tensor = [item[8] for item in batch]
        num_visit_drop_duplicates_tensor = [item[9] for item in batch]
        num_visit_tensor = [item[10] for item in batch]
        num_visit_same_city_tensor = [item[11] for item in batch]
        num_stay_consecutively_tensor = [item[12] for item in batch]
        if self.is_train:
            targets = [item[-1] for item in batch]

        def _pad_sequences(data, maxlen: int, dtype=torch.long) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=dtype)

        lens = [len(s) for s in city_id_tensor]
        city_id_tensor = _pad_sequences(city_id_tensor, max(lens))
        booker_country_tensor = _pad_sequences(booker_country_tensor, max(lens))
        device_class_tensor = _pad_sequences(device_class_tensor, max(lens))
        affiliate_id_tensor = _pad_sequences(affiliate_id_tensor, max(lens))
        month_checkin_tensor = _pad_sequences(month_checkin_tensor, max(lens))
        num_checkin_tensor = _pad_sequences(
            num_checkin_tensor, max(lens), dtype=torch.float
        )
        days_stay_tensor = _pad_sequences(
            days_stay_tensor, max(lens), dtype=torch.float
        )
        days_move_tensor = _pad_sequences(
            days_move_tensor, max(lens), dtype=torch.float
        )
        hotel_country_tensor = _pad_sequences(hotel_country_tensor, max(lens))
        num_visit_drop_duplicates_tensor = _pad_sequences(num_visit_drop_duplicates_tensor, max(lens))
        num_visit_tensor = _pad_sequences(num_visit_tensor, max(lens))
        num_visit_same_city_tensor = _pad_sequences(num_visit_same_city_tensor, max(lens))
        num_stay_consecutively_tensor = _pad_sequences(num_stay_consecutively_tensor, max(lens))
        if self.is_train:
            targets = torch.tensor(targets, dtype=torch.long)
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
                targets,
            )

        return (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
            month_checkin_tensor,
            num_checkin_tensor,
            days_stay_tensor,
            days_move_tensor,
            hotel_country_tensor,
            num_visit_drop_duplicates_tensor,
            num_visit_tensor,
            num_visit_same_city_tensor,
            num_stay_consecutively_tensor,
        )


class BookingDatasetMtl(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True) -> None:
        super().__init__
        self.is_train = is_train
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        city_id_tensor = self.df["past_city_id"].values[index]
        booker_country_tensor = self.df["booker_country"].values[index]
        device_class_tensor = self.df["device_class"].values[index]
        affiliate_id_tensor = self.df["affiliate_id"].values[index]
        target_tensor = self.df["city_id"].values[index][
            -1
        ]  # extrast last value of sequence
        target_h_tensor = self.df["hotel_country"].values[index][
            -1
        ]  # extrast last value of sequence
        month_checkin_tensor = self.df["month_checkin"].values[index]
        num_checkin_tensor = self.df["num_checkin"].values[index]
        days_stay_tensor = self.df["days_stay"].values[index]
        days_move_tensor = self.df["days_move"].values[index]
        hotel_country_tensor = self.df["past_hotel_country"].values[index]
        num_visit_drop_duplicates_tensor = self.df["num_visit_drop_duplicates"].values[index]
        num_visit_tensor = self.df["num_visit"].values[index]
        num_visit_same_city_tensor = self.df["num_visit_same_city"].values[index]
        num_stay_consecutively_tensor = self.df["num_stay_consecutively"].values[index]

        if self.is_train:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
                target_tensor,
                target_h_tensor,
            )
        else:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
            )


class MyCollatorMtl(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        city_id_tensor = [item[0] for item in batch]
        booker_country_tensor = [item[1] for item in batch]
        device_class_tensor = [item[2] for item in batch]
        affiliate_id_tensor = [item[3] for item in batch]
        month_checkin_tensor = [item[4] for item in batch]
        num_checkin_tensor = [item[5] for item in batch]
        days_stay_tensor = [item[6] for item in batch]
        days_move_tensor = [item[7] for item in batch]
        hotel_country_tensor = [item[8] for item in batch]
        num_visit_drop_duplicates_tensor = [item[9] for item in batch]
        num_visit_tensor = [item[10] for item in batch]
        num_visit_same_city_tensor = [item[11] for item in batch]
        num_stay_consecutively_tensor = [item[12] for item in batch]
        if self.is_train:
            targets = [item[-2] for item in batch]
            targets_h = [item[-1] for item in batch]

        def _pad_sequences(data, maxlen: int, dtype=torch.long) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=dtype)

        lens = [len(s) for s in city_id_tensor]
        city_id_tensor = _pad_sequences(city_id_tensor, max(lens))
        booker_country_tensor = _pad_sequences(booker_country_tensor, max(lens))
        device_class_tensor = _pad_sequences(device_class_tensor, max(lens))
        affiliate_id_tensor = _pad_sequences(affiliate_id_tensor, max(lens))
        month_checkin_tensor = _pad_sequences(month_checkin_tensor, max(lens))
        num_checkin_tensor = _pad_sequences(
            num_checkin_tensor, max(lens), dtype=torch.float
        )
        days_stay_tensor = _pad_sequences(
            days_stay_tensor, max(lens), dtype=torch.float
        )
        days_move_tensor = _pad_sequences(
            days_move_tensor, max(lens), dtype=torch.float
        )
        hotel_country_tensor = _pad_sequences(hotel_country_tensor, max(lens))
        num_visit_drop_duplicates_tensor = _pad_sequences(num_visit_drop_duplicates_tensor, max(lens))
        num_visit_tensor = _pad_sequences(num_visit_tensor, max(lens))
        num_visit_same_city_tensor = _pad_sequences(num_visit_same_city_tensor, max(lens))
        num_stay_consecutively_tensor = _pad_sequences(num_stay_consecutively_tensor, max(lens))
        if self.is_train:
            targets = torch.tensor(targets, dtype=torch.long)
            targets_h = torch.tensor(targets_h, dtype=torch.long)
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
                targets,
                targets_h,
            )

        return (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
            month_checkin_tensor,
            num_checkin_tensor,
            days_stay_tensor,
            days_move_tensor,
            hotel_country_tensor,
            num_visit_drop_duplicates_tensor,
            num_visit_tensor,
            num_visit_same_city_tensor,
            num_stay_consecutively_tensor,
        )


class BookingDatasetAug(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True) -> None:
        super().__init__
        self.is_train = is_train
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        city_id_tensor = self.df["past_city_id"].values[index]
        # booker_country_tensor = self.df["booker_country"].values[index]
        device_class_tensor = self.df["device_class"].values[index]
        affiliate_id_tensor = self.df["affiliate_id"].values[index]
        target_tensor = self.df["city_id"].values[index][
            -1
        ]  # extrast last value of sequence
        target_h_tensor = self.df["hotel_country"].values[index][
            -1
        ]  # extrast last value of sequence
        month_checkin_tensor = self.df["month_checkin"].values[index]
        num_checkin_tensor = self.df["num_checkin"].values[index]
        days_stay_tensor = self.df["days_stay"].values[index]
        days_move_tensor = self.df["days_move"].values[index]
        hotel_country_tensor = self.df["past_hotel_country"].values[index]
        num_visit_drop_duplicates_tensor = self.df["num_visit_drop_duplicates"].values[index]
        num_visit_tensor = self.df["num_visit"].values[index]
        num_visit_same_city_tensor = self.df["num_visit_same_city"].values[index]
        num_stay_consecutively_tensor = self.df["num_stay_consecutively"].values[index]

        if self.is_train:
            return (
                city_id_tensor,
                # booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
                target_tensor,
                target_h_tensor,
            )
        else:
            return (
                city_id_tensor,
                # booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
            )


class MyCollatorAug(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        city_id_tensor = [item[0] for item in batch]
        # booker_country_tensor = [item[1] for item in batch]
        device_class_tensor = [item[1] for item in batch]
        affiliate_id_tensor = [item[2] for item in batch]
        month_checkin_tensor = [item[3] for item in batch]
        num_checkin_tensor = [item[4] for item in batch]
        days_stay_tensor = [item[5] for item in batch]
        days_move_tensor = [item[6] for item in batch]
        hotel_country_tensor = [item[7] for item in batch]
        num_visit_drop_duplicates_tensor = [item[8] for item in batch]
        num_visit_tensor = [item[9] for item in batch]
        num_visit_same_city_tensor = [item[10] for item in batch]
        num_stay_consecutively_tensor = [item[11] for item in batch]
        if self.is_train:
            targets = [item[-2] for item in batch]
            targets_h = [item[-1] for item in batch]

        def _pad_sequences(data, maxlen: int, dtype=torch.long) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=dtype)

        lens = [len(s) for s in city_id_tensor]
        city_id_tensor = _pad_sequences(city_id_tensor, max(lens))
        # booker_country_tensor = _pad_sequences(booker_country_tensor, max(lens))
        device_class_tensor = _pad_sequences(device_class_tensor, max(lens))
        affiliate_id_tensor = _pad_sequences(affiliate_id_tensor, max(lens))
        month_checkin_tensor = _pad_sequences(month_checkin_tensor, max(lens))
        num_checkin_tensor = _pad_sequences(
            num_checkin_tensor, max(lens), dtype=torch.float
        )
        days_stay_tensor = _pad_sequences(
            days_stay_tensor, max(lens), dtype=torch.float
        )
        days_move_tensor = _pad_sequences(
            days_move_tensor, max(lens), dtype=torch.float
        )
        hotel_country_tensor = _pad_sequences(hotel_country_tensor, max(lens))
        num_visit_drop_duplicates_tensor = _pad_sequences(num_visit_drop_duplicates_tensor, max(lens))
        num_visit_tensor = _pad_sequences(num_visit_tensor, max(lens))
        num_visit_same_city_tensor = _pad_sequences(num_visit_same_city_tensor, max(lens))
        num_stay_consecutively_tensor = _pad_sequences(num_stay_consecutively_tensor, max(lens))
        if self.is_train:
            targets = torch.tensor(targets, dtype=torch.long)
            targets_h = torch.tensor(targets_h, dtype=torch.long)
            return (
                city_id_tensor,
                # booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                month_checkin_tensor,
                num_checkin_tensor,
                days_stay_tensor,
                days_move_tensor,
                hotel_country_tensor,
                num_visit_drop_duplicates_tensor,
                num_visit_tensor,
                num_visit_same_city_tensor,
                num_stay_consecutively_tensor,
                targets,
                targets_h,
            )

        return (
            city_id_tensor,
            # booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
            month_checkin_tensor,
            num_checkin_tensor,
            days_stay_tensor,
            days_move_tensor,
            hotel_country_tensor,
            num_visit_drop_duplicates_tensor,
            num_visit_tensor,
            num_visit_same_city_tensor,
            num_stay_consecutively_tensor,
        )
