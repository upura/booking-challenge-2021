import torch
from catalyst import metrics
from catalyst.dl import Runner
from catalyst.dl.utils import any2device


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        (
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
            y,
        ) = batch
        out = self.model(
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
        loss = self.criterion(out, y)
        accuracy01, accuracy04 = metrics.accuracy(out, y, topk=(1, 4))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy04": accuracy04}
        )
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = any2device(batch, self.device)
        if len(batch) == 13:
            (
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
            ) = batch
        elif len(batch) == 14:
            (
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
                y,
            ) = batch
        else:
            raise RuntimeError
        out = self.model(
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
        return out


class CustomRunnerMtl(Runner):
    def _handle_batch(self, batch):
        (
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
            y_s,
            y_h,
        ) = batch
        out_s, out_h = self.model(
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
        loss_s = self.criterion(out_s, y_s)
        loss_h = self.criterion(out_h, y_h)
        loss = loss_s * 0.8 + loss_h * 0.2
        accuracy01, accuracy04 = metrics.accuracy(out_s, y_s, topk=(1, 4))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy04": accuracy04}
        )
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = any2device(batch, self.device)
        if len(batch) == 13:
            (
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
            ) = batch
        elif len(batch) == 15:
            (
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
                y_s,
                y_h,
            ) = batch
        else:
            raise RuntimeError
        out_s, out_h = self.model(
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
        return out_s, out_h


class CustomRunnerAug(Runner):
    def _handle_batch(self, batch):
        (
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
            y_s,
            y_h,
        ) = batch
        out_s, out_h = self.model(
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
        loss_s = self.criterion(out_s, y_s)
        loss_h = self.criterion(out_h, y_h)
        loss = loss_s * 0.8 + loss_h * 0.2
        accuracy01, accuracy04 = metrics.accuracy(out_s, y_s, topk=(1, 4))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy04": accuracy04}
        )
        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def predict_batch(self, batch):
        batch = any2device(batch, self.device)
        if len(batch) == 12:
            (
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
            ) = batch
        elif len(batch) == 14:
            (
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
                y_s,
                y_h,
            ) = batch
        else:
            raise RuntimeError
        out_s, out_h = self.model(
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
        return out_s, out_h
