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
        if len(batch) == 9:
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
            ) = batch
        elif len(batch) == 10:
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
        )
        return out
