from catalyst import metrics
from catalyst.dl import Runner
from catalyst.dl.utils import any2device
import torch


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        x, y, cat, num = batch
        out, hidden = self.model(x, cat, num)
        loss = self.criterion(out, y.view(y.size(0) * y.size(1)))
        accuracy01, accuracy04 = metrics.accuracy(out, y.view(y.size(0) * y.size(1)), topk=(1, 4))
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
        if len(batch) == 3:
            x, cat, num = batch
        elif len(batch) == 4:
            x, y, cat, num = batch
        else:
            raise RuntimeError
        out, hidden = self.model(x, cat, num)
        return out


class CustomRunnerMtl(Runner):
    def _handle_batch(self, batch):
        x_s, y_s, x_h, y_h, cat, num = batch
        (out_s, out_h), hidden = self.model(x_s, x_h, cat, num)
        loss = self.criterion(out_s, y_s.view(y_s.size(0) * y_s.size(1))) + self.criterion(out_h, y_h.view(y_h.size(0) * y_h.size(1)))
        accuracy01, accuracy04 = metrics.accuracy(out_s, y_s.view(y_s.size(0) * y_s.size(1)), topk=(1, 4))
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
        if len(batch) == 4:
            x_s, x_h, cat, num = batch
        elif len(batch) == 6:
            x_s, y_s, x_h, y_h, cat, num = batch
        else:
            raise RuntimeError
        (out_s, out_h), hidden = self.model(x_s, x_h, cat, num)
        return (out_s, out_h)
