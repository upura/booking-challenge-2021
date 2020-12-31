from catalyst import metrics
from catalyst.dl import Runner
from catalyst.dl.utils import any2device
import torch


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        x, y, cat = batch
        out, hidden = self.model(x, cat)
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
        if len(batch) == 2:
            x, cat = batch
        elif len(batch) == 3:
            x, y, cat = batch
        else:
            raise RuntimeError
        out, hidden = self.model(x, cat)
        return out
