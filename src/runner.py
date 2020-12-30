from catalyst.dl import Runner
from catalyst.dl.utils import any2device
import torch


class CustomRunner(Runner):
    def _handle_batch(self, batch):
        x, y, cat = batch
        out, hidden = self.model(x)
        loss = self.criterion(out, y.view(y.size(0) * y.size(1)))
        self.batch_metrics = {'loss': loss}
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
        out, hidden = self.model(x)
        return out
