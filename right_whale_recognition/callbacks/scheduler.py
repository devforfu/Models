from math import exp

from .base import Callback


class Schedule(Callback):

    def __init__(self, schedule_on_batch=False):
        super().__init__()
        self.schedule_on_batch = schedule_on_batch

    def on_epoch_end(self, epoch, metrics):
        if self.schedule_on_batch:
            return
        self.update_rate(epoch)

    def on_batch_end(self, epoch, batch_index, metrics):
        if self.schedule_on_batch:
            self.update_rate(epoch, batch_index)

    def update_rate(self, epoch, batch_index=None):
        t = epoch
        if self.schedule_on_batch:
            n = self.observed_model.get_fit_parameter('batches_per_epoch')
            t = t*n + batch_index
        lr = self._schedule(t)
        self.observed_model.learning_rate = lr

    def _schedule(self, t):
        raise NotImplementedError()


class ExpoDecay(Schedule):
    """Exponential decay of learning rate."""

    def __init__(self, decay=0.05, **schedule_params):
        super().__init__(**schedule_params)
        self.decay = decay
        self.init_rate = None

    def on_training_start(self):
        self.init_rate = self.observed_model.learning_rate

    def _schedule(self, t):
        return self.init_rate*exp(-self.decay*t)
