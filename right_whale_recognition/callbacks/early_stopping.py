from operator import lt, gt

import numpy as np

from .base import Callback


class EarlyStopping(Callback):

    def __init__(self, metric='val_loss', minimize=True, patience=10):
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.minimize = minimize
        self._op = lt if self.minimize else gt
        self._no_improvement = 0
        self._prev_value = np.inf if minimize else -np.inf
        self._best_epoch = 0

    def on_epoch_end(self, epoch, metrics):
        metric_value = metrics.get(self.metric)
        if not metric_value:
            return

        if self._better(metric_value, self._prev_value):
            self._prev_value = metric_value
            self._no_improvement = 0
            self._best_epoch = epoch
        else:
            self._no_improvement += 1
            if self._no_improvement < self.patience:
                return
            self.observed_model.stop_training = True

    def _better(self, a, b):
        return self._op(a, b)
