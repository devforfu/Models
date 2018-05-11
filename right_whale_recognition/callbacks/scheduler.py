from .base import Callback


class Schedule(Callback):

    def __init__(self, schedule_on_batch=False):
        super().__init__()
        self.schedule_on_batch = schedule_on_batch

    def update_rate(self, epoch, batch_index):
        pass


class ExpoDecay(Callback):

    def __init__(self, schedule_on_batch=False):
        super().__init__()
        self.batch = schedule_on_batch

    def set_model(self, model):
        self.observed_model = model

    def on_epoch_start(self, epoch):
        pass

    def on_batch_start(self, epoch, batch_index):
        pass
