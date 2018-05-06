class Callback:

    def __init__(self):
        self.observed_model = None

    def set_model(self, model):
        self.observed_model = model

    def on_training_start(self):
        pass

    def on_training_end(self):
        pass

    def on_epoch_start(self, epoch):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def on_batch_start(self, epoch, batch_index):
        pass

    def on_batch_end(self, epoch, batch_index, metrics):
        pass


class CallbacksGroup(Callback):

    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_training_start(self):
        for callback in self.callbacks:
            callback.on_training_start()

    def on_training_end(self):
        for callback in self.callbacks:
            callback.on_training_end()

    def on_epoch_start(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_start(epoch)

    def on_epoch_end(self, epoch, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)

    def on_batch_start(self, epoch, batch_index):
        for callback in self.callbacks:
            callback.on_batch_start(epoch, batch_index)

    def on_batch_end(self, epoch, batch_index, metrics):
        for callback in self.callbacks:
            callback.on_batch_end(epoch, batch_index, metrics)
