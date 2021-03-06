from collections import defaultdict, OrderedDict

import tensorflow as tf

from callbacks.base import CallbacksGroup
from utils import get_collection, ArrayBatchGenerator


class Classifier:
    """Base class for all tensorflow-based classifiers."""

    def __init__(self):
        self.training = True
        self.scheduler = None
        self._graph = None
        self._session = None
        self._saver = None
        self._learning_rate = None
        self._fit_params = {}

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._session

    @property
    def saver(self):
        return self._saver

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value

    def get_fit_parameter(self, name):
        if name not in self._fit_params:
            raise KeyError(
                'unknown fit parameter: %s. Please check if the model '
                'fitting was appropriately invoked' % name)
        return self._fit_params[name]

    def generate_feed(self, tensors, **values):
        """Creates a dictionary with tensors and their values to be fed
        into session's training method.

        Default implementation uses X and y batch, learning rate value and
        training flags to create required dictionary.
        """
        feed = {}
        for key, value in values.items():
            if key in tensors:
                feed[tensors[key]] = value
        return feed

    def build(self, optimizer=None, graph=None):
        """Builds model.

        Default building procedure expects inheriting classes to override
        `create_inputs`, `build_model` and `create_optimizer` methods which
        should create tensors required for model.

        Args:
            optimizer: Optimizer class object.
            graph: Graph where model's tensors are added.

        """
        if graph is None:
            graph = tf.Graph()
        with graph.as_default():
            self.create_inputs()
            self.create_model()
            self.create_optimizer(optimizer)
        self._graph = graph

    def fit(self, X, y, epochs, lr0=1e-4, batch_size=32,
            validation_data=None, callbacks=None):

        generator = ArrayBatchGenerator(
            X, y, infinite=True,
            batch_size=batch_size,
            shuffle_on_restart=True)
        return self.fit_generator(
            generator, epochs, generator.n_batches, lr0,
            validation_data, callbacks)

    def fit_generator(self, generator, epochs, batches_per_epoch,
                      lr0=10e-4, validation_data=None,
                      callbacks=None):

        if self._session is not None:
            self._session.close()

        self._learning_rate = lr0
        self._fit_params['total_epochs'] = epochs
        self._fit_params['batches_per_epoch'] = batches_per_epoch

        graph = self.graph
        with graph.as_default():
            inputs = get_collection('inputs')
            metrics = get_collection('metrics')
            training_op = get_collection('training')
            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        if callbacks is None:
            callbacks = []

        history = []
        callbacks = CallbacksGroup(callbacks)
        self._session = session = tf.Session(graph=graph)
        init.run(session=session)
        callbacks.set_model(self)
        callbacks.on_training_start()
        for epoch in range(1, epochs + 1):
            stats = OrderedDict()
            stats['epoch'] = epoch
            epoch_metrics = defaultdict(lambda: 0)
            callbacks.on_epoch_start(epoch)
            for batch_index in range(batches_per_epoch):
                callbacks.on_batch_start(epoch, batch_index)
                stats['learning_rate'] = self._learning_rate
                x_batch, y_batch = next(generator)
                feed = self.generate_feed(
                    tensors=inputs, x=x_batch, y=y_batch,
                    training=True, learning_rate=self._learning_rate)
                session.run([training_op], feed)
                batch_metrics = session.run(metrics, feed)
                callbacks.on_batch_end(epoch, batch_index, batch_metrics)
                for metric, value in batch_metrics.items():
                    epoch_metrics[metric] += value
            for k, v in epoch_metrics.items():
                stats[k] = v/batches_per_epoch
            if validation_data:
                x_val, y_val = validation_data
                feed = self.generate_feed(
                    inputs, x=x_val, y=y_val, training=False)
                valid_metrics = session.run(metrics, feed)
                for metric, value in valid_metrics.items():
                    stats['val_' + metric] = value
            history.append(stats)
            callbacks.on_epoch_end(epoch, stats)
        callbacks.on_training_end()
        return history

    def score(self, X, y):
        """Scores model quality on testing dataset, returning dictionary with
        model's metrics.
        """
        self._check_if_session_exists()

        graph = self.graph
        with graph.as_default():
            inputs = get_collection('inputs')
            metrics = get_collection('metrics')

        feed = self.generate_feed(tensors=inputs, x=X, y=y, training=False)
        scores = self._session.run(metrics, feed)
        return scores

    def predict_proba(self, X):
        """Predicts classes probabilities for dataset."""

        self._check_if_session_exists()

        graph = self.graph
        with graph.as_default():
            inputs = get_collection('inputs')
            model = get_collection('model')
            probabilities = model['probabilities']

        feed = self.generate_feed(tensors=inputs, x=X, training=False)
        probs = self._session.run(probabilities, feed)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        classes = probs.argmax(axis=1)
        return classes

    def create_inputs(self):
        """Creates placeholders required to feed into model."""
        raise NotImplementedError()

    def create_model(self):
        """Creates model using input tensors."""
        raise NotImplementedError()

    def create_optimizer(self, optimizer):
        """Creates model optimizer."""
        raise NotImplementedError()

    def _check_if_session_exists(self):
        if self._session is None:
            raise RuntimeError('cannot score model until it is not fit')
