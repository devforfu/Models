import re
from collections import defaultdict, OrderedDict

import numpy as np
import tensorflow as tf

from data import get_mnist


class Model:
    """
    Base class for all tensorflow-based models.
    """
    def __init__(self):
        self._graph = None
        self._session = None
        self._saver = None

    @property
    def graph(self):
        return self._graph

    @property
    def session(self):
        return self._session

    @property
    def saver(self):
        return self._saver

    def create_inputs(self):
        """Creates placeholders required to feed into model."""
        raise NotImplementedError()

    def build_model(self):
        """Creates model using input tensors."""
        raise NotImplementedError()

    def create_optimizer(self):
        """Creates model optimizer."""
        raise NotImplementedError()

    def generate_feed(self, batches, inputs, **params):
        """Creates a dictionary with tensors and their values to be fed
        into session.
        """
        raise NotImplementedError()

    def build(self, graph=None):
        if graph is None:
            graph = tf.Graph()
        with graph.as_default():
            self.create_inputs()
            self.build_model()
            self.create_optimizer()
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

        graph = self.graph
        with graph.as_default():
            inputs = get_collection('inputs')
            metrics = get_collection('metrics')
            training_op = get_collection('training')
            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        history = []
        self._session = session = tf.Session(graph=graph)
        init.run(session=session)
        for epoch in range(1, epochs + 1):
            stats = OrderedDict()
            stats['epoch'] = epoch
            epoch_metrics = defaultdict(lambda: 0)
            for batch_index in range(batches_per_epoch):
                batches = next(generator)
                feed = self.generate_feed(batches, inputs, lr=lr0)
                session.run([training_op], feed)
                batch_metrics = session.run(metrics, feed)
                for metric, value in batch_metrics.items():
                    epoch_metrics[metric] += value
            for k, v in epoch_metrics.items():
                stats[k] = v/batches_per_epoch
            if validation_data:
                feed = self.generate_feed(validation_data, inputs, train=False)
                valid_metrics = session.run(metrics, feed)
                for metric, value in valid_metrics.items():
                    stats['val_' + metric] = value
            history.append(stats)
            print(as_string(stats))

        return history


class DenseNetworkClassifer(Model):

    def __init__(self, input_size, config, n_classes,
                 optimizer=tf.train.GradientDescentOptimizer,
                 activation=tf.nn.relu):

        super().__init__()
        self.input_size = input_size
        self.config = config
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.activation = activation

    def create_inputs(self):
        with tf.name_scope('inputs'):
            tensors = [
                tf.placeholder(
                    tf.float32, shape=(None, self.input_size), name='x'),
                tf.placeholder(
                    tf.float32, shape=(None, self.n_classes), name='y'),
                tf.placeholder(tf.float32, name='learning_rate'),
                tf.placeholder(tf.bool, name='training')]
            add_to_collection('inputs', *tensors)

    def build_model(self):
        inputs = get_collection('inputs')
        with tf.name_scope('model'):
            x = inputs['x']
            for layer_config in self.config:
                layer = Dense(**layer_config)
                x = layer.build(x, training=inputs['training'])
            logits = tf.layers.dense(x, units=self.n_classes, name='logits')
            activate = tf.nn.sigmoid if self.n_classes == 2 else tf.nn.softmax
            probabilities = activate(logits, name='probabilities')
            predictions = tf.argmax(probabilities, axis=1, name='predictions')
        add_to_collection('model', logits, probabilities, predictions)

    def create_optimizer(self):
        model = get_collection('model')
        inputs = get_collection('inputs')

        x, y, logits, probabilities, predictions = (
            inputs['x'],
            inputs['y'],
            model['logits'],
            model['probabilities'],
            model['predictions']
        )

        with tf.name_scope('metrics'):
            xe = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y)
            loss = tf.reduce_mean(xe, name='loss')
            targets = tf.argmax(y, axis=1, name='targets')
            match = tf.cast(tf.equal(predictions, targets), tf.float32)
            accuracy = tf.reduce_mean(match, name='accuracy')
        add_to_collection('metrics', loss, accuracy)

        with tf.name_scope('training'):
            opt = self.optimizer(inputs['learning_rate'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                training_op = opt.minimize(loss)
        add_to_collection('training', training_op)

    def generate_feed(self, batches, inputs, train=True, lr=10e-4):
        x_batch, y_batch = batches
        x, y, learning_rate, training = inputs.values()
        feed = {x: x_batch, y: y_batch, learning_rate: lr, training: train}
        return feed


class Dense:
    """
    Fully-connected layer blueprint.
    """
    def __init__(self, units, dropout=None, batch_norm=True,
                 activation=tf.nn.relu, name=None):

        self.units = units
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.name = name

    def build(self, x, training=None):
        init = tf.variance_scaling_initializer(mode='fan_avg')
        x = tf.layers.dense(
            inputs=x, units=self.units,
            activation=None, kernel_initializer=init)
        if self.batch_norm:
            if training is None:
                raise ValueError(
                    'cannot add batch normalization without training switch')
            x = tf.layers.batch_normalization(x, training=training)
        if self.dropout is not None:
            x = tf.layers.dropout(x, rate=self.dropout)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ArrayBatchGenerator:
    def __init__(self, *arrays, same_size_batches=False, batch_size=32,
                 infinite=False, shuffle_on_restart=False):

        assert same_length(arrays)
        self.same_size_batches = same_size_batches
        self.batch_size = batch_size
        self.infinite = infinite
        self.shuffle_on_restart = shuffle_on_restart

        total = len(arrays[0])
        n_batches = total // batch_size
        if same_size_batches and (total % batch_size != 0):
            n_batches += 1

        self.array_size = len(arrays[0])
        self.current_batch = 0
        self.n_batches = n_batches
        self.arrays = list(arrays)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.current_batch == self.n_batches:
            if not self.infinite:
                raise StopIteration()
            self.current_batch = 0
            if self.shuffle_on_restart:
                index = np.random.permutation(self.array_size)
                self.arrays = [arr[index] for arr in self.arrays]

        start = self.current_batch * self.batch_size
        batches = [arr[start:(start + self.batch_size)] for arr in self.arrays]
        self.current_batch += 1
        return batches


def same_length(*arrays):
    first, *rest = arrays
    n = len(first)
    for arr in rest:
        if len(arr) != n:
            return False
    return True


def add_to_collection(name, *ops):
    """Combines group of tensors into collection."""
    for op in ops:
        tf.add_to_collection(name, op)


def get_collection(name):
    ops = OrderedDict()
    for op in tf.get_collection(name):
        ops[short_name(name, op.name)] = op
    if len(ops) == 1:
        return list(ops.values())[0]
    return ops


def short_name(collection_name, tensor_name):
    match = re.match(f'^{collection_name}/(\w+)(:\d)?', tensor_name)
    name, *_ = match.groups()
    return name


def as_string(dictionary, stats_formats=None):
    if stats_formats is None:
        stats_formats = OrderedDict()
        stats_formats['epoch'] = '05d'
        stats_formats['loss'] = '2.6f'
        stats_formats['val_loss'] = '2.6f'
        stats_formats['accuracy'] = '2.2%'
        stats_formats['val_accuracy'] = '2.2%'

    format_strings = [
        '%s: {%s:%s}' % (name, name, value)
        for name, value in stats_formats.items()]
    format_string = ' - '.join(format_strings)
    return format_string.format(**dictionary)


def main():
    dataset = get_mnist()
    x_train, y_train = dataset['train']
    num_features = x_train.shape[1]
    num_classes = dataset['n_classes']

    blueprint = [
        {'units': 300, 'dropout': 0.50},
        {'units': 300, 'dropout': 0.50},
        {'units': 200, 'dropout': 0.25},
        {'units': 200, 'dropout': 0.25},
        {'units': 100}]

    model = DenseNetworkClassifer(
        input_size=num_features,
        n_classes=num_classes,
        config=blueprint,
        optimizer=tf.train.AdamOptimizer,
        activation=tf.nn.elu)

    model.build()

    model.fit(
        X=x_train,
        y=y_train,
        batch_size=1000,
        epochs=200,
        lr0=0.01,
        validation_data=dataset['valid'])


if __name__ == '__main__':
    main()
