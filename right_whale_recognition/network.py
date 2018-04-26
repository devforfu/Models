from collections import defaultdict, OrderedDict

import tensorflow as tf


def add_to_collection(name, *ops):
    """Combines group of tensors into collection."""
    for op in ops:
        tf.add_to_collection(name, op)


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

    def build_model(self, inputs: dict):
        """Creates model using input tensors."""
        raise NotImplementedError()

    def create_optimizer(self, model, inputs):
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
            inputs = self.create_inputs()
            model = self.build_model(inputs)
            training_op, metrics = self.create_optimizer(model, inputs)
            add_to_collection('inputs', *inputs.values())
            add_to_collection('metrics', *metrics.values())
            add_to_collection('training', training_op, model)
        self._graph = graph

    def fit_generator(self, generator, epochs, batches_per_epoch,
                      lr0=10e-4, validation_data=None,
                      callbacks=None):

        if self._session is not None:
            self._session.close()

        graph = self.graph

        with graph.as_default():
            inputs = tf.get_collection('inputs')
            metrics = tf.get_collection('metrics')
            model, training_op = tf.get_collection('training')
            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        history = []
        self._session = session = tf.Session(graph=graph)
        init.run(session=session)
        for epoch in range(1, epochs + 1):
            stats = {'epoch': epoch}
            epoch_metrics = defaultdict(lambda: 0)
            for batch_index in range(batches_per_epoch):
                batches = next(generator)
                feed = self.generate_feed(batches, inputs, lr=lr0)
                session.run([training_op], feed)
                values = session.run(metrics, feed)
                for metric, value in zip(metrics, values):
                    epoch_metrics[metric.name] += value
            stats.update({k: v/batches_per_epoch for k, v in epoch_metrics})
            if validation_data:
                feed = self.generate_feed(validation_data, inputs, train=False)
                val_values = session.run(metrics, feed)
                val_metrics = {}
                for metric, value in zip(metrics, val_values):
                    val_metrics['val_' + metric.name] = value
                stats.update(val_metrics)
            history.append(stats)

        return history


class DenseNetworkClassifer(Model):

    def __init__(self, input_shape, config, n_classes,
                 optimizer=tf.train.GradientDescentOptimizer,
                 activation=tf.nn.relu):

        super().__init__()
        self.input_shape = input_shape
        self.config = config
        self.n_classes = n_classes
        self.optimizer = optimizer
        self.activation = activation

    def create_inputs(self):
        with tf.name_scope('inputs'):
            x = tf.placeholder(tf.float32,
                               shape=(None, self.input_shape),
                               name='x')
            y = tf.placeholder(tf.float32,
                               shape=(None,),
                               name='y')
            lr = tf.placeholder(tf.float32, name='learing_rate')
            training = tf.placeholder(tf.bool, name='training')
        inputs = OrderedDict([
            ('x', x),
            ('y', y),
            ('learning_rate', lr),
            ('training', training)
        ])
        return inputs

    def build_model(self, inputs: dict):
        with tf.name_scope('model'):
            x = inputs['x']
            for layer_config in self.config:
                layer = Dense(**layer_config)
                layer.build(x)
                x = layer
            logits = tf.layers.dense(x, units=self.n_classes, name='logits')
            activate = tf.nn.sigmoid if self.n_classes == 2 else tf.nn.softmax
            probabilities = activate(logits, name='predictions')
            predictions = tf.argmax(probabilities, axis=1, name='predictions')
        model = OrderedDict([
            ('logits', logits),
            ('probabilities', probabilities),
            ('predictions', predictions)
        ])
        return model

    def create_optimizer(self, model, inputs):
        x, y = inputs['x'], inputs['y']
        logits, probabilities, predictions = (
            model['logits'], model['probabilities'], model['predictions'])

        with tf.name_scope('metrics'):
            xe = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=y)
            loss = tf.reduce_mean(xe, name='loss')
            targets = tf.argmax(y, axis=1, name='targets')
            match = tf.cast(tf.equal(predictions, targets), tf.float32)
            accuracy = tf.reduce_mean(match, name='accuracy')

        metrics = OrderedDict([('loss', loss), ('accuracy', accuracy)])
        with tf.name_scope('train'):
            opt = self.optimizer(inputs['lr'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                training_op = opt.minimize(loss)

        return training_op, metrics

    def generate_feed(self, batches, inputs, train=True, lr=10e-4):
        x_batch, y_batch = next(batches)
        x, y, learning_rate, training = inputs.values()
        feed = {x: x_batch, y: y_batch, learning_rate: lr, training: train}
        return feed


class Dense:
    """
    Fully-connected layer blueprint.
    """
    def __init__(self, units, dropout=None, batch_norm=False,
                 activation=None, name=None):

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
            x = tf.layers.batch_normalization(x, training=training)
        x = self.activation(x)
        if self.dropout is not None:
            x = tf.layers.dropout(x, rate=self.dropout)
        return x


def main():
    pass


if __name__ == '__main__':
    main()
