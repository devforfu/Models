import tensorflow as tf

from data import get_mnist
from model.base import Model
from utils import get_collection, add_to_collection, cross_entropy


class DenseNetworkClassifier(Model):

    def __init__(self, input_size, config, n_classes, activation=tf.nn.relu):
        super().__init__()
        self.input_size = input_size
        self.config = config
        self.n_classes = n_classes
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

    def create_model(self):
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

    def create_optimizer(self, optimizer=None):
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
            xe = cross_entropy(self.n_classes, logits=logits, labels=y)
            loss = tf.reduce_mean(xe, name='loss')
            targets = tf.argmax(y, axis=1, name='targets')
            match = tf.cast(tf.equal(predictions, targets), tf.float32)
            accuracy = tf.reduce_mean(match, name='accuracy')
        add_to_collection('metrics', loss, accuracy)

        with tf.name_scope('training'):
            if optimizer is None:
                optimizer = tf.train.GradientDescentOptimizer
            opt = optimizer(inputs['learning_rate'])
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                training_op = opt.minimize(loss)
        add_to_collection('training', training_op)


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

    model = DenseNetworkClassifier(
        input_size=num_features,
        n_classes=num_classes,
        config=blueprint,
        activation=tf.nn.elu)

    model.build(optimizer=tf.train.AdamOptimizer)

    model.fit(
        X=x_train,
        y=y_train,
        batch_size=1000,
        epochs=200,
        lr0=0.01,
        validation_data=dataset['valid'])


if __name__ == '__main__':
    main()
