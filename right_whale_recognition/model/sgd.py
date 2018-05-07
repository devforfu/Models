import numpy as np
import tensorflow as tf
from sklearn.utils import compute_class_weight

from data import get_mnist
from model.base import Model
from callbacks import StreamLogger
from utils import add_to_collection, get_collection, cross_entropy


class LogisticClassifier(Model):

    def __init__(self, n_features, n_classes, class_weights='balanced',
                 alpha=0.001, l1_ratio=0.15):

        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.class_weights_ = None

    def create_inputs(self):
        with tf.name_scope('inputs'):
            tensors = [
                tf.placeholder(
                    tf.float32, shape=(None, self.n_features), name='x'),
                tf.placeholder(
                    tf.float32, shape=(None, self.n_classes), name='y'),
                tf.placeholder(
                    tf.float32, shape=(self.n_classes,), name='class_weights'),
                tf.placeholder(tf.float32, name='learning_rate')]
            add_to_collection('inputs', *tensors)

    def create_model(self):
        inputs = get_collection('inputs')
        with tf.name_scope('model'):
            init = tf.truncated_normal((self.n_features, self.n_classes))
            theta = tf.Variable(init, name='theta')
            bias = tf.Variable(0.0, name='bias')
            logits = tf.add(tf.matmul(inputs['x'], theta), bias, name='logits')
            activate = tf.nn.sigmoid if self.n_classes == 2 else tf.nn.softmax
            probabilities = activate(logits, name='probabilities')
            predictions = tf.argmax(probabilities, axis=1, name='predictions')
        add_to_collection('model', theta, logits, probabilities, predictions)

    def create_optimizer(self, optimizer=None):
        model = get_collection('model')
        inputs = get_collection('inputs')

        alpha, l1_ratio = self.alpha, self.l1_ratio
        (x, y, class_weights, learning_rate, theta,
         logits, probabilities, predictions) = (
            inputs['x'],
            inputs['y'],
            inputs['class_weights'],
            inputs['learning_rate'],
            model['theta'],
            model['logits'],
            model['probabilities'],
            model['predictions']
        )

        with tf.name_scope('metrics'):
            xe = cross_entropy(self.n_classes, logits=logits, labels=y)
            loss = tf.reduce_mean(xe, name='loss')
            weights = tf.reduce_sum(class_weights * y, axis=1)
            weighted_loss = tf.reduce_mean(xe * weights, name='weighted_loss')
            penalty = elastic_net(theta, l1_ratio=l1_ratio)
            penalized_loss = tf.add(
                weighted_loss, alpha*penalty, name='penalized_loss')
            targets = tf.argmax(y, axis=1, name='targets')
            match = tf.cast(tf.equal(predictions, targets), tf.float32)
            accuracy = tf.reduce_mean(match, name='accuracy')
        add_to_collection('metrics', loss, penalized_loss, accuracy)

        with tf.name_scope('training'):
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            training_op = opt.minimize(penalized_loss)
        add_to_collection('training', training_op)

    def fit(self, X, y, epochs, **params):
        labels = y.argmax(axis=1)
        classes = np.unique(labels)
        if self.class_weights == 'balanced':
            class_weights = compute_class_weight(
                class_weight='balanced', classes=classes, y=labels)
            self.class_weights_ = class_weights
        else:
            assert len(classes) == len(self.class_weights)
            self.class_weights_ = self.class_weights
        return super().fit(X, y, epochs, **params)

    def generate_feed(self, tensors, **values):
        feed = super().generate_feed(tensors, **values)
        class_weights = tensors['class_weights']
        feed[class_weights] = self.class_weights_
        return feed


def elastic_net(theta, l1_ratio):
    l1 = tf.norm(theta, ord=1, name='l1_norm')
    l2 = tf.norm(theta, ord=2, name='l2_norm')
    return tf.add(
        tf.multiply(l1_ratio, l1),
        tf.multiply(1 - l1_ratio, l2),
        name='elastic_net')


def main():
    dataset = get_mnist()
    x_train, y_train = dataset['train']
    num_features = x_train.shape[1]
    num_classes = dataset['n_classes']

    model = LogisticClassifier(num_features, num_classes)

    model.build()

    model.fit(
        X=x_train,
        y=y_train,
        batch_size=1000,
        epochs=100,
        lr0=0.01,
        validation_data=dataset['valid'],
        callbacks=[StreamLogger()])


if __name__ == '__main__':
    main()
