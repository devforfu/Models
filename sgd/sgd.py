from os.path import expanduser

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


MNIST_IMAGES = expanduser('~/data/mnist')


def main():
    # dataset = get_synthetic(classes=3)
    dataset = get_mnist()
    x_train, y_train = dataset['train']
    x_valid, y_valid = dataset['valid']
    x_test, y_test = dataset['test']
    num_samples = x_train.shape[0]
    num_features = x_train.shape[1]
    num_classes = dataset['n_classes']

    batch_size = 128
    epochs = 100
    init_lr = 1.0
    no_improvement_threshold = 10

    x, y, lr = create_inputs(num_features, num_classes)
    logits, metrics = create_model(x, y, lr)
    training_op = create_optimizer(metrics['loss'], lr)
    acc_and_loss = [metrics['accuracy'], metrics['loss']]
    init = tf.global_variables_initializer()
    schedule = learning_rate_scheduler(init_lr)

    with tf.Session() as session:
        session.run(init)
        best_loss = np.inf
        no_improvement = 0
        stop_training = False
        history = []

        for epoch in range(1, epochs + 1):
            if stop_training:
                break

            epoch_loss = 0.0
            learning_rate = schedule(epoch)
            index = np.random.permutation(num_samples)
            gen = ArrayBatchGenerator(
                x_train[index], y_train[index],
                batch_size=batch_size, infinite=False)
            for batch_index in range(gen.n_batches):
                x_batch, y_batch = next(gen)
                feed = {x: x_batch, y: y_batch, lr: learning_rate}
                _, batch_loss = session.run(
                    [training_op, metrics['loss']], feed)
                epoch_loss += batch_loss
            epoch_loss /= gen.n_batches
            feed = {x: x_valid, y: y_valid, lr: learning_rate}
            predictions = metrics['predictions'].eval(feed)
            history.append(predictions)
            val_acc, val_loss = session.run(acc_and_loss, feed)
            print(f'Epoch {epoch:5d} - '
                  f'lr: {learning_rate:2.6f} - '
                  f'loss: {epoch_loss:2.6f} - '
                  f'val_acc: {val_acc:2.2%} - '
                  f'val_loss: {val_loss:2.6f}')
            if val_loss < best_loss:
                best_loss = val_loss
                no_improvement = 0
            elif no_improvement < no_improvement_threshold:
                no_improvement += 1
            else:
                print('Early stopping...')
                break

        feed = {x: x_test, y: y_test}
        test_acc, test_loss = session.run(acc_and_loss, feed)
        print(f'Testing loss: {test_loss:2.4f}')
        print(f'Testing accuracy: {test_acc:2.2%}')

    np.save('history.npy', np.array(history))


def get_mnist(path=MNIST_IMAGES, target_encoded=True, scaled=True):
    """MNIST dataset to validate correctness of SGD training process."""

    mnist = input_data.read_data_sets(path)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels
    if scaled:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)
    if target_encoded:
        encoder = OneHotEncoder()
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_valid = encoder.transform(y_valid.reshape(-1, 1)).toarray()
        y_test = encoder.transform(y_test.reshape(-1, 1)).toarray()
    return {'train': (x_train, y_train),
            'valid': (x_valid, y_valid),
            'test': (x_test, y_test),
            'n_classes': 10}


def get_synthetic(random_state=1, classes=2):
    """Generates a random classification dataset with specific number of
    classes.
    """
    X, y = make_classification(n_samples=2000,
                               n_features=2,
                               n_redundant=0,
                               n_classes=classes,
                               n_clusters_per_class=1,
                               random_state=random_state)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state)

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=0.5, stratify=y_holdout,
        random_state=random_state)

    encoder = OneHotEncoder()
    encoder.fit(y_train.reshape(-1, 1))

    return {'train': (X_train, onehot(encoder, y_train)),
            'valid': (X_valid, onehot(encoder, y_valid)),
            'test': (X_test, onehot(encoder, y_test)),
            'n_classes': classes}


def onehot(encoder, arr):
    return encoder.transform(arr.reshape(-1, 1)).toarray()


def create_inputs(num_features, num_classes):
    """Creates placeholders required to train model."""

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(None, num_features), name='x')
        y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
        lr = tf.placeholder(tf.float32, name='learning_rate')
    return x, y, lr


def create_model(*inputs):
    """Creates optimization target and performance metrics."""
    x, y, lr = inputs

    with tf.name_scope('model'):
        num_features = x.shape.as_list()[1]
        num_classes = y.shape.as_list()[1]
        init = tf.truncated_normal((num_features, num_classes))
        theta = tf.Variable(init, name='theta')
        bias = tf.Variable(0.0, name='bias')
        logits = tf.matmul(x, theta) + bias

    with tf.name_scope('metrics'):
        xe = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xe, name='plain_loss')
        probabilities = tf.nn.softmax(logits, name='probabilities')
        predictions = tf.argmax(probabilities, axis=1, name='predictions')
        targets = tf.argmax(y, axis=1, name='targets')
        match = tf.cast(tf.equal(predictions, targets), tf.float32)
        accuracy = tf.reduce_mean(match, name='accuracy')
        metrics = {'loss': loss,
                   'accuracy': accuracy,
                   'predictions': predictions}

    return logits, metrics


def create_optimizer(loss, lr):
    """Creates an instance of model optimizer and returns training operation
    which should be invoked on each batch to update model's weights.
    """
    with tf.name_scope('train'):
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        training_op = opt.minimize(loss)
    return training_op


class ArrayBatchGenerator:
    """Iterates a group of arrays using batches of specific size."""

    def __init__(self, *arrays, same_size_batches=False,
                 batch_size=32, infinite=False):

        assert same_length(arrays)
        self.same_size_batches = same_size_batches
        self.batch_size = batch_size
        self.infinite = infinite

        total = len(arrays[0])
        n_batches = total // batch_size
        if same_size_batches and (total % batch_size != 0):
            n_batches += 1

        self.current_batch = 0
        self.n_batches = n_batches
        self.arrays = list(arrays)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.current_batch == self.n_batches:
            if self.infinite:
                self.current_batch = 0
            else:
                raise StopIteration()
        start = self.current_batch * self.batch_size
        batches = [arr[start:(start + self.batch_size)] for arr in self.arrays]
        self.current_batch += 1
        return batches


def same_length(*arrays):
    """Checks if all arrays have the same length."""

    first, *rest = arrays
    n = len(first)
    for arr in rest:
        if len(arr) != n:
            return False
    return True


def learning_rate_scheduler(init_value, power=0.5):
    """Exponential decay learning rate scheduler factory. Creates a function
    which accepts an epoch number and returns its learning rate.
    """
    def schedule(step):
        return init_value / (step ** power)
    return schedule


if __name__ == '__main__':
    main()
