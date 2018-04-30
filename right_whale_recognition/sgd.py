import numpy as np
import tensorflow as tf
from prettytable import PrettyTable
from sklearn.metrics import log_loss
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.examples.tutorials.mnist import input_data

from basedir import MNIST_IMAGES
from data import get_features, get_mnist


class ArrayBatchGenerator:
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
    first, *rest = arrays
    n = len(first)
    for arr in rest:
        if len(arr) != n:
            return False
    return True


def elastic_net(theta, l1_ratio):
    l1 = tf.norm(theta, ord=1, name='l1_norm')
    l2 = tf.norm(theta, ord=2, name='l2_norm')
    return tf.add(
        tf.multiply(l1_ratio, l1),
        tf.multiply(1 - l1_ratio, l2),
        name='elastic_net')


def learning_rate_scheduler(init_value, power=0.5):
    def schedule(step):
        return init_value / (step ** power)
    return schedule


def main():
    dataset = get_features('all_features.npz')
    model_name = 'whales_features'

    # dataset = get_mnist()
    # model_name = 'mnist'

    x_train, y_train = dataset['train']
    x_valid, y_valid = dataset['valid']
    num_samples = x_train.shape[0]
    num_features = x_train.shape[1]
    num_classes = dataset['n_classes']
    labels = y_train.argmax(axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(labels), y=labels)

    batch_size = 128
    epochs = 50
    init_lr = 5.0
    alpha = 1./num_samples
    l1_ratio = 0.15
    no_improvement_threshold = 1000
    schedule = learning_rate_scheduler(init_lr)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(None, num_features), name='x')
        y = tf.placeholder(tf.float32, shape=(None, num_classes), name='y')
        lr = tf.placeholder(tf.float32, name='learning_rate')

    with tf.name_scope('model'):
        init = tf.truncated_normal((num_features, num_classes))
        theta = tf.Variable(init, name='theta')
        bias = tf.Variable(0.0, name='bias')
        logits = tf.matmul(x, theta) + bias

    with tf.name_scope('metrics'):
        xe = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        weights = tf.reduce_sum(class_weights * y, axis=1)
        plain_loss = tf.reduce_mean(xe, name='plain_loss')
        loss = tf.reduce_mean(xe * weights, name='loss')

        penalty = elastic_net(theta, l1_ratio=l1_ratio)
        penalized_loss = tf.add(loss, alpha*penalty, name='penalized_loss')

        probabilities = tf.nn.softmax(logits, name='probabilities')
        predictions = tf.argmax(probabilities, axis=1, name='predictions')
        targets = tf.argmax(y, axis=1, name='targets')
        match = tf.cast(tf.equal(predictions, targets), tf.float32)
        accuracy = tf.reduce_mean(match, name='accuracy')

    with tf.name_scope('train'):
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        training_op = opt.minimize(penalized_loss)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        best_loss = np.inf
        no_improvement = 0
        stop_training = False

        for epoch in range(1, epochs + 1):
            if stop_training:
                break
            epoch_loss = 0.0
            learning_rate = schedule(epoch)
            index = np.random.permutation(num_samples)
            x_train = x_train[index]
            y_train = y_train[index]
            generator = ArrayBatchGenerator(
                x_train, y_train, batch_size=batch_size, infinite=False)
            for batch_index in range(generator.n_batches):
                x_batch, y_batch = next(generator)
                feed = {x: x_batch, y: y_batch, lr: learning_rate}
                _, batch_loss = session.run([training_op, loss], feed)
                epoch_loss += batch_loss
            epoch_loss /= generator.n_batches
            feed = {x: x_valid, y: y_valid, lr: learning_rate}
            val_acc, val_loss = session.run([accuracy, loss], feed)
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

        saver.save(session, f'model/{model_name}')

        if 'test' in dataset:
            x_test, y_test = dataset['test']
        else:
            x_test, y_test = x_valid, y_valid
        tensors = [plain_loss, accuracy]
        train_loss, train_acc = session.run(tensors, {x: x_train, y: y_train})
        test_loss, test_acc = session.run(tensors, {x: x_test, y: y_test})

        table = PrettyTable(field_names=['', 'train', 'test'])
        table.add_row(['accuracy', f'{train_acc:2.2%}', f'{test_acc:2.2%}'])
        table.add_row(['loss', f'{train_loss:2.6f}', f'{test_loss:2.6f}'])
        print(table)


if __name__ == '__main__':
    main()
