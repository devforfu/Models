from os.path import expanduser

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import StandardScaler, OneHotEncoder


MNIST_IMAGES = expanduser('~/data/mnist')


def get_features(path, target_encoded=True, scaled=True):
    """Reads an NPZ (Numpy archive) file with training dataset features
    extracted using ImageNet-pretrained models.
    """
    dataset = np.load(path)
    x_train = dataset['x_train']
    y_train = dataset['y_train']
    x_valid = dataset['x_valid']
    y_valid = dataset['y_valid']
    if scaled:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_valid = scaler.transform(x_valid)
    if target_encoded:
        encoder = OneHotEncoder()
        y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_valid = encoder.transform(y_valid.reshape(-1, 1)).toarray()
    return {'train': (x_train, y_train),
            'valid': (x_valid, y_valid),
            'n_classes': 447}


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
