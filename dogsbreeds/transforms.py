from os.path import exists

import numpy as np
from swissknife.images import compute_featurewise_mean_and_std
from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator

from basedir import TRAIN_IMAGES, VALID_IMAGES


def augment_images(width_shift=0.2,
                   height_shift=0.2,
                   zoom=0.2,
                   rotation=30,
                   vertical_flip=False,
                   horizontal_flip=False):

    transformer = ImageDataGenerator(
            width_shift_range=width_shift,
            height_shift_range=height_shift,
            zoom_range=zoom,
            rotation_range=rotation,
            vertical_flip=vertical_flip,
            horizontal_flip=horizontal_flip)

    iterator = None

    while True:
        x, y = yield
        if iterator is None:
            iterator = NumpyArrayIterator(
                x, y, transformer,
                batch_size=len(x),
                shuffle=False,
                seed=None,
                data_format=transformer.data_format)
        else:
            iterator.n = x.shape[0]
            iterator.x = x
            iterator.y = y
        transformed = next(iterator)
        yield transformed


def normalize_images(target_size: tuple):
    """Applies sample-wise mean and std normalization to batch of images."""

    filename = 'stats_%s_%s_%s.npz' % target_size
    if not exists(filename):
        mean, std = compute_featurewise_mean_and_std(
            target_size, TRAIN_IMAGES, VALID_IMAGES)
        with open(filename, 'wb') as fp:
            np.savez(fp, mean=mean, std=std)
    else:
        with open(filename, 'rb') as fp:
            stats = np.load(fp)
            mean, std = stats['mean'], stats['std']

    while True:
        x, y = yield
        x -= mean
        x /= (std + 1e-07)
        yield x, y


def take(n):
    """Picks only nth element from tuple sent into generator."""

    while True:
        items = yield
        if hasattr(items, '__len__'):
            if len(items) > n:
                yield items[n]
            else:
                raise ValueError(
                    'Cannot take nth element from sequence of length %d',
                    len(items))
        else:
            yield items


def shuffle_samples():
    """Randomly shuffles batch of images."""

    while True:
        x, y = yield
        index = np.random.permutation(len(x))
        yield x[index], y[index]


def apply_to_samples(func):
    """Applies a function to each sample in (x, y) batch, leaving
    target values without changes.
    """
    while True:
        x, y = yield
        transformed = func(x)
        yield transformed, y
