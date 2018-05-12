import re
from collections import OrderedDict

import numpy as np
import tensorflow as tf


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


def cross_entropy(num_classes, logits, labels, onehot=True):
    assert num_classes >= 2, 'Number of classes cannot be less then 2'
    if num_classes == 2:
        func = tf.nn.sigmoid_cross_entropy_with_logits
    elif onehot:
        func = tf.nn.softmax_cross_entropy_with_logits
    else:
        func = tf.nn.sparse_softmax_cross_entropy_with_logits
    return func(logits=logits, labels=labels)
