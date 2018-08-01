import os
import json
import shutil
from datetime import datetime
from os.path import join, exists

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import MaxNorm
from keras.models import Model, load_model
from keras.layers import Flatten, Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers import GlobalAvgPool2D, GlobalMaxPool2D

from utils import path
from legacy import LeakyReLU
from config import NUM_OF_LANDMARKS
from generators import AnnotatedImagesGenerator


def reset_session():
    tf.reset_default_graph()
    session = tf.InteractiveSession()
    K.set_session(session)
    K.set_image_dim_ordering('tf')


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def load_custom_model(path):
    return load_model(path, custom_objects={
        'root_mean_squared_error': root_mean_squared_error})


class BaseLandmarksModel:

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.subfolder = None
        self.history_path = None
        self.weights_path = None
        self.parameters_path = None
        self._keras_model = None
        self._prep_fn = None
        self._parameters = None

    def create(self, *args, **kwargs):
        raise NotImplementedError()

    def compile(self, optimizer):
        self._keras_model.compile(
            loss=root_mean_squared_error, optimizer=optimizer)

    def train(self, train_folder: str, valid_folder: str=None,
              n_epochs: int=100, batch_size: int=32, callbacks: list=None,
              augment: bool=True, shuffle: bool=True, normalize: bool=True):

        train_gen = AnnotatedImagesGenerator(
            root=train_folder,
            batch_size=batch_size,
            target_size=self.input_shape[:2],
            model_preprocessing=self._prep_fn,
            normalize=normalize,
            augment=augment)

        if valid_folder is None:
            valid_gen = None
            valid_steps = None
        else:
            valid_gen = AnnotatedImagesGenerator(
                root=valid_folder,
                batch_size=batch_size,
                target_size=self.input_shape[:2],
                model_preprocessing=self._prep_fn,
                normalize=normalize,
                augment=False)
            valid_steps = valid_gen.n_batches

        self._keras_model.fit_generator(
            generator=train_gen,
            steps_per_epoch=train_gen.n_batches,
            validation_data=valid_gen,
            validation_steps=valid_steps,
            epochs=n_epochs,
            callbacks=callbacks,
            shuffle=shuffle)

    def predict(self, X, image_coordinates: bool=True):
        y_pred = self._keras_model.predict(X)
        if image_coordinates:
            target_size = self.input_shape[:2]
            assert target_size[0] == target_size[1]
            shift = target_size[0] / 2
            y_pred *= shift
            y_pred += shift
        return y_pred

    def predict_generator(self, folder, image_coordinates: bool=True):
        gen = AnnotatedImagesGenerator(root=folder)
        y_pred = self._keras_model.predict_generator(gen, steps=gen.n_batches)
        if image_coordinates:
            y_pred = self._rescale_landmarks(y_pred)
        return y_pred

    def score(self, folder):
        gen = AnnotatedImagesGenerator(
            root=folder,
            target_size=self.input_shape[:2],
            infinite=False,
            augment=False,
            normalize=False,
            same_size_batches=False,
            model_preprocessing=self._prep_fn)

        losses = [
            self._keras_model.evaluate(*batch, verbose=0)
            for batch in gen]

        avg_loss = np.mean(losses)
        return avg_loss

    def create_model_folder(self, root: str, subfolder: str=None):
        if subfolder is None:
            timestamp = datetime.now().strftime('%s')
            folder = path(root, timestamp)
        else:
            folder = path(root, subfolder)

        template = 'weights_{epoch:03d}_{val_loss:2.4f}.hdf5'
        history_path = join(folder, 'history.csv')
        weights_path = join(folder, template)
        parameters_path = join(folder, 'parameters.json')

        if exists(folder):
            print('Model folder already exists. It will be deleted.', end=' ')
            while True:
                print('Proceed? [y/n]')
                response = input().lower().strip()
                if response in ('y', 'n'):
                    if response == 'n':
                        return False
                    else:
                        shutil.rmtree(folder)

        os.makedirs(folder, exist_ok=True)
        self.subfolder = folder
        self.history_path = history_path
        self.weights_path = weights_path
        self.parameters_path = parameters_path
        return True

    def save_parameters(self, filename):
        if self._parameters is None:
            return
        with open(filename, 'w') as file:
            json.dump(self._parameters, file)

    def _rescale_landmarks(self, y):
        target_size = self.input_shape[:2]
        assert target_size[0] == target_size[1]
        shift = target_size[0] / 2
        return (y + 1)*shift


class PretrainedModel(BaseLandmarksModel):

    def create(self, model_fn, prep_fn=None, pool: str='flatten',
               n_dense: int=5, units: int=500,
               n_outputs: int=NUM_OF_LANDMARKS*2, freeze: bool=True,
               bn: bool=True, dropouts: float=0.25, maxnorm: int=3,
               l2_reg: float=0.001):

        units = _as_list(units, n_dense, extend=True)
        dropouts = _as_list(dropouts, n_dense)

        base = model_fn(self.input_shape)
        if freeze:
            for layer in base.layers:
                layer.trainable = False

        x = _create_pool_layer(pool)(base.output)
        for i, n_units in enumerate(units):
            x = Dense(
                units=n_units,
                kernel_regularizer=_create_if_not_none(l2, l2_reg),
                kernel_constraint=_create_if_not_none(MaxNorm, maxnorm))(x)
            if dropouts:
                if i < len(dropouts):
                    x = Dropout(dropouts[i])(x)
            if bn:
                x = BatchNormalization()(x)
            x = LeakyReLU()(x)

        classifier = Dense(
            units=NUM_OF_LANDMARKS*2,
            activation='linear',
            kernel_regularizer=_create_if_not_none(l2, l2_reg),
            kernel_constraint=_create_if_not_none(MaxNorm, maxnorm))(x)

        parameters = dict(
            pool=pool, n_dense=n_dense, units=units, n_outputs=n_outputs,
            freeze=freeze, bn=bn, dropouts=dropouts, maxnorm=maxnorm,
            l2_reg=l2_reg)

        model = Model(inputs=base.input, outputs=classifier)
        self._keras_model = model
        self._prep_fn = prep_fn
        self._parameters = parameters


def _create_pool_layer(name: str):
    try:
        layer = {
            'flatten': Flatten,
            'avg': GlobalAvgPool2D,
            'max': GlobalMaxPool2D
        }[name]
    except KeyError:
        raise ValueError('unexpected pool layer: %s' % name)
    else:
        return layer()


def _create_if_not_none(cls, value):
    return None if value is None else cls(value)


def _as_list(value, length, extend=False):
    if not value:
        return value
    if isinstance(value, int):
        value = [value] * length
    elif hasattr(value, '__len__'):
        value = list(value)
        if extend:
            while len(value) < length:
                value.append(value[-1])
    return value
