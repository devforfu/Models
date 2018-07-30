import os
from datetime import datetime
from os.path import join, exists

import tensorflow as tf
from keras import layers
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import MaxNorm
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.layers import Input, Flatten
from keras.layers import Dense, Conv2D, Activation
from keras.layers import Dropout, BatchNormalization
from keras.layers import GlobalAvgPool2D, GlobalMaxPool2D
from keras.layers import ZeroPadding2D, MaxPooling2D, AveragePooling2D

from utils import path
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
        self._keras_model = None
        self._prep_fn = None

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
        y_pred = self._keras_model.predict_generator(gen)
        if image_coordinates:
            y_pred = self._rescale_landmarks(y_pred)
        return y_pred

    def create_model_folder(self, root: str, subfolder: str=None):
        if subfolder is None:
            timestamp = datetime.now().strftime('%s')
            subfolder = path(root, timestamp)

        template = 'weights_{epoch:03d}_{val_loss:2.4f}.hdf5'
        history_path = join(subfolder, 'history.csv')
        weights_path = join(subfolder, template)
        if not exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)

        self.subfolder = subfolder
        self.history_path = history_path
        self.weights_path = weights_path

    def _rescale_landmarks(self, y):
        target_size = self.input_shape[:2]
        assert target_size[0] == target_size[1]
        shift = target_size[0] / 2
        return (y + 1)*shift


class PretrainedModel(BaseLandmarksModel):

    def create(self, model_fn, prep_fn=None, pool: str='flatten',
               n_dense: int=5, n_units: int=500,
               n_outputs: int=NUM_OF_LANDMARKS*2, freeze: bool=True,
               bn: bool=True, dropout: float=0.25, maxnorm: int=3,
               l2_reg: float=0.001):

        base = model_fn(self.input_shape)
        if freeze:
            for layer in base.layers:
                layer.trainable = False

        x = _create_pool_layer(pool)(base.output)
        for _ in range(n_dense):
            reg = _create_if_not_none(MaxNorm, maxnorm)
            x = Dense(units=n_units, kernel_regularizer=reg)(x)
            if dropout is not None:
                x = Dropout(dropout)(x)
            if bn:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

        classifier = Dense(units=NUM_OF_LANDMARKS*2, activation='linear')(x)
        model = Model(inputs=base.input, outputs=classifier)
        self._keras_model = model


        # base = model_fn(self.input_shape)
        # base.trainable = not freeze
        # x = _create_pool_layer(pool)(base.output)
        # x = Dense(units=500, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(units=500, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(units=500, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(units=NUM_OF_LANDMARKS * 2, activation='linear')(x)
        #
        # model = Model(inputs=base.input, outputs=x)
        #
        # self._keras_model = model

        # if freeze:
        #     for layer in base.layers:
        #         layer.trainable = False

        # pool_layer = _create_pool_layer(pool)
        # x = pool_layer(base.output)
        # for _ in range(n_dense):
        #     x = Dense(
        #         units=n_units,
        #         activation='relu',
        #         kernel_regularizer=_create_if_not_none(l2, l2_reg))(x)
        #     if bn:
        #         x = BatchNormalization()(x)
        #
        # classifier = Dense(
        #     units=n_outputs,
        #     activation='linear',
        #     kernel_constraint=_create_if_not_none(l2, l2_reg))(x)
        #
        # self._keras_model = Model(inputs=base.input, outputs=classifier)
        # self._prep_fn = prep_fn


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
