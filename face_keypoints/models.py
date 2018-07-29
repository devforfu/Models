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

    def __init__(self, input_shape, prep_fn):
        self.input_shape = input_shape
        self.folder = None
        self.history_path = None
        self.weights_path = None
        self._keras_model = None
        self._prep_fn = prep_fn

    def create(self, *args, **kwargs):
        raise NotImplementedError()

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

    def create_model_folder(self, root):
        timestamp = datetime.now().strftime('%s')
        folder = join(root, timestamp)
        history_path = join(folder, 'history.csv')
        weights_path = join(folder, 'weights_{epoch:03d}_{val_loss:2.4f}.hdf5')

        if not exists(folder):
            os.makedirs(folder, exist_ok=True)

        self.folder = folder
        self.history_path = history_path
        self.weights_path = weights_path


class PretrainedModel(BaseLandmarksModel):

    def create(self, model_fn, pool: str='flatten', n_dense: int=5,
               n_units: int=500, n_outputs: int=NUM_OF_LANDMARKS*2,
               freeze: bool=True, bn: bool=True, maxnorm: int=3,
               l2_reg: float=0.001):

        base = model_fn(self.input_shape)
        if freeze:
            for layer in base.layers:
                layer.trainable = False

        pool_layer = _create_pool_layer(pool)
        x = pool_layer(base.output)
        for _ in range(n_dense):
            x = Dense(
                units=n_units,
                activation='relu',
                kernel_regularizer=_create_if_not_none(l2, l2_reg))(x)
            if bn:
                x = BatchNormalization()(x)

        classifier = Dense(
            units=n_outputs,
            activation='linear',
            kernel_constraint=_create_if_not_none(l2, l2_reg))(x)

        self._keras_model = Model(inputs=base.inputs, outputs=classifier)


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
