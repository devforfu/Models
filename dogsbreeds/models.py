"""
Small wrapping utilities build on top of Keras models.

Each model is represented as an instance of some template class which tracks
information required to instantiate and train models, like preprocessing
function, input size, etc.
"""
import importlib

from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D


_registry = {}


def preprocessing(name):
    try:
        module = importlib.import_module(f'keras.applications.{name}')
    except ModuleNotFoundError:
        raise ValueError(
            f'there is no Keras application with name {name}')
    return getattr(module, 'preprocess_input', identity)


def get_model(name: str):
    if name == 'default':
        return _registry['vgg16_top_only_adam']
    if name not in _registry:
        raise ValueError('model not found: %s' % name)
    return _registry[name]


def model_names():
    return sorted(_registry.keys())


def identity(x):
    return x


class BaseModelTemplate:
    """Base class for all model templates."""

    def __init__(self, unique_id, pretrained=True,
                 optimizer='adam', target_size=(224, 224, 3),
                 **build_params):

        if unique_id in _registry:
            raise ValueError('model already exists: %s' % unique_id)

        self.unique_id = unique_id
        self.pretrained = pretrained
        self.optimizer = optimizer
        self.target_size = target_size
        self.build_params = build_params
        _registry[unique_id] = self

    @staticmethod
    def layers_status(model, n_dashes=65):
        print('Layers training status:')
        print('-' * n_dashes)
        for layer in model.layers:
            frozen_status = '' if layer.trainable else 'FROZEN'
            status = '[%6s] %s' % (frozen_status, layer.name)
            print(status)
        print('-' * n_dashes)

    @property
    def weights(self):
        return 'imagenet' if self.pretrained else None

    @property
    def preprocessing_function(self):
        return identity

    @property
    def name(self):
        return self.unique_id

    def build(self, *args, **kwargs):
        if self.build_params:
            kwargs.update(self.build_params)
        return self._build(*args, **kwargs)

    def _build(self, n_classes: int):
        raise NotImplementedError()


class FineTunedTemplate(BaseModelTemplate):
    """Template for model with fine-tuned layer."""

    @property
    def model_factory(self):
        raise NotImplementedError()

    def _build(self, n_classes: int, method='flatten', trainable_layers=1):
        base = self.model_factory(
            include_top=False,
            input_shape=self.target_size,
            weights=self.weights)

        x = base.output
        if method == 'flatten':
            x = Flatten(name='flatten')(x)
        elif method == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif method == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)
        else:
            raise ValueError('unexpected method \'%s\'' % method)

        x = Dense(n_classes, activation='softmax', name='top')(x)
        model = Model(inputs=base.input, outputs=x)
        for layer in model.layers[:-trainable_layers]:
            layer.trainable = False

        model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy')
        return model


class ResNet50Top(FineTunedTemplate):

    @property
    def model_factory(self):
        from keras.applications import ResNet50
        return ResNet50

    @property
    def preprocessing_function(self):
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input


class InceptionV3Top(FineTunedTemplate):

    @property
    def model_factory(self):
        from keras.applications import InceptionV3
        return InceptionV3

    @property
    def preprocessing_function(self):
        from keras.applications.inception_v3 import preprocess_input
        return preprocess_input


class InceptionResNetTop(FineTunedTemplate):

    @property
    def model_factory(self):
        from keras.applications import InceptionResNetV2
        return InceptionResNetV2

    @property
    def preprocessing_function(self):
        from keras.applications.inception_resnet_v2 import preprocess_input
        return preprocess_input


class DenseNetTop(FineTunedTemplate):

    @property
    def model_factory(self):
        from keras.applications import DenseNet201
        return DenseNet201

    @property
    def preprocessing_function(self):
        from keras.applications.densenet import preprocess_input
        return preprocess_input


class XceptionTop(FineTunedTemplate):

    @property
    def model_factory(self):
        from keras.applications import Xception
        return Xception

    @property
    def preprocessing_function(self):
        from keras.applications.xception import preprocess_input
        return preprocess_input


class NASNetTop(FineTunedTemplate):

    @property
    def model_factory(self):
        from keras.applications import NASNetLarge
        return NASNetLarge

    @property
    def preprocessing_function(self):
        from keras.applications.nasnet import preprocess_input
        return preprocess_input


# +---------------------------------------------------------------------------+
# |                             MODELS INSTANCES                              |
# +---------------------------------------------------------------------------+


ResNet50Top(
    unique_id='resnet50_sgd_m099',
    target_size=(224, 224, 3),
    method='avg',
    trainable_layers=1,
    optimizer=SGD(nesterov=True, momentum=0.99))

InceptionV3Top(
    unique_id='inception3_sgd_m099',
    target_size=(299, 299, 3),
    method='avg',
    optimizer=SGD(nesterov=True, momentum=0.99))

InceptionResNetTop(
    unique_id='inception_resnet_sgd_m099',
    target_size=(299, 299, 3),
    method='avg',
    optimizer=SGD(nesterov=True, momentum=0.99))

DenseNetTop(
    unique_id='densenet_sgd_m099_avg',
    target_size=(224, 224, 3),
    method='avg',
    optimizer=SGD(nesterov=True, momentum=0.99))

XceptionTop(
    unique_id='xception_sgd_m099_avg',
    target_size=(299, 299, 3),
    method='avg',
    optimizer=SGD(nesterov=True, momentum=0.99))

NASNetTop(
    unique_id='nasnet_sgd_m099_avg',
    target_size=(331, 331, 3),
    method='avg',
    optimizer=SGD(nesterov=True, momentum=0.99))
