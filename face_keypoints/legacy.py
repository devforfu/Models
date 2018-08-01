import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class LeakyReLU(Layer):
    """A Leaky version of a Rectified Linear Unit back-ported from the newer
    versions of Keras to work with older versions of TensorFlow, i.e., before
    the tf.nn.leaky_relu layer was introduced.
    """
    def __init__(self, alpha=0.3, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs, **kwargs):
        return tf.maximum(self.alpha * inputs, inputs)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(LeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
