"""Pre-processing layers for few image op(s)."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils

from image_augmentation.image.image_ops import cutout


class RandomCutout(Layer):
    def __init__(self, size=16, color=128, name=None, **kwargs):
        super(RandomCutout, self).__init__(name=name, **kwargs)
        self.size = size
        self.color = color

    def call(self, inputs, training=True):
        with tf.name_scope(self.name or "RandomCutout"):
            if training is None:
                training = K.learning_phase()

            if training:
                return tf.map_fn(lambda x: cutout(x, self.size, self.color), inputs)
            else:
                return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "size": self.size
        }
        base_config = super(RandomCutout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectPadding2D(Layer):
    def __init__(self, padding=4, name=None, **kwargs):
        super(ReflectPadding2D, self).__init__(name=name, **kwargs)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        '1st entry of padding')
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
        cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
        return tf.TensorShape([
            input_shape[0], rows, cols, input_shape[3]])

    def call(self, inputs, *kwargs):
        # currently, only NHWC format is supported
        with tf.name_scope(self.name or "ReflectPadding2D"):
            pattern = [[0, 0], list(self.padding[0]), list(self.padding[1]), [0, 0]]
            return tf.pad(inputs, pattern, mode='REFLECT')

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
