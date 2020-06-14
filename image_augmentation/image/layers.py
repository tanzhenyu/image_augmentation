from .image_ops import cutout

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class RandomCutout(Layer):
    def __init__(self,
                 size=16,
                 name=None,
                 **kwargs):
        self.size = size
        super(RandomCutout, self).__init__(name=name, **kwargs)

    def call(self, inputs, training=True):
        if training is None:
            training = K.learning_phase()

        if training:
            return tf.map_fn(lambda x: cutout(x, self.size), inputs)
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
