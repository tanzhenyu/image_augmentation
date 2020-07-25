"""Data preprocessing and baseline augmentation."""

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomCrop
from tensorflow.keras.layers import ZeroPadding2D

from tensorflow.keras import backend as K

from image_augmentation.image import RandomCutout, ReflectPadding2D

CHANNEL_AXIS = 1 if K.image_data_format() == 'channels_first' else -1
CIFAR_MEAN = [125.3, 123.0, 113.9]
CIFAR_STD = [63.0, 62.1, 66.7]


def cifar_standardization(x, mode='FEATURE_NORMALIZE', data_samples=None):
    mode = mode.upper()
    assert mode in ['FEATURE_NORMALIZE', 'PIXEL_MEAN_SUBTRACT']

    if mode == 'PIXEL_MEAN_SUBTRACT' and data_samples == None:
        raise ValueError('`data_samples` argument should not be `None`, '
                         'when `mode="PIXEL_MEAN_SUBTRACT"`.')

    if mode == 'FEATURE_NORMALIZE':
        cifar_mean = tf.cast(CIFAR_MEAN, tf.float32)
        cifar_std = tf.cast(CIFAR_STD, tf.float32)

        x = Rescaling(scale=1 / cifar_std, offset=-(cifar_mean / cifar_std), name='mean_normalization')(x)
    elif mode == 'PIXEL_MEAN_SUBTRACT':
        mean_subtraction_layer = Normalization(axis=[1, 2, 3], name='pixel_mean_subtraction')
        mean_subtraction_layer.adapt(data_samples)

        # set variance=1. and keep mean values as is
        mean_pixels = mean_subtraction_layer.get_weights()[0]
        mean_subtraction_layer.set_weights([mean_pixels, tf.ones_like(mean_pixels)])

        x = mean_subtraction_layer(x)
        x = Rescaling(scale=1 / 255., name='rescaling')(x)
    return x


def cifar_baseline_augmentation(x, padding_mode='ZEROS', cutout=True):
    x = RandomFlip(mode='horizontal', name='h_flip')(x)

    padding_mode = padding_mode.upper()
    assert padding_mode in ['ZEROS', 'REFLECT']
    if padding_mode == 'ZEROS':
        x = ZeroPadding2D((4, 4), name='padding')(x)
    elif padding_mode == 'REFLECT':
        x = ReflectPadding2D((4, 4), name='padding')(x)

    x = RandomCrop(32, 32, name='crop')(x)

    if cutout:
        x = RandomCutout(16, 0.0, name='cutout')(x)
    return x


def imagenet_standardization(x):
    # scale images to a range (-1, +1)
    x = Rescaling(scale=1 / 127.5, offset=-1, name='rescaling')(x)
    return x


def imagenet_baseline_augmentation(x):
    x = RandomFlip(mode='horizontal', name='h_flip')(x)
    return x
