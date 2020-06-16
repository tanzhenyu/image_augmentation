from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomCrop
from tensorflow.keras.layers import ZeroPadding2D

from tensorflow.keras import backend as K

from image_augmentation.image import RandomCutout

CHANNEL_AXIS = 1 if K.image_data_format() == 'channels_first' else -1


def cifar_standardization(x, data_samples):
    norm_layer = Normalization(axis=CHANNEL_AXIS, name='mean_normalization')
    # compute mean and std of images
    norm_layer.adapt(data_samples)

    # mean normalize the images
    x = norm_layer(x)
    return x


def cifar_baseline_augmentation(x):
    x = RandomFlip(mode='horizontal', name='h_flip')(x)
    x = ZeroPadding2D((4, 4), name='padding')(x)
    x = RandomCrop(32, 32, name='crop')(x)
    x = RandomCutout(16, name='cutout')(x)
    return x


def imagenet_standardization(x):
    # scale images to a range (-1, +1)
    x = Rescaling(scale=1 / 127.5, offset=-1, name='rescaling')(x)
    return x


def imagenet_baseline_augmentation(x):
    x = RandomFlip(mode='horizontal', name='h_flip')(x)
    return x
