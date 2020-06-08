from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip

from tensorflow.keras import backend as K

channel_axis = 1 if K.image_data_format() == 'channels_first' else -1


def preprocess_imagenet(x):
    # scale images to a range (-1, +1)
    x = Rescaling(scale=1 / 127.5, offset=-1, name='rescaling')(x)
    return x


def preprocess_cifar(x, data_samples):
    norm_layer = Normalization(axis=channel_axis, name='mean_normalization')
    # compute mean and std of images
    norm_layer.adapt(data_samples)

    # mean normalize the images
    x = norm_layer(x)
    return x


def baseline_augmentation(x):
    x = RandomFlip(mode='horizontal', name='h_flip')(x)
    return x
