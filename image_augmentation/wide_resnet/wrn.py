from tensorflow.keras.layers import Conv2D, Add
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import Input, Model

from tensorflow.keras import backend as K

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from functools import partial

channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
batch_norm = partial(BatchNormalization, axis=channel_axis)

relu = partial(Activation, 'relu')


def conv1_block(input, name='conv1'):
    x = input

    x = Conv2D(16, (3, 3), padding='same', name=name + '/conv_3x3')(x)
    x = batch_norm(name=name + '/conv_bn')(x)
    x = relu(name=name + '/conv_out')(x)

    return x


def residual_block(input, k=1, stride=1, num_filters=16, dropout=0.0, name='conv'):
    num_filters = num_filters * k
    init = input

    if init.shape[channel_axis] != num_filters:
        init = Conv2D(num_filters, (1, 1), strides=stride, padding='same',
                      name=name + '/conv_identity_1x1')(input)
        init = batch_norm(name=name + '/conv_identity_bn')(init)

    x = Conv2D(num_filters, (3, 3), strides=stride, padding='same',
               name=name + '/conv1_3x3')(input)
    x = batch_norm(name=name + '/conv1_bn')(x)
    x = relu(name=name + '/conv1_out')(x)

    if dropout > 0.0:
        x = Dropout(dropout, name=name + '/dropout')(x)

    x = Conv2D(num_filters, (3, 3), strides=1, padding='same', 
               name=name + '/conv2_3x3')(x)
    x = batch_norm(name=name + '/conv2_bn')(x)

    x = Add(name=name + '/add')([init, x])
    x = relu(name=name + '/out')(x)

    return x
