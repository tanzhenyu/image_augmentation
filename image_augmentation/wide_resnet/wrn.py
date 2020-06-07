from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras import Input, Model

from tensorflow.keras import backend as K

from functools import partial

channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
batch_norm = partial(BatchNormalization, axis=channel_axis)

relu = partial(Activation, 'relu')


def conv_block(input, name='conv1'):
    x = input

    x = Conv2D(16, (3, 3), padding='same', name=name + '/conv_3x3')(x)
    x = batch_norm(name=name + '/conv_bn')(x)
    x = relu(name=name + '/conv_out')(x)

    return x


def residual_block(input, num_filters=16, k=1,
                   stride=1, dropout=0.0, name='conv2'):

    num_filters = num_filters * k
    init = input

    if init.shape[channel_axis] != num_filters:
        init = Conv2D(num_filters, (1, 1), strides=stride, padding='same',
                      name=name + '/conv_identity_1x1')(input)

    x = Conv2D(num_filters, (3, 3), strides=stride, padding='same',
               name=name + '/conv1_3x3')(input)
    x = batch_norm(name=name + '/conv1_bn')(x)
    x = relu(name=name + '/conv1_out')(x)

    if dropout > 0.0:
        x = Dropout(dropout, name=name + '/dropout')(x)

    x = Conv2D(num_filters, (3, 3), strides=1, padding='same',
               name=name + '/conv2_3x3')(x)

    x = Add(name=name + '/add')([init, x])
    x = batch_norm(name=name + '/bn')(x)
    x = relu(name=name + '/out')(x)

    return x


def WideResNet(input_shape, depth=28, k=10, dropout=0.0,
               num_classes=10, name=None):
    if name is None:
        name = 'WideResNet' + '-' + str(depth) + '-' + str(k)

    assert (depth - 4) % 6 == 0, "depth must be 6n+4"
    n = (depth - 4) // 6

    filters = [(16 * (2 ** i)) for i in range(3)]

    inp = Input(input_shape, name='input')

    # conv1
    x = conv_block(inp, name='conv1')

    # conv2: n blocks
    for i in range(n):
        x = residual_block(x, num_filters=filters[0], k=k,
                           stride=1, dropout=dropout,
                           name='conv2' + '/block' + str(i + 1))

    # conv3: n blocks
    for i in range(n):
        stride = 2 if i == 0 else 1
        x = residual_block(x, num_filters=filters[1], k=k,
                           stride=stride, dropout=dropout,
                           name='conv3' + '/block' + str(i + 1))

    # conv4: n blocks
    for i in range(n):
        stride = 2 if i == 0 else 1
        x = residual_block(x, num_filters=filters[2], k=k,
                           stride=stride, dropout=dropout,
                           name='conv4' + '/block' + str(i + 1))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='preds')(x)

    net = Model(inp, x, name=name)
    return net
