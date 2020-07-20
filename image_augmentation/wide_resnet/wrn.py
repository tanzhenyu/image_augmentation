"""WideResNet architecture."""

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, BatchNormalization, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

from tensorflow.keras import backend as K

from functools import partial

CHANNEL_AXIS = 1 if K.image_data_format() == 'channels_first' else -1

_batch_norm = partial(BatchNormalization, axis=CHANNEL_AXIS)
_relu = partial(Activation, 'relu')


def _residual_block(input, num_filters=16, k=1, stride=1,
                    l2_reg=0.0, dropout=0.0, name='res_block'):
    """Pre-activated residual block."""
    num_filters = num_filters * k
    init = branch = input

    init = _batch_norm(name=name + '/bn1')(init)
    init = _relu(name=name + '/relu1')(init)
    if init.shape[CHANNEL_AXIS] != num_filters or name.endswith("block1"):
        branch = Conv2D(num_filters, (1, 1), strides=stride, padding='same',
                        use_bias=False, kernel_initializer=he_normal(),
                        kernel_regularizer=L2(l2_reg),
                        name=name + '/conv_identity_1x1')(init)

    x = Conv2D(num_filters, (3, 3), strides=stride, padding='same',
               use_bias=False, kernel_initializer=he_normal(),
               kernel_regularizer=L2(l2_reg),
               name=name + '/conv1_3x3')(init)

    if dropout > 0.0:
        x = Dropout(dropout, name=name + '/dropout')(x)

    x = _batch_norm(name=name + '/bn2')(x)
    x = _relu(name=name + '/relu2')(x)
    x = Conv2D(num_filters, (3, 3), strides=1, padding='same',
               use_bias=False, kernel_initializer=he_normal(),
               kernel_regularizer=L2(l2_reg),
               name=name + '/conv2_3x3')(x)

    x = Add(name=name + '/add')([branch, x])

    return x


def WideResNet(input_shape, depth=28, k=10, dropout=0.0,
               l2_reg=0.0, num_classes=10, name=None):
    """This is an implementation of WideResNet architecture
    as described in "Wide Residual Networks" by Zagoruyko, Komodakis
    (https://arxiv.org/abs/1605.07146).
    """
    if name is None:
        name = 'WideResNet' + '-' + str(depth) + '-' + str(k)

    assert (depth - 4) % 6 == 0, "depth must be 6n + 4"
    n = (depth - 4) // 6

    filters = [(16 * (2 ** i)) for i in range(3)]

    inp = Input(input_shape, name='input')

    # conv1
    x = Conv2D(16, (3, 3), padding='same', use_bias=False,
               kernel_initializer=he_normal(), kernel_regularizer=L2(l2_reg),
               name='conv1/conv_3x3')(inp)

    # conv2: n blocks
    for i in range(n):
        x = _residual_block(x, num_filters=filters[0], k=k,
                            stride=1, l2_reg=l2_reg, dropout=dropout,
                            name='conv2' + '/block' + str(i + 1))

    # conv3: n blocks
    for i in range(n):
        stride = 2 if i == 0 else 1
        x = _residual_block(x, num_filters=filters[1], k=k,
                            stride=stride, l2_reg=l2_reg, dropout=dropout,
                            name='conv3' + '/block' + str(i + 1))

    # conv4: n blocks
    for i in range(n):
        stride = 2 if i == 0 else 1
        x = _residual_block(x, num_filters=filters[2], k=k,
                            stride=stride, l2_reg=l2_reg, dropout=dropout,
                            name='conv4' + '/block' + str(i + 1))

    x = _batch_norm(name='bn')(x)
    x = _relu(name='relu')(x)

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax',
              kernel_regularizer=L2(l2_reg),
              kernel_initializer=he_normal(), name='preds')(x)

    net = Model(inp, x, name=name)
    return net
