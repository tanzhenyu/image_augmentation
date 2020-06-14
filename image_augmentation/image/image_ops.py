import tensorflow as tf

GRAY = 128


@tf.function
def invert(img):
    img = tf.convert_to_tensor(img)
    orig_dtype = img.dtype

    img = tf.cast(img, tf.int16)

    inv_img = (img * -1) + 255
    inv_img = tf.cast(inv_img, orig_dtype)

    return inv_img


@tf.function
def cutout(img, size=16, color=None):
    img = tf.convert_to_tensor(img)

    img_shape = tf.shape(img)
    height, width, channels = img_shape[0], img_shape[1], img_shape[2]

    loc_x = tf.random.uniform((), 0, width, tf.int32)
    loc_y = tf.random.uniform((), 0, height, tf.int32)

    ly, lx = tf.maximum(0, loc_y - size // 2), tf.maximum(0, loc_x - size // 2)
    uy, ux = tf.minimum(height, loc_y + size // 2), tf.minimum(width, loc_x + size // 2)

    if color is None:
        color = tf.repeat(_gray, channels)
    else:
        color = tf.convert_to_tensor(color)
    color = tf.cast(color, img.dtype)

    cut = tf.ones((uy - ly, ux - lx, channels), img.dtype)

    top = img[0: ly, 0: width]
    between = tf.concat([
        img[ly: uy, 0: lx],
        cut * color,
        img[ly: uy, ux: width]
    ], axis=1)
    bottom = img[uy: height, 0: width]

    cutout_img = tf.concat([top, between, bottom], axis=0)
    return cutout_img
