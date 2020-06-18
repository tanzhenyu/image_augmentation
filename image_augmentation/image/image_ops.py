import tensorflow as tf

GRAY = 128


@tf.function
def invert(img):
    img = tf.convert_to_tensor(img)

    inv_img = 255 - img
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
        color = tf.repeat(GRAY, channels)
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


@tf.function
def solarize(img, threshold):
    img = tf.convert_to_tensor(img)
    threshold = tf.cast(threshold, img.dtype)

    inverted_img = invert(img)
    mask = img < threshold

    solarized_img = tf.where(mask, img, inverted_img)
    return solarized_img


@tf.function
def posterize(img, num_bits):
    img = tf.convert_to_tensor(img)
    img = tf.cast(img, tf.uint8)

    num_bits = tf.cast(num_bits, tf.int32)
    mask = tf.cast(2 ** (8 - num_bits) - 1, tf.uint8)
    mask = tf.bitwise.invert(mask)

    posterized_img = tf.bitwise.bitwise_and(img, mask)
    return posterized_img


@tf.function
def equalize(img):
    img = tf.convert_to_tensor(img)
    orig_dtype = img.dtype
    orig_shape = tf.shape(img)
    img = tf.cast(img, tf.int32)

    bins = 256

    histogram = tf.math.bincount(img, minlength=bins)
    num_pixels = tf.reduce_sum(histogram)
    norm_histogram = tf.cast(histogram, tf.float32) / tf.cast(num_pixels, tf.float32)

    cumulative_histogram = tf.math.cumsum(norm_histogram)
    equalized_histogram = cumulative_histogram * bins
    equalized_histogram = tf.math.round(equalized_histogram)
    equalized_histogram = tf.cast(equalized_histogram, tf.int32)

    flat_img = tf.reshape(img, [tf.reduce_prod(orig_shape)])
    equalized_flat_img = tf.gather(equalized_histogram, flat_img)
    equalized_flat_img = tf.cast(equalized_flat_img, orig_dtype)

    equalized_img = tf.reshape(equalized_flat_img, orig_shape)
    return equalized_img

