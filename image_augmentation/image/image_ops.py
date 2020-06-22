import tensorflow as tf

GRAY = tf.constant(128)

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

    def equalize_grayscale(img_channel):
        current_shape = tf.shape(img_channel)

        bins = tf.constant(256, tf.int32)

        histogram = tf.math.bincount(img_channel, minlength=bins)
        num_pixels = tf.reduce_sum(histogram)
        norm_histogram = tf.cast(histogram, tf.float32) / tf.cast(num_pixels, tf.float32)

        bins = tf.cast(bins, tf.float32)

        cumulative_histogram = tf.math.cumsum(norm_histogram)
        levels = tf.math.round(cumulative_histogram * (bins - 1))
        levels = tf.cast(levels, tf.int32)

        flat_img = tf.reshape(img_channel, [tf.reduce_prod(current_shape)])
        equalized_flat_img = tf.gather(levels, flat_img)
        equalized_flat_img = tf.cast(equalized_flat_img, tf.int32)

        equalized_img_channel = tf.reshape(equalized_flat_img, current_shape)
        return equalized_img_channel

    if orig_shape[-1] == 3:
        red_channel, green_channel, blue_channel = img[..., 0], img[..., 1], img[..., 2]

        red_equalized_img = equalize_grayscale(red_channel)
        green_equalized_img = equalize_grayscale(green_channel)
        blue_equalized_img = equalize_grayscale(blue_channel)

        equalized_img = tf.stack([red_equalized_img, green_equalized_img, blue_equalized_img], axis=-1)

    else:
        equalized_img = equalize_grayscale(img)

    equalized_img = tf.cast(equalized_img, orig_dtype)
    return equalized_img


@tf.function
def auto_contrast(img):
    img = tf.convert_to_tensor(img)
    orig_dtype = img.dtype

    img = tf.cast(img, tf.float32)
    min_val, max_val = tf.reduce_min(img, axis=[0, 1]), tf.reduce_max(img, axis=[0, 1])

    bright = tf.constant(255., tf.float32)

    norm_img = (img - min_val) / (max_val - min_val)
    norm_img = norm_img * bright
    norm_img = tf.cast(norm_img, orig_dtype)
    return norm_img


@tf.function
def blend(img1, img2, factor):
    img1 = tf.convert_to_tensor(img1)
    img2 = tf.convert_to_tensor(img2)

    orig_dtype = img1.dtype

    if factor == 0.0:
        return img1
    elif factor == 1.0:
        return img2

    img1, img2 = tf.cast(img1, tf.float32), tf.cast(img2, tf.float32)
    scaled_diff = (img2 - img1) * factor

    blended_img = img1 + scaled_diff
    blended_img = tf.clip_by_value(blended_img, 0.0, 255.0)
    blended_img = tf.cast(blended_img, orig_dtype)
    return blended_img


@tf.function
def color(img, magnitude):
    img = tf.convert_to_tensor(img)
    orig_dtype = img.dtype
    grayed_img = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(img))

    colored_img = blend(grayed_img, img, magnitude)
    colored_img = tf.cast(colored_img, orig_dtype)
    return colored_img


@tf.function
def sharpen(img, magnitude):
    img = tf.convert_to_tensor(img)
    orig_dtype = img.dtype
    img = tf.cast(img, tf.float32)

    blur_kernel = tf.constant([[1, 1, 1],
                               [1, 5, 1],
                               [1, 1, 1]], tf.float32, shape=[3, 3, 1, 1]) / 13
    blur_kernel = tf.tile(blur_kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]

    # add extra dimension to img before conv
    blurred_img = tf.nn.depthwise_conv2d(img[None, ...], blur_kernel,
                                         strides, padding="VALID")
    blurred_img = tf.clip_by_value(blurred_img, 0., 255.)
    # remove extra dimension
    blurred_img = blurred_img[0]

    mask = tf.ones_like(blurred_img)
    extra_padding = tf.constant([[1, 1],
                                 [1, 1],
                                 [0, 0]], tf.int32)
    padded_mask = tf.pad(mask, extra_padding)
    padded_blurred_img = tf.pad(blurred_img, extra_padding)

    blurred_img = tf.where(padded_mask == 1, padded_blurred_img, img)

    sharpened_img = blend(blurred_img, img, magnitude)
    sharpened_img = tf.cast(sharpened_img, orig_dtype)
    return sharpened_img


@tf.function
def shear(img, x_magnitude, y_magnitude):
    return img


@tf.function
def sample_pairing(img1, img2, weight):
    return blend(img1, img2, weight)
