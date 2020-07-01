import tensorflow as tf

GRAY = tf.constant(128)


@tf.function
def invert(image):
    image = tf.convert_to_tensor(image)

    inv_image = 255 - image
    return inv_image


@tf.function
def cutout(image, size=16, color=None):
    image = tf.convert_to_tensor(image)

    image_shape = tf.shape(image)
    height, width, channels = image_shape[0], image_shape[1], image_shape[2]

    loc_x = tf.random.uniform((), 0, width, tf.int32)
    loc_y = tf.random.uniform((), 0, height, tf.int32)

    ly, lx = tf.maximum(0, loc_y - size // 2), tf.maximum(0, loc_x - size // 2)
    uy, ux = tf.minimum(height, loc_y + size // 2), tf.minimum(width, loc_x + size // 2)

    if color is None:
        color = tf.repeat(GRAY, channels)
    else:
        color = tf.convert_to_tensor(color)
    color = tf.cast(color, image.dtype)

    cut = tf.ones((uy - ly, ux - lx, channels), image.dtype)

    top = image[0: ly, 0: width]
    between = tf.concat([
        image[ly: uy, 0: lx],
        cut * color,
        image[ly: uy, ux: width]
    ], axis=1)
    bottom = image[uy: height, 0: width]

    cutout_image = tf.concat([top, between, bottom], axis=0)
    return cutout_image


@tf.function
def solarize(image, threshold):
    image = tf.convert_to_tensor(image)
    threshold = tf.cast(threshold, image.dtype)

    inverted_image = invert(image)
    mask = image < threshold

    solarized_image = tf.where(mask, image, inverted_image)
    return solarized_image


@tf.function
def posterize(image, num_bits):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.uint8)

    num_bits = tf.cast(num_bits, tf.int32)
    mask = tf.cast(2 ** (8 - num_bits) - 1, tf.uint8)
    mask = tf.bitwise.invert(mask)

    posterized_image = tf.bitwise.bitwise_and(image, mask)
    return posterized_image


@tf.function
def equalize(image):
    image = tf.convert_to_tensor(image)
    orig_dtype = image.dtype
    orig_shape = tf.shape(image)

    image = tf.cast(image, tf.int32)

    def equalize_grayscale(image_channel):
        current_shape = tf.shape(image_channel)

        bins = tf.constant(256, tf.int32)

        histogram = tf.math.bincount(image_channel, minlength=bins)
        num_pixels = tf.reduce_sum(histogram)
        norm_histogram = tf.cast(histogram, tf.float32) / tf.cast(num_pixels, tf.float32)

        bins = tf.cast(bins, tf.float32)

        cumulative_histogram = tf.math.cumsum(norm_histogram)
        levels = tf.math.round(cumulative_histogram * (bins - 1))
        levels = tf.cast(levels, tf.int32)

        flat_image = tf.reshape(image_channel, [tf.reduce_prod(current_shape)])
        equalized_flat_image = tf.gather(levels, flat_image)
        equalized_flat_image = tf.cast(equalized_flat_image, tf.int32)

        equalized_image_channel = tf.reshape(equalized_flat_image, current_shape)
        return equalized_image_channel

    if orig_shape[-1] == 3:
        red_channel, green_channel, blue_channel = image[..., 0], image[..., 1], image[..., 2]

        red_equalized_image = equalize_grayscale(red_channel)
        green_equalized_image = equalize_grayscale(green_channel)
        blue_equalized_image = equalize_grayscale(blue_channel)

        equalized_image = tf.stack([red_equalized_image, green_equalized_image, blue_equalized_image], axis=-1)

    else:
        equalized_image = equalize_grayscale(image)

    equalized_image = tf.cast(equalized_image, orig_dtype)
    return equalized_image


@tf.function
def auto_contrast(image):
    image = tf.convert_to_tensor(image)
    orig_dtype = image.dtype

    image = tf.cast(image, tf.float32)
    min_val, max_val = tf.reduce_min(image, axis=[0, 1]), tf.reduce_max(image, axis=[0, 1])

    bright = tf.constant(255., tf.float32)

    norm_image = (image - min_val) / (max_val - min_val)
    norm_image = norm_image * bright
    norm_image = tf.cast(norm_image, orig_dtype)
    return norm_image


@tf.function
def blend(image1, image2, factor):
    image1 = tf.convert_to_tensor(image1)
    image2 = tf.convert_to_tensor(image2)

    orig_dtype = image2.dtype

    if factor == 0.0:
        return image1
    elif factor == 1.0:
        return image2

    image1, image2 = tf.cast(image1, tf.float32), tf.cast(image2, tf.float32)
    scaled_diff = (image2 - image1) * factor

    blended_image = image1 + scaled_diff
    blended_image = tf.clip_by_value(blended_image, 0.0, 255.0)
    blended_image = tf.cast(blended_image, orig_dtype)
    return blended_image


@tf.function
def color(image, magnitude):
    image = tf.convert_to_tensor(image)
    orig_dtype = image.dtype
    tiled_gray_image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))

    colored_image = blend(tiled_gray_image, image, magnitude)
    colored_image = tf.cast(colored_image, orig_dtype)
    return colored_image


@tf.function
def sharpness(image, magnitude):
    image = tf.convert_to_tensor(image)
    orig_dtype = image.dtype
    image = tf.cast(image, tf.float32)

    blur_kernel = tf.constant([[1, 1, 1],
                               [1, 5, 1],
                               [1, 1, 1]], tf.float32, shape=[3, 3, 1, 1]) / 13
    blur_kernel = tf.tile(blur_kernel, [1, 1, 3, 1])
    strides = [1, 1, 1, 1]

    # add extra dimension to image before conv
    blurred_image = tf.nn.depthwise_conv2d(image[None, ...], blur_kernel,
                                           strides, padding="VALID")
    blurred_image = tf.clip_by_value(blurred_image, 0., 255.)
    # remove extra dimension
    blurred_image = blurred_image[0]

    mask = tf.ones_like(blurred_image)
    extra_padding = tf.constant([[1, 1],
                                 [1, 1],
                                 [0, 0]], tf.int32)
    padded_mask = tf.pad(mask, extra_padding)
    padded_blurred_image = tf.pad(blurred_image, extra_padding)

    blurred_image = tf.where(padded_mask == 1, padded_blurred_image, image)

    sharpened_image = blend(blurred_image, image, magnitude)
    sharpened_image = tf.cast(sharpened_image, orig_dtype)
    return sharpened_image


@tf.function
def sample_pairing(image1, image2, weight):
    paired_image = blend(image1, image2, weight)
    paired_image = tf.cast(paired_image, image1.dtype)
    return paired_image


@tf.function
def brightness(image, magnitude):
    image = tf.convert_to_tensor(image)
    dark = tf.zeros_like(image)

    bright_image = blend(dark, image, magnitude)
    return bright_image


@tf.function
def contrast(image, magnitude):
    image = tf.convert_to_tensor(image)
    orig_dtype = image.dtype

    grayed_image = tf.image.rgb_to_grayscale(image)
    grayed_image = tf.cast(grayed_image, tf.int32)

    bins = tf.constant(256, tf.int32)
    histogram = tf.math.bincount(grayed_image, minlength=bins)
    histogram = tf.cast(histogram, tf.float32)
    mean = tf.reduce_sum(tf.cast(grayed_image, tf.float32)) / tf.reduce_sum(histogram)
    mean = tf.clip_by_value(mean, 0.0, 255.0)

    mean_image = tf.ones_like(grayed_image, tf.float32) * mean
    mean_image = tf.cast(mean_image, tf.uint8)
    mean_image = tf.image.grayscale_to_rgb(mean_image)

    contrast_image = blend(mean_image, image, magnitude)
    contrast_image = tf.cast(contrast_image, orig_dtype)
    return contrast_image
