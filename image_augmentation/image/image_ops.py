"""Image processing op(s) for data augmentation."""

import tensorflow as tf

IMAGE_DTYPES = [tf.uint8, tf.float32, tf.float16, tf.float64]


def _check_image_dtype(image):
    assert image.dtype in IMAGE_DTYPES, "image with " + str(image.dtype) + " is not supported for this operation"


@tf.function
def invert(image, name=None):
    """Inverts the pixels of an `image`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "invert"):
        if image.dtype == tf.uint8:
            inv_image = 255 - image
        else:
            inv_image = 1. - image
        return inv_image


@tf.function
def cutout(image, size=16, color=None, name=None):
    """This is an implementation of Cutout as described in "Improved
    Regularization of Convolutional Neural Networks with Cutout" by
    DeVries & Taylor (https://arxiv.org/abs/1708.04552).
    It applies a random square patch of specified `size` over an `image`
    and by replacing those pixels with value of `color`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        size: A 0-D int tensor or single int value that is divisible by 2.
        color: A single pixel value (grayscale) or tuple of 3 values (RGB),
            in case a single value is used for RGB image the value is tiled.
            Gray color (128) is used by default.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "cutout"):
        image_shape = tf.shape(image)
        height, width, channels = image_shape[0], image_shape[1], image_shape[2]

        loc_x = tf.random.uniform((), 0, width, tf.int32)
        loc_y = tf.random.uniform((), 0, height, tf.int32)

        ly, lx = tf.maximum(0, loc_y - size // 2), tf.maximum(0, loc_x - size // 2)
        uy, ux = tf.minimum(height, loc_y + size // 2), tf.minimum(width, loc_x + size // 2)

        gray = tf.constant(128)
        if color is None:
            if image.dtype == tf.uint8:
                color = tf.repeat(gray, channels)
            else:
                color = tf.repeat(tf.cast(gray, tf.float32) / 255., channels)
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
def solarize(image, threshold, name=None):
    """Inverts the pixels of an `image` above a certain `threshold`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        threshold: A 0-D int / float tensor or int / float value for setting
            inversion threshold.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "solarize"):
        threshold = tf.cast(threshold, image.dtype)

        inverted_image = invert(image)
        solarized_image = tf.where(image < threshold, image, inverted_image)
        return solarized_image


@tf.function
def solarize_add(image, addition, threshold=None, name=None):
    """Adds `addition` intensity to each pixel and inverts the pixels
    of an `image` above a certain `threshold`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        addition: A 0-D int / float tensor or int / float value that is to be
            added to each pixel.
        threshold: A 0-D int / float tensor or int / float value for setting
            inversion threshold. 128 (int) / 0.5 (float) is used by default.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "solarize_add"):
        if threshold is None:
            threshold = tf.image.convert_image_dtype(tf.constant(128, tf.uint8), image.dtype)

        addition = tf.cast(addition, image.dtype)
        added_image = image + addition

        dark, bright = tf.constant(0, tf.uint8), tf.constant(255, tf.uint8)
        added_image = tf.clip_by_value(added_image, tf.image.convert_image_dtype(dark, image.dtype),
                                       tf.image.convert_image_dtype(bright, image.dtype))
        return solarize(added_image, threshold)


@tf.function
def posterize(image, num_bits, name=None):
    """Reduces the number of bits used to represent an `image`
    for each color channel.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        num_bits: A 0-D int tensor or integer value representing number of bits.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "posterize"):
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.uint8)

        num_bits = tf.cast(num_bits, tf.int32)
        mask = tf.cast(2 ** (8 - num_bits) - 1, tf.uint8)
        mask = tf.bitwise.invert(mask)

        posterized_image = tf.bitwise.bitwise_and(image, mask)
        posterized_image = tf.image.convert_image_dtype(posterized_image, orig_dtype, saturate=True)
        return posterized_image


@tf.function
def equalize(image, name=None):
    """Equalizes the `image` histogram. In case of an RGB image, equalization
    individually for each channel.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "equalize"):
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
        image = tf.cast(image, tf.int32)

        def equalize_grayscale(image_channel):
            """Equalizes the histogram of a grayscale (2D) image."""
            bins = tf.constant(256, tf.int32)

            histogram = tf.math.bincount(image_channel, minlength=bins)
            nonzero = tf.where(tf.math.not_equal(histogram, 0))
            nonzero_histogram = tf.reshape(tf.gather(histogram, nonzero), [-1])
            step = (tf.reduce_sum(nonzero_histogram) - nonzero_histogram[-1]) // (bins - 1)

            # use a lut similar to PIL
            def normalize(histogram, step):
                norm_histogram = (tf.math.cumsum(histogram) + (step // 2)) // step
                norm_histogram = tf.concat([[0], norm_histogram], axis=0)
                norm_histogram = tf.clip_by_value(norm_histogram, 0, bins - 1)
                return norm_histogram

            return tf.cond(tf.math.equal(step, 0),
                           lambda: image_channel,
                           lambda: tf.gather(normalize(histogram, step), image_channel))

        channels_first_image = tf.transpose(image, [2, 0, 1])
        channels_first_equalized_image = tf.map_fn(equalize_grayscale, channels_first_image)
        equalized_image = tf.transpose(channels_first_equalized_image, [1, 2, 0])

        equalized_image = tf.cast(equalized_image, tf.uint8)
        equalized_image = tf.image.convert_image_dtype(equalized_image, orig_dtype)
        return equalized_image


@tf.function
def auto_contrast(image, name=None):
    """Normalizes `image` contrast by remapping the `image` histogram such
    that the brightest pixel becomes 1.0 (float) / 255 (unsigned int) and
    darkest pixel becomes 0.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "auto_contrast"):
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)

        min_val, max_val = tf.reduce_min(image, axis=[0, 1]), tf.reduce_max(image, axis=[0, 1])

        norm_image = (image - min_val) / (max_val - min_val)
        norm_image = tf.image.convert_image_dtype(norm_image, orig_dtype, saturate=True)
        return norm_image


@tf.function
def blend(image1, image2, factor, name=None):
    """Blends an image with another using `factor`.

    Args:
        image1: An int or float tensor of shape `[height, width, num_channels]`.
        image2: An int or float tensor of shape `[height, width, num_channels]`.
        factor: A 0-D float tensor or single floating point value depicting
            a weight above 0.0 for combining the example_images.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image1` and `image2`.
    """
    _check_image_dtype(image1)
    _check_image_dtype(image2)
    assert image1.dtype == image2.dtype, "image1 type should exactly match type of image2"

    if factor == 0.0:
        return image1
    elif factor == 1.0:
        return image2
    else:
        with tf.name_scope(name or "blend"):
            orig_dtype = image2.dtype

            image1, image2 = tf.image.convert_image_dtype(image1, tf.float32), tf.image.convert_image_dtype(image2, tf.float32)
            scaled_diff = (image2 - image1) * factor

            blended_image = image1 + scaled_diff

            blended_image = tf.image.convert_image_dtype(blended_image, orig_dtype, saturate=True)
            return blended_image


@tf.function
def sample_pairing(image1, image2, weight, name=None):
    """Alias of `blend`. This is an implementation of SamplePairing
    as described in "Data Augmentation by Pairing Samples for Images Classification"
    by Inoue (https://arxiv.org/abs/1801.02929).

    Args:
        image1: An int or float tensor of shape `[height, width, num_channels]`.
        image2: An int or float tensor of shape `[height, width, num_channels]`.
        weight: A 0-D float tensor or single floating point value depicting
            a weight factor above 0.0 for combining the example_images.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image1`.
    """
    with tf.name_scope(name or "sample_pairing"):
        paired_image = blend(image1, image2, weight)
        return paired_image


@tf.function
def color(image, magnitude, name=None):
    """Adjusts the `magnitude` of color of an `image`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        magnitude: A 0-D float tensor or single floating point value above 0.0.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "color"):
        tiled_gray_image = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        colored_image = blend(tiled_gray_image, image, magnitude)
        return colored_image


@tf.function
def sharpness(image, magnitude, name=None):
    """Adjusts the `magnitude` of sharpness of an `image`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        magnitude: A 0-D float tensor or single floating point value above 0.0.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "sharpness"):
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)
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

        sharpened_image = tf.cast(sharpened_image, tf.uint8)
        sharpened_image = tf.image.convert_image_dtype(sharpened_image, orig_dtype)
        return sharpened_image


@tf.function
def brightness(image, magnitude, name=None):
    """Adjusts the `magnitude` of brightness of an `image`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        magnitude: A 0-D float tensor or single floating point value above 0.0.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "brightness"):
        dark = tf.zeros_like(image)
        bright_image = blend(dark, image, magnitude)
        return bright_image


@tf.function
def contrast(image, magnitude, name=None):
    """Adjusts the `magnitude` of contrast of an `image`.

    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        magnitude: A 0-D float tensor or single floating point value above 0.0.
        name: An optional string for name of the operation.

    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "contrast"):
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

        grayed_image = tf.image.rgb_to_grayscale(image)
        grayed_image = tf.cast(grayed_image, tf.int32)
        bins = tf.constant(256, tf.int32)
        histogram = tf.math.bincount(grayed_image, minlength=bins)
        histogram = tf.cast(histogram, tf.float32)
        mean = tf.reduce_sum(tf.cast(grayed_image, tf.float32)) / tf.reduce_sum(histogram)
        mean = tf.clip_by_value(mean, 0.0, 255.0)

        mean = tf.cast(mean, tf.uint8)
        mean_image = tf.ones_like(grayed_image, tf.uint8) * mean
        mean_image = tf.image.grayscale_to_rgb(mean_image)

        contrast_image = blend(mean_image, image, magnitude)
        contrast_image = tf.image.convert_image_dtype(contrast_image, orig_dtype, saturate=True)
        return contrast_image
