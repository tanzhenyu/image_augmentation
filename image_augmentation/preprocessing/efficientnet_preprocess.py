"""Data preprocessing for EfficientNet training and evaluation on ImageNet."""

import tensorflow as tf

IMAGE_SIZE = 224
CROP_PADDING = 32
RESIZE_METHOD = 'bicubic'


@tf.function
def _resize(image, image_size):
    """Resizes an image using the Bi-cubic method."""
    target_size = tf.stack([image_size, image_size])
    resized_image = tf.image.resize(image, target_size,
                                    method=RESIZE_METHOD)
    return tf.cast(resized_image, image.dtype)


@tf.function
def center_crop_and_resize(image, image_size=None):
    """Center crop (with padding) an image and resize it."""
    if image_size is None:
        image_size = IMAGE_SIZE

    image_shape = tf.shape(image)
    image_height, image_width = image_shape[0], image_shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)
    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2

    cropped_image = tf.image.crop_to_bounding_box(image, offset_height,
                                                  offset_width, padded_center_crop_size,
                                                  padded_center_crop_size)
    resized_image = _resize(cropped_image, image_size)
    return resized_image


@tf.function
def random_crop_and_resize(image, image_size=None):
    """Randomly crops an image and resizes it."""
    if image_size is None:
        image_size = IMAGE_SIZE

    original_shape = tf.shape(image)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        original_shape, bbox, min_object_covered=0.1, aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0), max_attempts=10, use_image_if_no_bounding_boxes=True)

    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    cropped_image = tf.image.crop_to_bounding_box(image, offset_y,
                                                  offset_x, target_height,
                                                  target_width)

    cropped_shape = tf.shape(cropped_image)
    match = tf.equal(original_shape, cropped_shape)
    match = tf.cast(match, tf.int32)
    is_bad_crop = tf.math.greater_equal(tf.reduce_sum(match), 3)

    resized_image = tf.cond(is_bad_crop,
                            lambda: center_crop_and_resize(image, image_size),
                            lambda: _resize(image, image_size))
    return resized_image


def preprocess_fn_builder(image_size, num_classes, is_training):
    if is_training:
        def image_preprocess_fn(image, image_size_):
            image = random_crop_and_resize(image, image_size_)
            image = tf.image.flip_left_right(image)
            return image
    else:
        image_preprocess_fn = center_crop_and_resize

    def preprocess_fn(image, label):
        image = image_preprocess_fn(image, image_size)
        label = tf.one_hot(label, num_classes)
        return image, label
    return preprocess_fn
