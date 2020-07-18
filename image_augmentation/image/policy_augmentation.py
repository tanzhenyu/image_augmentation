"""Apply data augmentation on images using AutoAugment / RandAugment / custom policies."""

import tensorflow as tf
import tensorflow_addons as tfa

from image_augmentation.image import auto_contrast, invert, equalize, solarize, posterize
from image_augmentation.image import contrast, color, brightness, sharpness, cutout

MAX_LEVEL = 10

TRANSFORMS = {
    "ShearX": tfa.image.shear_x,
    "ShearY": tfa.image.shear_y,
    "TranslateX": tfa.image.translate_xy,
    "TranslateY": tfa.image.translate_xy,
    "Rotate": tfa.image.rotate,
    "AutoContrast": auto_contrast,
    "Invert": invert,
    "Equalize": equalize,
    "Solarize": solarize,
    "Posterize": posterize,
    "Contrast": contrast,
    "Color": color,
    "Brightness": brightness,
    "Sharpness": sharpness,
    "Cutout": cutout
}


def some_test_policy():
    """Policy with 4 random data augmentation op(s), to
    be used for testing purposes only.

    Returns:
        An augmentation policy which is a nested list of tuples eg.
            [[('op_name', probability, level), ('op_name', probability, level)], ...]
    """
    policy = [
        [('Cutout', 0.7, 4), ('Invert', 0.3, 10)],
        [('Posterize', 0.6, 10), ('Brightness', 0.3, 2)]
    ]
    return policy


def autoaugment_policy(dataset='reduced_imagenet'):
    """Data augmentation policy as described in "AutoAugment: Learning
    Augmentation Policies from Data" by Cubuk, Zoph et al.
    (https://arxiv.org/abs/1805.09501) found on popular image classification
    datasets.

    Args:
        dataset: a string containing any of the following 'reduced_cifar10',
            'reduced_svhn' or 'reduced_imagenet'.

    Returns:
        An AutoAugment policy which is a nested list of tuples eg.
            [[('op_name', probability, level), ('op_name', probability, level)],
            ...].
    """
    policies = {
        "reduced_cifar10": [
            [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
            [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
            [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
            [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
            [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)],
            [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
            [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
            [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
            [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
            [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)],
            [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
            [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
            [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
            [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
            [('Solarize', 0.5, 2), ('Invert', 0.0, 3)],
            [('Equalize', 0.2, 0), ('AutoContrast', 0.6, 0)],
            [('Equalize', 0.2, 8), ('Equalize', 0.6, 4)],
            [('Color', 0.9, 9), ('Equalize', 0.6, 6)],
            [('AutoContrast', 0.8, 4), ('Solarize', 0.2, 8)],
            [('Brightness', 0.1, 3), ('Color', 0.7, 0)],
            [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
            [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
            [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
            [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
            [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)]
        ],
        "reduced_svhn": [
            [('ShearX', 0.9, 4), ('Invert', 0.2, 3)],
            [('ShearY', 0.9, 8), ('Invert', 0.7, 5)],
            [('Equalize', 0.6, 5), ('Solarize', 0.6, 6)],
            [('Invert', 0.9, 3), ('Equalize', 0.6, 3)],
            [('Equalize', 0.6, 1), ('Rotate', 0.9, 3)],
            [('ShearX', 0.9, 4), ('AutoContrast', 0.8, 3)],
            [('ShearY', 0.9, 8), ('Invert', 0.4, 5)],
            [('ShearY', 0.9, 5), ('Solarize', 0.2, 6)],
            [('Invert', 0.9, 6), ('AutoContrast', 0.8, 1)],
            [('Equalize', 0.6, 3), ('Rotate', 0.9, 3)],
            [('ShearX', 0.9, 4), ('Solarize', 0.3, 3)],
            [('ShearY', 0.8, 8), ('Invert', 0.7, 4)],
            [('Equalize', 0.9, 5), ('TranslateY', 0.6, 6)],
            [('Invert', 0.9, 4), ('Equalize', 0.6, 7)],
            [('Contrast', 0.3, 3), ('Rotate', 0.8, 4)],
            [('Invert', 0.8, 5), ('TranslateY', 0.0, 2)],
            [('ShearY', 0.7, 6), ('Solarize', 0.4, 8)],
            [('Invert', 0.6, 4), ('Rotate', 0.8, 4)],
            [('ShearY', 0.3, 7), ('TranslateX', 0.9, 3)],
            [('ShearX', 0.1, 6), ('Invert', 0.6, 5)],
            [('Solarize', 0.7, 2), ('TranslateY', 0.6, 7)],
            [('ShearY', 0.8, 4), ('Invert', 0.8, 8)],
            [('ShearX', 0.7, 9), ('TranslateY', 0.8, 3)],
            [('ShearY', 0.8, 5), ('AutoContrast', 0.7, 3)],
            [('ShearX', 0.7, 2), ('Invert', 0.1, 5)]
        ],
        "reduced_imagenet": [
            [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)],
            [('Posterize', 0.6, 7), ('Posterize', 0.6, 6)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
            [('Equalize', 0.4, 4), ('Rotate', 0.8, 8)],
            [('Solarize', 0.6, 3), ('Equalize', 0.6, 7)],
            [('Posterize', 0.8, 5), ('Equalize', 1.0, 2)],
            [('Rotate', 0.2, 3), ('Solarize', 0.6, 8)],
            [('Equalize', 0.6, 8), ('Posterize', 0.4, 6)],
            [('Rotate', 0.8, 8), ('Color', 0.4, 0)],
            [('Rotate', 0.4, 9), ('Equalize', 0.6, 2)],
            [('Equalize', 0.0, 7), ('Equalize', 0.8, 8)],
            [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
            [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
            [('Rotate', 0.8, 8), ('Color', 1.0, 2)],
            [('Color', 0.8, 8), ('Solarize', 0.8, 7)],
            [('Sharpness', 0.4, 7), ('Invert', 0.6, 8)],
            [('ShearX', 0.6, 5), ('Equalize', 1.0, 9)],
            [('Color', 0.4, 0), ('Equalize', 0.6, 3)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Invert', 0.6, 4), ('Equalize', 1.0, 8)],
            [('Color', 0.6, 4), ('Contrast', 1.0, 8)],
            [('Equalize', 0.8, 8), ('Equalize', 0.6, 3)]
        ]
    }
    assert dataset in policies.keys()
    return policies[dataset]


def levels_to_args(translate_max_loc=150, rotate_max_deg=30, cutout_max_size=60):
    """Converts levels in augmentation policy to specific magnitude
    for applying with image op(s). Note: Some image op(s) do not use
    magnitude values are for them value of level can be ignored.

    Args:
        translate_max_loc: An int hyperparameter that is used to determine the
            allowed maximum number of pixels for translation. Default is `150`.
        rotate_max_deg: An int hyperparameter in the range `[0, 360]` to determine
            the allowed maximum degree of rotation. Default is `130`.
        cutout_max_size: An int hyperparameter to determine the allowed maximum size
            of square patch for cutout (should be divisible by 2). Default is `60`.

    Returns:
        dictionary of op names and a convenience function for
            applying them using subpolicy levels.
    """
    # shear will have range [-0.3, 0.3] by applying random negation
    shear_min_arg, shear_max_arg = 0.0, 0.3
    # translate will have range [-0.3, 0.3] by applying random negation
    translate_min_arg, translate_max_arg = 0.0, translate_max_loc
    # rotate will have range [-0.3, 0.3] by applying random negation
    rotate_min_arg, rotate_max_arg = 0.0, rotate_max_deg
    solarize_min_arg, solarize_max_arg = 0, 256
    posterize_min_arg, posterize_max_arg = 4, 8
    cutout_min_arg, cutout_max_arg = 0, cutout_max_size

    # contrast, color, brightness, sharpness uses the same range
    blend_min_arg, blend_max_arg = 0.1, 1.9

    gray_color = (128, 128, 128)

    def param(level, min_arg, max_arg):
        return (level * (max_arg - min_arg) / MAX_LEVEL) + min_arg

    def randomly_negate(arg):
        random_draw = tf.floor(tf.random.uniform([]) + 0.5)
        should_negate = tf.cast(random_draw, tf.bool)
        arg = tf.cond(should_negate, lambda: -arg, lambda: arg)
        return arg

    def _shear_args(level):
        level = param(level, shear_min_arg, shear_max_arg)
        level = randomly_negate(level)
        replace = gray_color
        return level, replace
    shear_x_args = shear_y_args = _shear_args

    def _translate_args(level, is_x):
        level = param(level, translate_min_arg, translate_max_arg)
        level = randomly_negate(level)
        replace = gray_color

        # if is_x use for translate x, else for translate y
        if is_x:
            translate_to = [round(level), 0]
        else:
            translate_to = [0, round(level)]
        return translate_to, replace
    translate_x_args = lambda level: _translate_args(level, is_x=True)
    translate_y_args = lambda level: _translate_args(level, is_x=False)

    def rotate_args(level):
        angle = param(level, rotate_min_arg, rotate_max_arg)
        angle = randomly_negate(angle)
        return angle,

    def _no_args(_):
        return ()
    # auto_contrast, invert, equalize uses no args
    auto_contrast_args = invert_args = equalize_args = _no_args

    def solarize_args(level):
        threshold = param(level, solarize_min_arg, solarize_max_arg)
        threshold = round(threshold)
        return threshold,

    def posterize_args(level):
        num_bits = param(level, posterize_min_arg, posterize_max_arg)
        num_bits = round(num_bits)
        return num_bits,

    def _blend_args(level):
        magnitude = param(level, blend_min_arg, blend_max_arg)
        return magnitude,
    # contrast, color, brightness, sharpness uses the same args
    contrast_args = color_args = brightness_args = sharpness_args = _blend_args

    def cutout_args(level):
        size = param(level, cutout_min_arg, cutout_max_arg)
        size = round(size)
        size = size + 1 if size % 2 != 0 else size
        return size,

    return {
        "ShearX": shear_x_args,
        "ShearY": shear_y_args,
        "TranslateX": translate_x_args,
        "TranslateY": translate_y_args,
        "Rotate": rotate_args,
        "AutoContrast": auto_contrast_args,
        "Invert": invert_args,
        "Equalize": equalize_args,
        "Solarize": solarize_args,
        "Posterize": posterize_args,
        "Contrast": contrast_args,
        "Color": color_args,
        "Brightness": brightness_args,
        "Sharpness": sharpness_args,
        "Cutout": cutout_args
    }


def apply_subpolicy(image, subpolicy, args):
    """Applies a specific subpolicy on an image."""
    def apply_operation(image_, op_name_, level_):
        return TRANSFORMS[op_name_](image_, *args[op_name_](level_))

    # iterates each op in the subpolicy and applies on the image (if probable)
    for op_name, prob, level in subpolicy:
        random_draw = tf.random.uniform([])
        should_apply_op = tf.floor(random_draw + tf.cast(prob, tf.float32))
        should_apply_op = tf.cast(should_apply_op, tf.bool)

        image = apply_operation(image, op_name, level) if should_apply_op else image
    return image


def randomly_select_subpolicy(policy):
    """Randomly select a single subpolicy from complete policy."""
    n_subpolicies = len(policy)
    random_selection = tf.random.uniform([], 0, n_subpolicies, tf.int32)
    return policy[random_selection]


class PolicyAugmentation:
    """Apply data augmentation on images using a data augmentation policy.
    The data augmentation policy must be either an AutoAugment (https://arxiv.org/abs/1805.09501),
    RandAugment (https://arxiv.org/abs/1909.13719) or a custom policy. AutoAugment policies can
    be obtained using `autoaugment_policy(dataset)`.

    Returns:
        A 1-arg callable that can be used to augment a batch of images
            or a single image.
    """

    def __init__(self, policy, translate_max=150, rotate_max_degree=30, cutout_max_size=60, seed=None):
        """Applies data augmentation on image(s) using a `policy`.

        Args:
            policy: A nested list of tuples of form [[('op_name', probability, level),
                ('op_name', probability, level)], ...]. Note: The maximum level is `10`.
            translate_max: An int hyperparameter that is used to determine the
                allowed maximum number of pixels for translation. Default is `150`.
            rotate_max_degree: An int hyperparameter in the range [0, 360] to determine
                the allowed maximum degree of rotation. Default is `30`.
            cutout_max_size: An int hyperparameter to determine the allowed maximum
                size of square patch for cutout (should be divisible by 2). Default is `60`.
            seed: An int value for setting seed to ensure deterministic results.
                Default is `None`.
        """
        self.translate_max = translate_max
        self.rotate_max_degree = rotate_max_degree
        self.cutout_max_size = cutout_max_size
        self.args_level = levels_to_args(translate_max, rotate_max_degree, cutout_max_size)
        self.policy = policy

        if seed is not None:
            tf.random.set_seed(seed)

    def apply(self, images):
        """Applies augmentation on a batch of `images` or on a single image.

        Args:
            images: An int or float tensor of shape `[height, width, num_channels]` or
                `[num_images, height, width, num_channels]`.

        Returns:
             A tensor with same shape and type as that of `images`.
        """
        images = tf.convert_to_tensor(images)
        is_image_batch = tf.rank(images) == 4

        def apply_on_image(image):
            """Applies data augmentation on a single image."""
            subpolicy = randomly_select_subpolicy(self.policy)
            augmented_image = apply_subpolicy(image, subpolicy, self.args_level)
            return augmented_image

        # if batch, use map_fn and then apply, else apply directly
        augmented_images = tf.cond(is_image_batch,
                                   lambda: tf.map_fn(apply_on_image, images),
                                   lambda: apply_on_image(images))
        return augmented_images

    def __call__(self, images):
        """Calls self as a function. Alias of `apply` method."""
        return self.apply(images)
