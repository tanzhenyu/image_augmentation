"""Apply data augmentation on images using AutoAugment / RandAugment / custom policies."""

import tensorflow as tf
import tensorflow_addons as tfa

from image_augmentation.image import auto_contrast, invert, equalize, solarize, solarize_add
from image_augmentation.image import posterize, contrast, color, brightness, sharpness, cutout


def convenient_type(tfa_image_fn):
    """Convenience function to cast `replace` argument to match image dtype.
    Required for `tfa.image.translate_xy`, `tfa.image.shear_x`, `tfa.image.shear_y`
    function calls.
    """
    def wrapper(image, level, replace):
        casted_replace = replace / 255 if image.dtype != tf.uint8 else replace
        return tfa_image_fn(image, level, casted_replace)
    return wrapper


TRANSFORMS = {
    "ShearX": convenient_type(tfa.image.shear_x),
    "ShearY": convenient_type(tfa.image.shear_y),
    "TranslateX": convenient_type(tfa.image.translate_xy),
    "TranslateY": convenient_type(tfa.image.translate_xy),
    "Rotate": tfa.image.rotate,
    "AutoContrast": auto_contrast,
    "Invert": invert,
    "Equalize": equalize,
    "Solarize": solarize,
    "SolarizeAdd": solarize_add,
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


def autoaugment_policy(dataset='reduced_imagenet', efficientnet=False):
    """Data augmentation policy as described in "AutoAugment: Learning
    Augmentation Policies from Data" by Cubuk, Zoph et al.
    (https://arxiv.org/abs/1805.09501) found on popular image classification
    datasets.

    Args:
        dataset: a string containing any of the following 'reduced_cifar10',
            'reduced_svhn' or 'reduced_imagenet'.
        efficientnet: a boolean whether to use AutoAugment policies specific
            to official EfficientNet implementation (https://github.com/tensorflow
            /tpu/blob/master/models/official/efficientnet/autoaugment.py) or not.
            Note: AutoAugment policy used to train EfficientNet are different
            from the one discussed (for ImageNet) in original AutoAugment paper.

    Returns:
        An AutoAugment policy which is a nested list of tuples eg.
            [[('op_name', probability, level), ('op_name', probability, level)],
            ...].
    """
    if efficientnet:
        assert dataset == "imagenet"
        efficientnet_policy = [
            [('Equalize', 0.8, 1), ('ShearY', 0.8, 4)],
            [('Color', 0.4, 9), ('Equalize', 0.6, 3)],
            [('Color', 0.4, 1), ('Rotate', 0.6, 8)],
            [('Solarize', 0.8, 3), ('Equalize', 0.4, 7)],
            [('Solarize', 0.4, 2), ('Solarize', 0.6, 2)],
            [('Color', 0.2, 0), ('Equalize', 0.8, 8)],
            [('Equalize', 0.4, 8), ('SolarizeAdd', 0.8, 3)],
            [('ShearX', 0.2, 9), ('Rotate', 0.6, 8)],
            [('Color', 0.6, 1), ('Equalize', 1.0, 2)],
            [('Invert', 0.4, 9), ('Rotate', 0.6, 0)],
            [('Equalize', 1.0, 9), ('ShearY', 0.6, 3)],
            [('Color', 0.4, 7), ('Equalize', 0.6, 0)],
            [('Posterize', 0.4, 6), ('AutoContrast', 0.4, 7)],
            [('Solarize', 0.6, 8), ('Color', 0.6, 9)],
            [('Solarize', 0.2, 4), ('Rotate', 0.8, 9)],
            [('Rotate', 1.0, 7), ('TranslateY', 0.8, 9)],
            [('ShearX', 0.0, 0), ('Solarize', 0.8, 4)],
            [('ShearY', 0.8, 0), ('Color', 0.6, 4)],
            [('Color', 1.0, 0), ('Rotate', 0.6, 2)],
            [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)],
            [('Equalize', 1.0, 4), ('AutoContrast', 0.6, 2)],
            [('ShearY', 0.4, 7), ('SolarizeAdd', 0.6, 7)],
            [('Posterize', 0.8, 2), ('Solarize', 0.6, 10)],
            [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)],
            [('Color', 0.8, 6), ('Rotate', 0.4, 5)],
        ]
        return efficientnet_policy

    else:
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


def levels_to_args(translate_max_loc=150, rotate_max_deg=30, cutout_max_size=60, max_level=10):
    """Converts levels in augmentation policy to specific magnitude
    for applying with image op(s). Note: Some image op(s) do not use
    magnitude values are for them the value of level can be ignored.

    Args:
        translate_max_loc: An int hyperparameter that is used to determine the
            allowed maximum number of pixels for translation. Default is `150`.
        rotate_max_deg: An int hyperparameter in the range `[0, 360]` to determine
            the allowed maximum degree of rotation. Default is `130`.
        cutout_max_size: An int hyperparameter to determine the allowed maximum size
            of square patch for cutout (should be divisible by 2). Default is `60`.
        max_level: An int value that is used to determine range of levels for
            applying the op(s). The resulting magnitudes of operation would be
            in the range `[0, max_level)`. Default is `10`.

    Returns:
        dictionary of op names and an associated convenience function for
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

    gray_color = 128

    def param(level, min_arg, max_arg):
        level = float(level)
        return (level * (max_arg - min_arg) / max_level) + min_arg

    def randomly_negate(arg):
        random_draw = tf.floor(tf.random.uniform([]) + 0.5)
        should_negate = tf.cast(random_draw, tf.bool)
        arg = tf.cond(should_negate, lambda: -arg, lambda: arg)
        return arg

    def _shear_args(level):
        level = param(level, shear_min_arg, shear_max_arg)
        level = randomly_negate(level)
        replace = gray_color
        return float(level), replace
    shear_x_args = shear_y_args = _shear_args

    def _translate_args(level, is_x):
        level = param(level, translate_min_arg, translate_max_arg)
        level = randomly_negate(level)
        replace = gray_color

        # if is_x use for translate x, else for translate y
        if is_x:
            translate_to = [int(level), 0]
        else:
            translate_to = [0, int(level)]
        return translate_to, replace
    translate_x_args = lambda level: _translate_args(level, is_x=True)
    translate_y_args = lambda level: _translate_args(level, is_x=False)

    def rotate_args(level):
        angle = param(level, rotate_min_arg, rotate_max_arg)
        angle = randomly_negate(angle)
        return float(angle),

    def _no_args(_):
        return ()
    # auto_contrast, invert, equalize uses no args
    auto_contrast_args = invert_args = equalize_args = _no_args

    def solarize_args(level):
        threshold = param(level, solarize_min_arg, solarize_max_arg)
        threshold = int(threshold)
        return threshold,

    def posterize_args(level):
        num_bits = param(level, posterize_min_arg, posterize_max_arg)
        num_bits = int(num_bits)
        return num_bits,

    def _blend_args(level):
        magnitude = param(level, blend_min_arg, blend_max_arg)
        return float(magnitude),
    # contrast, color, brightness, sharpness uses the same args
    contrast_args = color_args = brightness_args = sharpness_args = _blend_args

    def cutout_args(level):
        size = param(level, cutout_min_arg, cutout_max_arg)
        size = int(size)
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
        "SolarizeAdd": solarize_args,
        "Posterize": posterize_args,
        "Contrast": contrast_args,
        "Color": color_args,
        "Brightness": brightness_args,
        "Sharpness": sharpness_args,
        "Cutout": cutout_args
    }


def apply_subpolicy(image, parsed_subpolicy, num_ops, args):
    """Applies a specific subpolicy on an image."""
    image_shape = image.shape

    op_names, op_probs, op_levels = parsed_subpolicy
    transform_names = list(TRANSFORMS.keys())

    def get_op_fn_and_args(op_name_, level_, args):
        """Obtains the operation and relevant args given `op_name` and `level`."""
        return TRANSFORMS[op_name_], args[op_name_](level_)

    # iterates each op in the subpolicy and applies on the image (if probable)
    for idx in tf.range(num_ops):
        # set shape of image for `tf.while_loop` to prevent (None,) shapes
        # TODO: check why image op(s) produce None sizes
        tf.autograph.experimental.set_loop_options(shape_invariants=[
            (image, image_shape)
        ])

        op_name, op_prob, op_level = op_names[idx], op_probs[idx], op_levels[idx]

        # randomly draw a number in range (0, 1)
        # and choose whether op should be applied or not using probability
        random_draw = tf.random.uniform([])
        should_apply_op = tf.floor(random_draw + tf.cast(op_prob, tf.float32))
        should_apply_op = tf.cast(should_apply_op, tf.bool)

        # nested for loop to iterate each op
        # helps make graph serializable
        for op_name_ in transform_names:
            # set shape of image for `tf.while_loop` to prevent (None,) shapes
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                (image, image_shape)
            ])

            same_op = tf.equal(op_name, op_name_)
            op_level_ = op_level
            op_fn, op_arg = get_op_fn_and_args(op_name_, op_level_, args)

            image = tf.cond(should_apply_op and same_op,
                            lambda selected_op=op_fn, selected_op_arg=op_arg:
                                selected_op(image, *selected_op_arg),
                            lambda: image)
            image = tf.ensure_shape(image, image_shape)

    return image


def parse_policy(policy):
    """Parses a policy of nested list of tuples and converts them
    into a tuple of nested list of lists. Helps with TF serializability."""
    op_names = [[name for name, _, _ in subpolicy] for subpolicy in policy]
    op_probs = [[prob for _, prob, _ in subpolicy] for subpolicy in policy]
    op_levels = [[level for _, _, level in subpolicy] for subpolicy in policy]

    return op_names, op_probs, op_levels


def randomly_select_subpolicy(parsed_policy, num_subpolicies):
    """Randomly select a single subpolicy from complete policy."""
    op_names, op_probs, op_levels = parsed_policy

    random_selection = tf.random.uniform([], 0, num_subpolicies, tf.int32)

    return (tf.gather(op_names, random_selection),
            tf.gather(op_probs, random_selection),
            tf.gather(op_levels, random_selection))


class PolicyAugmentation:
    """Apply data augmentation on images using a data augmentation policy.
    The data augmentation policy must be either an AutoAugment (https://arxiv.org/abs/1805.09501)
    or a custom policy. AutoAugment policies can be obtained using `autoaugment_policy(dataset)`.

    Returns:
        A 1-arg callable that can be used to augment a batch of images
            or a single image.
    """

    def __init__(self, policy, translate_max=150, rotate_max_degree=30,
                 cutout_max_size=60, max_level=10, seed=None):
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
            max_level: An int value to determine the maximum level for magnitude values that will be
                used for applying an image op. Resulting levels are in the allowed range
                `[0, max_level)`. AutoAugment uses levels in the range `[0, 10)` hence, default is `10`.
            seed: An int value for setting seed to ensure deterministic results.
                Default is `None`.
        """
        self.translate_max = translate_max
        self.rotate_max_degree = rotate_max_degree
        self.cutout_max_size = cutout_max_size
        self.max_level = max_level
        self.args_level = levels_to_args(self.max_level, self.translate_max,
                                         self.rotate_max_degree, self.cutout_max_size)
        self.policy = PolicyAugmentation._fix_policy(policy)

        self.parsed_policy = parse_policy(self.policy)
        self.num_subpolicies = len(self.policy)
        self.num_ops = len(self.policy[0])

        if seed is not None:
            tf.random.set_seed(seed)

    @staticmethod
    def _fix_policy(policy):
        """Fixes a `policy` containing inconsistent number of ops in their subpolicies.
        Adds required number number of dummy ops to each subpolicy with lesser ops than the maximum.

        Args:
            policy: A nested list of tuples of form [[('op_name', probability, level),
                ('op_name', probability, level)], ...].

        Returns:
            A nested list of tuples of form [[('op_name', probability, level),
                ('op_name', probability, level)], ...].
        """
        # calculate number of ops in
        num_ops = [len(subpolicy) for subpolicy in policy]
        max_num_ops = max(num_ops)

        # a dummy subpolicy with probability 0.0
        some_op_name = list(TRANSFORMS.keys())[0]
        dummy_op = (some_op_name, 0.0, 0.0)

        for idx, each_num_ops in enumerate(num_ops):
            diff_num_ops = max_num_ops - each_num_ops
            if diff_num_ops != 0:
                policy[idx] += [dummy_op] * diff_num_ops

        return policy

    def apply_on_image(self, image):
        """Applies augmentation on a single `image`.

        Args:
            image: An int or float tensor of shape `[height, width, num_channels]`.

        Returns:
             A tensor with same shape and type as that of `image`.
        """
        parsed_subpolicy = randomly_select_subpolicy(self.parsed_policy, self.num_subpolicies)
        augmented_image = apply_subpolicy(image, parsed_subpolicy, self.num_ops, self.args_level)
        return augmented_image

    def apply(self, images):
        """Applies augmentation on a batch of `images`.

        Args:
            images: An int or float tensor of shape `[height, width, num_channels]` or
                `[num_images, height, width, num_channels]`.

        Returns:
             A tensor with same shape and type as that of `images`.
        """
        images = tf.convert_to_tensor(images)
        augmented_images = tf.map_fn(self.apply_on_image, images)
        return augmented_images

    def __call__(self, images):
        """Calls self as a function. Alias of `apply` method."""
        return self.apply(images)


def apply_randaugment(image, num_layers, magnitude, args):
    """Applies RandAugment on a single image with given value of `M` and `N`."""
    image_shape = image.shape
    op_names = list(TRANSFORMS.keys())

    def get_op_fn_and_args(op_name_, magnitude_, args):
        """Obtains the operation and relevant args given `op_name` and `magnitude`."""
        return TRANSFORMS[op_name_], args[op_name_](magnitude_)

    # select and apply random op(s) on the image sequentially for `num_layers` number of times
    for _ in tf.range(num_layers):
        # set shape of image for `tf.while_loop` to prevent (None,) shapes
        # TODO: check why image op(s) produce None sizes
        tf.autograph.experimental.set_loop_options(shape_invariants=[
            (image, image_shape)
        ])

        draw_op_idx = tf.random.uniform([], 0, len(op_names), dtype=tf.int32)
        for (idx, op_name) in enumerate(op_names):
            # set shape of image for `tf.while_loop` to prevent (None,) shapes
            tf.autograph.experimental.set_loop_options(shape_invariants=[
                (image, image_shape)
            ])

            op_fn, op_arg = get_op_fn_and_args(op_name, magnitude, args)
            image = tf.cond(idx == draw_op_idx,
                            lambda selected_op=op_fn, selected_op_arg=op_arg:
                                selected_op(image, *selected_op_arg),
                            lambda: image)
            image = tf.ensure_shape(image, image_shape)
    return image


class RandAugment:
    """Apply data augmentation on images using RandAugment.
    This is an implementation of RandAugment as described in "RandAugment: Practical automated data
    augmentation with a reduced search space" by Cubuk, Zoph et al.

    Returns:
        A 1-arg callable that can be used to augment a batch of images
            or a single image.
    """

    def __init__(self, magnitude, num_layers, translate_max=150,
                 rotate_max_degree=30, cutout_max_size=60, seed=None):
        """Applies data augmentation on image(s) using RandAugment strategy.

        Args:
            magnitude: An int hyperparameter `M` (as per paper) in the range [0, 30)
                used to determine the magnitude of operation of each image op.
                Usually best values are in the range `[5, 30]`.
            num_layers: An int hyperparameter `N` (as per paper) used to determine the
                number of randomly selected image op(s) that are to be applied on each image.
                Usually best values are in the range `[1, 3]`.
            translate_max: An int hyperparameter that is used to determine the
                allowed maximum number of pixels for translation. Default is `150`.
            rotate_max_degree: An int hyperparameter in the range [0, 360] to determine
                the allowed maximum degree of rotation. Default is `30`.
            cutout_max_size: An int hyperparameter to determine the allowed maximum
                size of square patch for cutout (should be divisible by 2). Default is `60`.
            seed: An int value for setting seed to ensure deterministic results.
                Default is `None`.
        """
        self.magnitude = magnitude
        self.num_layers = num_layers

        self.translate_max = translate_max
        self.rotate_max_degree = rotate_max_degree
        self.cutout_max_size = cutout_max_size

        max_level = 30  # RandAugment paper suggests max value as 30
        self.args_level = levels_to_args(max_level, self.translate_max,
                                         self.rotate_max_degree, self.cutout_max_size)

        if seed is not None:
            tf.random.set_seed(seed)

    def apply_on_image(self, image):
        """Applies augmentation on a single `image`.

        Args:
            image: An int or float tensor of shape `[height, width, num_channels]`.

        Returns:
             A tensor with same shape and type as that of `image`.
        """
        return apply_randaugment(image, self.num_layers, self.magnitude, self.args_level)

    def apply(self, images):
        """Applies augmentation on a batch of `images`.

        Args:
            images: An int or float tensor of shape
                `[num_images, height, width, num_channels]`.

        Returns:
             A tensor with same shape and type as that of `images`.
        """
        images = tf.convert_to_tensor(images)
        augmented_images = tf.map_fn(self.apply_on_image, images)
        return augmented_images

    def __call__(self, images):
        """Calls self as a function. Alias of `apply` method."""
        return self.apply(images)
