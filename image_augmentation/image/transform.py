import tensorflow as tf
import tensorflow_addons as tfa

from image_augmentation.image import auto_contrast, invert, equalize, solarize, posterize, contrast
from image_augmentation.image import color, brightness, sharpness, cutout, sample_pairing

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
    "Cutout": cutout,
    "SamplePairing": sample_pairing
}


def some_test_policy():
    policy = [
        [('Cutout', 0.7, 4), ('Invert', 0.3, 10)],
        [('Posterize', 0.6, 10), ('Brightness', 0.3, 2)]
    ]
    return policy


def levels_to_args(translate_max_loc=150, rotate_max_deg=30, cutout_max_size=60):
    shear_min_arg, shear_max_arg = -0.3, 0.3
    translate_min_arg, translate_max_arg = -translate_max_loc, translate_max_loc
    rotate_min_arg, rotate_max_arg = -rotate_max_deg, rotate_max_deg
    solarize_min_arg, solarize_max_arg = 0, 256
    posterize_min_arg, posterize_max_arg = 4, 8

    # contrast, color, brightness, sharpness uses the same range
    blend_min_arg, blend_max_arg = 0.1, 1.9

    cutout_min_arg, cutout_max_arg = 0, cutout_max_size
    sample_pairing_min_arg, sample_pairing_max_arg = 0, 0.4

    gray_color = (128, 128, 128)

    def param(level, min_arg, max_arg):
        return (level * (max_arg - min_arg) / MAX_LEVEL) + min_arg

    def _shear_args(level):
        level = param(level, shear_min_arg, shear_max_arg)
        replace = gray_color
        return level, replace
    shear_x_args = shear_y_args = _shear_args

    def _translate_args(level, is_x):
        level = param(level, translate_min_arg, translate_max_arg)
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
        return angle,

    def _no_args(_):
        return ()
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
    contrast_args = color_args = brightness_args = sharpness_args = _blend_args

    def cutout_args(level):
        size = param(level, cutout_min_arg, cutout_max_arg)
        size = round(size)
        size = size + 1 if size % 2 != 0 else size
        return size,

    def sample_pairing_args(level):
        weight = param(level, sample_pairing_min_arg, sample_pairing_max_arg)
        return weight,

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
        "Cutout": cutout_args,
        "SamplePairing": sample_pairing_args
    }


def apply_subpolicy(image, subpolicy, args):
    def apply_operation(image_, op_name_, level_):
        return TRANSFORMS[op_name_](image_, *args[op_name_](level_))

    for op_name, prob, level in subpolicy:
        random_draw = tf.random.uniform([])
        should_apply_op = tf.floor(random_draw + tf.cast(prob, tf.float32))
        should_apply_op = tf.cast(should_apply_op, tf.bool)

        image = apply_operation(image, op_name, level) if should_apply_op else image
    return image


def randomly_select_subpolicy(policy):
    n_subpolicies = len(policy)
    random_selection = tf.random.uniform([], 0, n_subpolicies, tf.int32)
    return policy[random_selection]


class PolicyAugmentation:
    def __init__(self, policy, translate_max=150, rotate_max_degree=30, cutout_max_size=60):
        self.translate_max = translate_max
        self.rotate_max_degree = rotate_max_degree
        self.cutout_max_size = cutout_max_size
        self.args_level = levels_to_args(translate_max, rotate_max_degree, cutout_max_size)
        self.policy = policy

    def apply(self, images):
        images = tf.convert_to_tensor(images)
        is_image_batch = tf.rank(images) == 4

        def apply_on_image(image):
            subpolicy = randomly_select_subpolicy(self.policy)
            augmented_image = apply_subpolicy(image, subpolicy, self.args_level)
            return augmented_image

        augmented_images = tf.cond(is_image_batch,
                                   lambda: tf.map_fn(apply_on_image, images),
                                   lambda: apply_on_image(images))
        return augmented_images

    def __call__(self, images):
        return self.apply(images)
