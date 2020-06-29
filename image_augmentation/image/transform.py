import tensorflow as tf
import tensorflow_addons as tfa

from image_augmentation.image import auto_contrast, invert, equalize, solarize, posterize, contrast
from image_augmentation.image import color, brightness, sharpness, cutout, sample_pairing

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


def levels_to_args(translate_max_loc, rotate_max_deg, cutout_max_size):
    shear_arg_min, shear_arg_max = -0.3, 0.3
    translate_arg_min, translate_arg_max = -translate_max_loc, translate_max_loc
    rotate_arg_min, rotate_arg_max = -rotate_max_deg, rotate_max_deg
    solarize_arg_min, solarize_arg_max = 0, 256
    posterize_arg_min, posterize_arg_max = 4, 8
    contrast_arg_min, contrast_arg_max = 0.1, 1.9
    color_arg_min, color_arg_max = 0.1, 1.9
    brightness_arg_min, brightness_arg_max = 0.1, 1.9
    sharpness_arg_min, sharpness_arg_max = 0.1, 1.9
    cutout_arg_min, cutout_arg_max = 0, cutout_max_size
    sample_pairing_arg_min, sample_pairing_arg_max = 0, 0.4

