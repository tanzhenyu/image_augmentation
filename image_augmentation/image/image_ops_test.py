import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageEnhance

from image_augmentation.image.image_ops import invert, solarize, cutout, posterize, equalize, auto_contrast
from image_augmentation.image.image_ops import sharpness, color, sample_pairing, brightness, contrast


def _rand_image():
    image = tf.random.uniform([32, 32, 3], 0, 256, dtype=tf.int32)
    image = tf.cast(image, tf.uint8)
    return image


def _display_images(image1, image2):
    plt.subplot(1, 2, 1)
    plt.imshow(image1)

    plt.subplot(1, 2, 2)
    plt.imshow(image2)

    plt.show()


def test_invert():
    image = _rand_image()
    inv_image = invert(image)

    pil_image = Image.fromarray(image.numpy())
    pil_inv_image = np.array(ImageOps.invert(pil_image))

    _display_images(image, inv_image)

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_inv_image = invert(float_image)
    _display_images(float_image, float_inv_image)

    assert tf.reduce_all(inv_image == pil_inv_image)


def test_solarize():
    image = _rand_image()
    threshold = 128
    sol_image = solarize(image, threshold)

    pil_image = Image.fromarray(image.numpy())
    pil_sol_image = np.array(ImageOps.solarize(pil_image, threshold))

    _display_images(image, sol_image)

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_sol_image = solarize(float_image, threshold / 255.)
    _display_images(float_image, float_sol_image)

    assert tf.reduce_all(sol_image == pil_sol_image)


def test_cutout():
    image = _rand_image()
    cut_image = cutout(image)

    _display_images(image, cut_image)

    gray = [128, ] * 3
    gray = tf.cast(gray, cut_image.dtype)

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_cut_image = cutout(float_image)
    _display_images(float_image, float_cut_image)

    float_gray = [128 / 255., ] * 3
    float_gray = tf.cast(float_gray, float_cut_image.dtype)

    # TODO: (warning!) improve this test to include more rigour
    assert tf.reduce_any(cut_image == gray) and tf.reduce_any(float_cut_image == float_gray)


def test_posterize():
    image = _rand_image()
    bits = 2
    post_image = posterize(image, bits)

    _display_images(image, post_image)

    pil_image = Image.fromarray(image.numpy())
    pil_post_image = np.array(ImageOps.posterize(pil_image, bits))

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_post_image = posterize(float_image, bits)
    _display_images(float_image, float_post_image)

    assert tf.reduce_all(post_image == pil_post_image)


def test_equalize():
    image = tf.random.normal([32, 32, 3], 127, 10)
    image = tf.math.round(image)
    image = tf.cast(image, tf.uint8)

    eq_image = equalize(image)

    _display_images(image, eq_image)

    def show_histogram(channel):
        bins = 256
        channel = tf.cast(channel, tf.int32)
        histogram = tf.math.bincount(channel, minlength=bins)

        plt.figure()
        plt.bar(tf.range(256).numpy(), histogram.numpy())

    show_histogram(image[..., 0])
    plt.title("Histogram of Red Channel of Original Image")

    show_histogram(eq_image[..., 0])
    plt.title("Histogram of Red Channel of Equalized Image")

    plt.show()

    pil_image = Image.fromarray(image.numpy())
    pil_eq_image = np.array(ImageOps.equalize(pil_image))

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_eq_image = equalize(float_image)
    _display_images(float_image, float_eq_image)

    assert tf.reduce_all(eq_image == pil_eq_image)


def test_auto_contrast():
    image = _rand_image()
    ac_image = auto_contrast(image)

    pil_image = Image.fromarray(image.numpy())
    pil_ac_image = np.array(ImageOps.autocontrast(pil_image))

    _display_images(image, ac_image)

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_ac_image = auto_contrast(float_image)
    _display_images(float_image, float_ac_image)
    assert tf.reduce_all(ac_image == pil_ac_image)


def test_color():
    image = tf.image.decode_jpeg(tf.io.read_file("../../images/test_example.jpg"))
    factor = 0.5
    colored_image = color(image, factor)

    pil_image = Image.fromarray(image.numpy())
    pil_colored_image = np.array(
        ImageEnhance.Color(pil_image).enhance(factor)
    )

    _display_images(image, colored_image)
    max_deviation = tf.reduce_max(pil_colored_image - colored_image)

    float_image = tf.image.convert_image_dtype(image, tf.float32)
    float_colored_image = color(float_image, factor)
    _display_images(float_image, float_colored_image)

    assert tf.reduce_all(max_deviation < 5)


def test_sharpness():
    image = tf.image.decode_jpeg(tf.io.read_file("/Users/swg/Desktop/a.jpg"))
    factor = 0.5
    sharpened_image = sharpness(image, factor)

    pil_image = Image.fromarray(image.numpy())
    pil_sharpened_image = np.array(
        ImageEnhance.Sharpness(pil_image).enhance(factor)
    )

    _display_images(image, sharpened_image)
    max_deviation = tf.reduce_max(pil_sharpened_image - sharpened_image)
    assert tf.reduce_all(max_deviation < 5)


def test_sample_pairing():
    image1 = tf.image.decode_jpeg(
        tf.io.read_file("/Volumes/Card/Datasets/flower_photos/roses/5060536705_b370a5c543_n.jpg"))
    image2 = tf.image.decode_jpeg(
        tf.io.read_file("/Volumes/Card/Datasets/flower_photos/daisy/2365428551_39f83f10bf_n.jpg"))

    paired_image = sample_pairing(image1, image2, 0.5)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(image1.numpy())
    plt.title("Image 1")

    plt.subplot(1, 2, 2)
    plt.imshow(image2.numpy())
    plt.title("Image 2")

    plt.figure()
    plt.imshow(paired_image.numpy())
    plt.title("Applying SamplePairing on Image 1 and Image 2")

    plt.show()

    float_image1 = tf.image.convert_image_dtype(image1, tf.float32)
    float_image2 = tf.image.convert_image_dtype(image2, tf.float32)
    paired_image = sample_pairing(float_image1, float_image2, 0.5)

    plt.figure()
    plt.imshow(paired_image.numpy())
    plt.title("Applying SamplePairing on Image 1 and Image 2")

    plt.show()


def test_brightness():
    image = _rand_image()
    factor = 0.5
    bright_image = brightness(image, factor)

    pil_image = Image.fromarray(image.numpy())
    pil_bright_image = np.array(
        ImageEnhance.Brightness(pil_image).enhance(factor)
    )

    _display_images(image, bright_image)
    max_deviation = tf.reduce_max(pil_bright_image - bright_image)
    assert tf.reduce_all(max_deviation < 1)


def test_contrast():
    image = tf.image.decode_jpeg((tf.io.read_file("/Users/swg/Desktop/a.jpg")))
    factor = 2.0
    contrast_image = contrast(image, factor)

    pil_image = Image.fromarray(image.numpy())
    pil_contrast_image = np.array(
        ImageEnhance.Contrast(pil_image).enhance(factor)
    )

    _display_images(image, contrast_image)
    _display_images(pil_contrast_image, contrast_image)
    max_deviation = tf.reduce_max(pil_contrast_image - contrast_image)
    assert tf.reduce_all(max_deviation < 1)
