import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

from image_augmentation.image.image_ops import invert, solarize, cutout, posterize, equalize


def _rand_image():
    img = tf.random.uniform([32, 32, 3], 0, 256, dtype=tf.int32)
    img = tf.cast(img, tf.uint8)
    return img


def _display_images(img1, img2):
    plt.subplot(1, 2, 1)
    plt.imshow(img1)

    plt.subplot(1, 2, 2)
    plt.imshow(img2)

    plt.show()


def test_invert():
    img = _rand_image()
    inv_img = invert(img)

    pil_img = Image.fromarray(img.numpy())
    pil_inv_img = np.array(ImageOps.invert(pil_img))

    _display_images(img, inv_img)
    assert tf.reduce_all(inv_img == pil_inv_img)


def test_solarize():
    img = _rand_image()
    threshold = 128
    sol_img = solarize(img, threshold)

    pil_img = Image.fromarray(img.numpy())
    pil_sol_img = np.array(ImageOps.solarize(pil_img, threshold))

    _display_images(img, sol_img)
    assert tf.reduce_all(sol_img == pil_sol_img)


def test_cutout():
    img = _rand_image()
    cut_img = cutout(img)

    _display_images(img, cut_img)

    gray = [128, ] * 3
    gray = tf.cast(gray, cut_img.dtype)

    # TODO: (warning!) improve this test to include more rigour
    assert tf.reduce_any(cut_img == gray)


def test_posterize():
    img = _rand_image()
    bits = 2
    post_img = posterize(img, bits)

    _display_images(img, post_img)

    pil_img = Image.fromarray(img.numpy())
    pil_post_img = np.array(ImageOps.posterize(pil_img, bits))

    assert tf.reduce_all(post_img == pil_post_img)


def test_equalize():
    img = tf.random.uniform([32, 32, 1], 0, 256, dtype=tf.int32)
    img = tf.cast(img, tf.uint8)

    eq_img = equalize(img)

    _display_images(img[:, :, 0], eq_img[:, :, 0])

    pil_img = Image.fromarray(img.numpy()[:, :, 0])
    pil_eq_img = np.array(ImageOps.equalize(pil_img))

    assert tf.reduce_all(eq_img[:, :, 0] == pil_eq_img)
