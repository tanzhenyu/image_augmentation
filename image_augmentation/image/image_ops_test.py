import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

from image_augmentation.image.image_ops import invert, solarize


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
    threshold = 128

    img = _rand_image()
    sol_img = solarize(img, threshold)

    pil_img = Image.fromarray(img.numpy())
    pil_sol_img = np.array(ImageOps.solarize(pil_img, threshold))

    _display_images(img, sol_img)
    assert tf.reduce_all(sol_img == pil_sol_img)
