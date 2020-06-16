import tensorflow as tf
from matplotlib import pyplot as plt

from image_augmentation.image.image_ops import invert, solarize


def test_invert():
    path = "/Users/swg/Desktop/a.jpg"

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)

    inverted_img = invert(img)

    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(inverted_img)

    plt.show()
    assert True


def test_solarize():
    path = "/Users/swg/Desktop/a.jpg"

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img)

    solarized_img = solarize(img, 130)

    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(solarized_img)

    plt.show()
    assert True
