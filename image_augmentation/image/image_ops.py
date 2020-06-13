import tensorflow as tf


@tf.function
def invert(img):
    img = tf.convert_to_tensor(img)
    orig_dtype = img.dtype

    img = tf.cast(img, tf.int16)

    inv_img = (img * -1) + 255
    inv_img = tf.cast(inv_img, orig_dtype)

    return inv_img
