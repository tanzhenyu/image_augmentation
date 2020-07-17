import tensorflow as tf
from matplotlib import pyplot as plt
import random

from image_augmentation.image import PolicyAugmentation, autoaugment_policy

image_paths = tf.io.gfile.glob("/Volumes/Card/Datasets/flower_photos/*/*.jpg")
random.shuffle(image_paths)

subset_size = 20

image_paths_ss = tf.convert_to_tensor(image_paths[:subset_size])
print(image_paths_ss)

desired_size = 331


@tf.function
def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize_with_crop_or_pad(image, desired_size, desired_size)
    return image


original_images = tf.map_fn(read_image, image_paths_ss, dtype=tf.uint8)
policy = autoaugment_policy()
augmenter = PolicyAugmentation(policy)

augmented_images = augmenter(original_images)


def show_images(images):
    plt.figure(figsize=[subset_size // 3, subset_size // 3])
    for idx, image in enumerate(images):
        plt.subplot(subset_size // 4, 4, idx + 1)
        plt.imshow(image.numpy())
        plt.axis("off")
    plt.tight_layout(0.5, rect=(0, 0, 1, 0.95))


show_images(original_images)
plt.suptitle("Original Images")

show_images(augmented_images)
plt.suptitle("Image Data Augmentation using AutoAugment (reduced ImageNet) Policy")
plt.show()
