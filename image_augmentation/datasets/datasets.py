"""Convenience functions for loading curated dataset(s) as pipeline(s)."""

import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 14

REDUCED_CIFAR_10_TRAIN_SIZE = 4000
REDUCED_CIFAR_10_VAL_SIZE = 1715

REDUCED_SVHN_TRAIN_SIZE = 1000
REDUCED_SVHN_VAL_SIZE = 430

REDUCED_IMAGENET_NUM_CLASSES = 120
REDUCED_IMAGENET_TRAIN_SIZE = 6000
REDUCED_IMAGENET_VAL_SIZE = 1200


def cifar10(data_dir=None):
    ds, info = tfds.load('cifar10', as_supervised=True, with_info=True, data_dir=data_dir)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }


def svhn(data_dir=None):
    ds, info = tfds.load('svhn_cropped', as_supervised=True, with_info=True, data_dir=data_dir)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }


def reduced_cifar10(data_dir=None):
    ds = tfds.load('cifar10', as_supervised=True, data_dir=data_dir)
    n_train = tf.data.experimental.cardinality(ds['train'])

    ds = ds['train'].shuffle(n_train, seed=SEED)

    train_ds = ds.take(REDUCED_CIFAR_10_TRAIN_SIZE)
    val_ds = ds.skip(REDUCED_CIFAR_10_TRAIN_SIZE).take(REDUCED_CIFAR_10_VAL_SIZE)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds
    }


def imagenet(data_dir=None):
    ds, info = tfds.load('imagenet_resized/32x32', shuffle_files=True,
                         as_supervised=True, with_info=True, data_dir=data_dir)
    return {
        "train_ds": ds['train'],
        "val_ds": ds['validation'],
        "info": info
    }


def reduced_svhn(data_dir=None):
    ds = tfds.load('svhn_cropped', as_supervised=True, data_dir=data_dir)
    n_train = tf.data.experimental.cardinality(ds['train'])

    ds = ds['train'].shuffle(n_train, seed=SEED)

    train_ds = ds.take(REDUCED_SVHN_TRAIN_SIZE)
    val_ds = ds.skip(REDUCED_SVHN_TRAIN_SIZE).take(REDUCED_SVHN_VAL_SIZE)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds
    }


def reduced_imagenet(data_dir=None):
    ds, info = tfds.load("imagenet_resized/32x32", shuffle_files=True,
                         as_supervised=True, with_info=True, data_dir=data_dir)

    actual_num_classes = info.features['label'].num_classes

    shuffled_classes = tf.random.shuffle(
        tf.range(actual_num_classes, dtype=tf.int64), seed=SEED)
    sampled_classes = shuffled_classes[:REDUCED_IMAGENET_NUM_CLASSES]

    # drop samples as required, samples are chosen from selected 120 classes
    @tf.function
    def filter_fn(image, label):
        return tf.reduce_any(label == sampled_classes)

    # convert labels from original range [0, 1000) to new range [0, 120)
    @tf.function
    def map_fn(image, label):
        new_label = tf.cast(label == sampled_classes, tf.int32)
        new_label = tf.argmax(new_label)
        return image, new_label

    ds = ds['train']
    ds = ds.filter(filter_fn)
    ds = ds.map(map_fn, tf.data.experimental.AUTOTUNE)

    train_ds = ds.take(REDUCED_IMAGENET_TRAIN_SIZE)
    val_ds = ds.skip(REDUCED_IMAGENET_TRAIN_SIZE).take(REDUCED_IMAGENET_VAL_SIZE)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds
    }
