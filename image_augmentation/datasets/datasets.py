"""Convenience functions for loading curated dataset(s) as pipeline(s) using TFDS."""

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
    """CIFAR-10 dataset as a `tf.data.Dataset` pipeline.

    Dataset homepage: https://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        data_dir: directory read/write data from TFDS
    """
    ds, info = tfds.load('cifar10', read_config=tfds.ReadConfig(skip_prefetch=True),
                         as_supervised=True, with_info=True, data_dir=data_dir)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }


def svhn(data_dir=None):
    """Street View Housing Numbers (SVHN) dataset as a
    `tf.data.Dataset` pipeline.

    Dataset homepage: http://ufldl.stanford.edu/housenumbers/

    Args:
        data_dir: directory read/write data from TFDS
    """
    ds, info = tfds.load('svhn_cropped', read_config=tfds.ReadConfig(skip_prefetch=True),
                         as_supervised=True, with_info=True, data_dir=data_dir)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }


def reduced_cifar10(data_dir=None):
    """Reduced CIFAR-10 dataset as a `tf.data.Dataset` pipeline.
    The dataset is inspired from work in AutoAugment paper and
    essentially consists of 4000 randomly selected samples from
    original CIFAR-10 dataset.

    Args:
        data_dir: directory read/write data from TFDS
    """
    ds = tfds.load('cifar10', read_config=tfds.ReadConfig(skip_prefetch=True),
                   as_supervised=True, data_dir=data_dir)
    n_train = tf.data.experimental.cardinality(ds['train'])

    ds = ds['train'].shuffle(n_train, seed=SEED)

    train_ds = ds.take(REDUCED_CIFAR_10_TRAIN_SIZE)
    val_ds = ds.skip(REDUCED_CIFAR_10_TRAIN_SIZE).take(REDUCED_CIFAR_10_VAL_SIZE)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": ds['test']
    }


def imagenet(data_dir=None):
    """Imagenet dataset as `tf.data.Dataset` pipeline.
    This dataset loads 32 x 32 samples (which had been prepared by
    applying box resizing method on original high resolution images).

    Dataset homepage: https://patrykchrabaszcz.github.io/Imagenet32/

    """
    ds, info = tfds.load('imagenet_resized/32x32', shuffle_files=True,
                         read_config=tfds.ReadConfig(skip_prefetch=True),
                         as_supervised=True, with_info=True, data_dir=data_dir)
    return {
        "train_ds": ds['train'],
        "val_ds": ds['validation'],
        "info": info
    }


def reduced_svhn(data_dir=None):
    """Reduced SVHN dataset as a `tf.data.Dataset` pipeline.
    The dataset is inspired from work in AutoAugment paper and
    essentially consists of 6000 samples from
    original SVHN dataset.

    Args:
        data_dir: directory read/write data from TFDS
    """
    ds = tfds.load('svhn_cropped', read_config=tfds.ReadConfig(skip_prefetch=True),
                   as_supervised=True, data_dir=data_dir)
    n_train = tf.data.experimental.cardinality(ds['train'])

    ds = ds['train'].shuffle(n_train, seed=SEED)

    train_ds = ds.take(REDUCED_SVHN_TRAIN_SIZE)
    val_ds = ds.skip(REDUCED_SVHN_TRAIN_SIZE).take(REDUCED_SVHN_VAL_SIZE)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds
    }


def reduced_imagenet(data_dir=None):
    """Reduced Imagenet (32 x 32) dataset as a `tf.data.Dataset` pipeline.
    The dataset is inspired from work in AutoAugment paper and
    essentially consists of 6000 samples selected from 120
    random Imagenet classes.

    Args:
        data_dir: directory read/write data from TFDS
    """
    ds, info = tfds.load("imagenet_resized/32x32", shuffle_files=True,
                         read_config=tfds.ReadConfig(skip_prefetch=True),
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


def large_imagenet(data_dir=None):
    """Imagenet dataset as a `tf.data.Dataset` pipeline.

    Note: JPEG decoding is skipped while loading this dataset
    and hence binary strings representing each image are returned.
    This is to improve the pipeline performance when additional
    transformations like random crops are used for training.

    Dataset homepage: http://www.image-net.org/

    Args:
        data_dir: directory read/write data from TFDS"""
    ds, info = tfds.load('imagenet2012', shuffle_files=True,
                         read_config=tfds.ReadConfig(skip_prefetch=True),
                         decoders={'image': tfds.decode.SkipDecoding()},
                         as_supervised=True, with_info=True, data_dir=data_dir)
    ds_dict = {
        "train_ds": ds['train'],
        "val_ds": ds['validation'],
        "info": info
    }

    if 'minival' in info.splits:
        ds_dict['minival_ds'] = ds['minival']

    return ds_dict
