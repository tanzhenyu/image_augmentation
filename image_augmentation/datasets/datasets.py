import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 14


def cifar10():
    ds, info = tfds.load('cifar10', as_supervised=True, with_info=True)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }


def svhn():
    ds, info = tfds.load('svhn_cropped', as_supervised=True, with_info=True)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }


def reduced_cifar10():
    ds = tfds.load('cifar10', as_supervised=True)
    n_train = tf.data.experimental.cardinality(ds['train'])

    ds = ds['train'].shuffle(n_train, seed=SEED)
    train_ds = ds.take(4000)
    val_ds = ds.skip(4000).take(1715)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds
    }


def reduced_svhn():
    ds = tfds.load('svhn_cropped', as_supervised=True)
    n_train = tf.data.experimental.cardinality(ds['train'])

    ds = ds['train'].shuffle(n_train, seed=SEED)
    train_ds = ds.take(1000)
    val_ds = ds.skip(1000).take(430)
    return {
        "train_ds": train_ds,
        "val_ds": val_ds
    }
