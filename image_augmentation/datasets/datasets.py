import tensorflow_datasets as tfds


def cifar10():
    ds, info = tfds.load('cifar10', as_supervised=True, with_info=True)
    return {
        "train_ds": ds['train'],
        "test_ds": ds['test'],
        "info": info
    }
