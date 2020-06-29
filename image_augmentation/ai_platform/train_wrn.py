import argparse
import logging

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from matplotlib import pyplot as plt

from image_augmentation.wide_resnet import WideResNet
from image_augmentation.preprocessing import imagenet_standardization, imagenet_baseline_augmentation
from image_augmentation.preprocessing import cifar_standardization, cifar_baseline_augmentation
from image_augmentation.datasets import reduced_cifar10, reduced_svhn, reduced_imagenet
from image_augmentation.datasets import cifar10, svhn, imagenet


def get_args():
    parser = argparse.ArgumentParser(description='Train WRN on Google AI Platform')

    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints of models and other results')
    parser.add_argument(
        '--epochs',
        type=int,
        default=120,
        help='number of times to go through the data, default=120')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='size of each batch, default=128')
    parser.add_argument(
        '--wrn-depth',
        default=40,
        type=int,
        help='depth of Wide ResNet, default=40')
    parser.add_argument(
        '--wrn-k',
        default=2,
        type=int,
        help='widening factor of Wide ResNet, default=2')
    parser.add_argument(
        '--dataset',
        default='cifar10',
        choices=["cifar10", "reduced_cifar10", "svhn",
                 "reduced_svhn", "imagenet", "reduced_imagenet"],
        help='dataset that is to be used for training and evaluating the model, '
             'default="cifar10"'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        type=str,
        help='local or GCS location for accessing data with TFDS '
             '(directory for tensorflow_datasets)'
    )
    parser.add_argument(
        '--init-lr',
        default=0.01,
        type=float,
        help='initial learning rate for training'
    )
    parser.add_argument(
        '--weight-decay',
        default=10e-4,
        type=float,
        help='weight decay of training step'
    )
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args = parser.parse_args()
    return args


NUM_CLASSES = {
    "cifar10": 10,
    "reduced_cifar10": 10,
    "svhn": 10,
    "reduced_svhn": 10,
    "imagenet": 1000,
    "reduced_imagenet": 120
}

BASELINE_METHOD = {
    "cifar10": (cifar_baseline_augmentation, cifar_standardization),
    "reduced_cifar10": (cifar_baseline_augmentation, cifar_standardization),
    "svhn": (cifar_baseline_augmentation, cifar_standardization),
    "reduced_svhn": (cifar_baseline_augmentation, cifar_standardization),
    "imagenet": (imagenet_baseline_augmentation, imagenet_standardization),
    "reduced_imagenet": (imagenet_baseline_augmentation, imagenet_standardization)
}

DATASET = {
    "cifar10": cifar10,
    "reduced_cifar10": reduced_cifar10,
    "svhn": svhn,
    "reduced_svhn": reduced_svhn,
    "imagenet": imagenet,
    "reduced_imagenet": reduced_imagenet
}

SGDR_T0 = 10
SGDR_T_MUL = 2


def main(args):
    # set level of verbosity
    logging.getLogger("tensorflow").setLevel(args.verbosity)

    # image input shape is set 32 x 32
    inp_shape = (32, 32, 3)
    # num classes and other pre-processing ops inferred based on given dataset name
    num_classes = NUM_CLASSES[args.dataset]
    baseline_augment, standardize = BASELINE_METHOD[args.dataset]
    ds = DATASET[args.dataset](args.data_dir)

    # get train and validation/test datasets
    train_ds = ds['train_ds']
    val_ds = ds['val_ds'] if 'val_ds' in ds else ds['test_ds']

    # show dataset distribution only for reduced datasets
    if args.dataset.startswith("reduced"):
        train_distro, val_distro = [tf.math.bincount(
            [label for image, label in curr_ds],
            minlength=num_classes)
            for curr_ds in (train_ds, val_ds)]

        plt.figure(figsize=(15, 4))

        plt.subplot(1, 2, 1)
        plt.bar(tf.range(num_classes).numpy(), train_distro.numpy(), color='y')
        plt.xlabel(args.dataset + " classes")
        plt.ylabel("number of samples")
        plt.title("Training Distribution")

        plt.subplot(1, 2, 2)
        plt.bar(tf.range(num_classes).numpy(), val_distro.numpy(), color='g')
        plt.xlabel(args.dataset + " classes")
        plt.ylabel("number of samples")
        plt.title("Validation Distribution")

        fig_file_path = args.job_dir + "/dataset_distribution.pdf"
        with tf.io.gfile.GFile(fig_file_path, "wb") as fig_file:
            plt.savefig(fig_file, format="pdf")
        print("Wrote file to", fig_file_path)

    wrn = WideResNet(inp_shape, depth=args.wrn_depth, k=args.wrn_k, num_classes=num_classes)
    wrn.summary()

    inp = keras.layers.Input(inp_shape, name='image_input')
    x = baseline_augment(inp)

    # mean normalization of CIFAR10, SVHN require that images be supplied
    if args.dataset.endswith("cifar10") or args.dataset.endswith("svhn"):
        images_only = train_ds.map(lambda image, label: image)
        x = standardize(x, images_only)
    # for ImageNet mean normalization is not required, rescaling is used instead
    else:
        x = standardize(x)

    x = wrn(x)
    # model combines baseline augmentation, standardization and wide resnet layers
    model = keras.Model(inp, x, name='WRN')
    model.summary()

    # use an SGDR optimizer with weight decay
    lr_schedule = keras.experimental.CosineDecayRestarts(args.init_lr, SGDR_T0, SGDR_T_MUL)
    opt = tfa.optimizers.SGDW(args.weight_decay, lr_schedule, momentum=0.9)

    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    # use top-5 accuracy metric with ImageNet and reduced-ImageNet only
    if args.dataset.endswith("imagenet"):
        metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=5))

    model.compile(opt, loss='sparse_categorical_crossentropy', metrics=metrics)

    # prepare tensorboard logging
    tb_path = args.job_dir + '/tensorboard'
    callbacks = [keras.callbacks.TensorBoard(tb_path)]
    print("Using tensorboard directory as", tb_path)

    # cache the dataset only if possible
    if args.dataset not in ['svhn', 'imagenet']:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # shuffle and batch the dataset
    train_ds = train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(args.batch_size)
    val_ds = val_ds.batch(args.batch_size)

    # prefetch dataset for faster access in case of larger datasets only
    if args.dataset in ['svhn', 'imagenet']:
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # train the model
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # save keras model
    save_path = args.job_dir + '/keras_model'
    keras.models.save_model(model, save_path)
    print("Model exported to", save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)