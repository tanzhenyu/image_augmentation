"""Train WideResNet(s) on Google AI platform or locally."""

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
from image_augmentation.image import autoaugment_policy, PolicyAugmentation, RandAugment


def get_args():
    parser = argparse.ArgumentParser(description='Train WideResNet on Google AI Platform')

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
        '--wrn-dropout',
        default=0.0,
        type=float,
        help='dropout value for WideResNet blocks, default=0.0, no dropout')
    parser.add_argument(
        '--no-cutout',
        default=False,
        const=True,
        action='store_const',
        help='use random Cutout for data augmentation, off by default (uses Cutout)')
    parser.add_argument(
        '--auto-augment',
        default=False,
        const=True,
        action='store_const',
        help='apply AutoAugment policy for data augmentation on training, off by default (no AutoAugment)')
    parser.add_argument(
        '--rand-augment-n',
        default=0,
        type=int,
        help='apply RandAugment with number of (N) image transforms for data augmentation on training, '
             'default=0, off (no RandAugment)')
    parser.add_argument(
        '--rand-augment-m',
        default=10,
        type=int,
        help='magnitude (M) of applying each image transform for RandAugment, '
             '(only when using RandAugment) default=10')
    parser.add_argument(
        '--dataset',
        default='cifar10',
        choices=["cifar10", "reduced_cifar10", "svhn",
                 "reduced_svhn", "imagenet", "reduced_imagenet"],
        help='dataset that is to be used for training and evaluating the model, default="cifar10"')
    parser.add_argument(
        '--data-dir',
        required=False,
        default=None,
        type=str,
        help='local or GCS location for accessing data with TFDS '
             '(directory for tensorflow_datasets)')
    parser.add_argument(
        '--padding-mode',
        choices=['zero', 'reflect'],
        default='reflect',
        help='padding mode to be used to pad pixels before cropping, '
             '(applicable only in case of CIFAR, SVHN) default is "reflect"')
    parser.add_argument(
        '--normalization',
        choices=['pixel_center', 'rgb_normalization'],
        default='rgb_normalization',
        help='normalization that is to be applied on images, '
             '(applicable only in case of CIFAR, SVHN) default is "rgb_normalization"')
    parser.add_argument(
        '--optimizer',
        default='sgdr',
        choices=["sgd", "adam", "sgdr"],
        help='optimizer that is to be used for training, default="sgd"')
    parser.add_argument(
        '--init-lr',
        default=0.01,
        type=float,
        help='initial learning rate for training, default=0.01')
    parser.add_argument(
        '--sgdr-t0',
        default=10,
        type=float,
        help='number of steps to decay over for SGDR, default=10')
    parser.add_argument(
        '--sgdr-t-mul',
        default=2,
        type=int,
        help='number of iterations in ith period for SGDR, default=2')
    parser.add_argument(
        '--drop-lr-by',
        default=0.0,
        type=float,
        help='drop learning rate by a factor (only when using SGD, not SGDR), '
             'default=0.0, off')
    parser.add_argument(
        '--drop-lr-every',
        default=[],
        action='append',
        type=int,
        help='drop learning rate duration of epochs (only when using SGD, not SGDR), '
             'default=60')
    parser.add_argument(
        '--sgd-nesterov',
        default=False,
        const=True,
        action='store_const',
        help='use Nesterov accelerated gradient with SGD optimizer, by default Nesterov is off')
    parser.add_argument(
        '--weight-decay-rate',
        default=0.0,
        type=float,
        help='rate of weight decay per training step, note: this '
             'rate is multiplied by learning rate to compute decay value '
             'before passing to SGDW / AdamW, default=0.0, off')
    parser.add_argument(
        '--l2-reg',
        default=0.0005,
        type=float,
        help='L2 regularization to be applied on all weights '
             'of the network, default=0.0005')
    parser.add_argument(
        '--multi-gpu',
        default=False,
        const=True,
        action='store_const',
        help='single host multi-GPU sync training, default is off')
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


def main(args):
    # set level of verbosity
    logging.getLogger("tensorflow").setLevel(args.verbosity)

    # display script args
    print(args)

    if args.multi_gpu:
        # set GPU memory growth to avoid fork cannot allocate memory warning in multi-GPU env
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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

    if args.multi_gpu:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    print('Number of available devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        wrn = WideResNet(inp_shape, depth=args.wrn_depth, k=args.wrn_k,
                         dropout=args.wrn_dropout, num_classes=num_classes)
        wrn.summary()

        inp = keras.layers.Input(inp_shape, name='image_input')

    # whether to use cutout in baseline augmentation step
    cutout = not args.no_cutout

    with strategy.scope():
        # CIFAR-10 and SVHN uses similar augmentation
        if args.dataset.endswith("cifar10") or args.dataset.endswith("svhn"):
            x = baseline_augment(inp, args.padding_mode, cutout)
        # ImageNet uses different
        else:
            x = baseline_augment(inp)

    with strategy.scope():
        if args.dataset.endswith("cifar10") or args.dataset.endswith("svhn"):
            # pixel center of CIFAR10, SVHN require that image samples be supplied
            # (to learn the pixel wise mean)
            if args.normalization == 'pixel_center':
                images_only = train_ds.map(lambda image, label: image)
                x = standardize(x, mode='pixel_mean_subtract', data_samples=images_only)
            # rgb normalization uses rescaling using known RGB mean(s) and std(s)
            else:
                x = standardize(x, mode='feature_normalize')
        # for ImageNet rescaling is used to scale inputs to range [-1, +1]
        else:
            x = standardize(x)

        x = wrn(x)
        # model combines baseline augmentation, standardization and wide resnet layers
        model = keras.Model(inp, x, name='WRN')
        model.summary()

    # cache the dataset only if possible
    if args.dataset not in ['svhn', 'imagenet']:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()

    # shuffle and batch the dataset
    train_ds = train_ds.shuffle(1000, reshuffle_each_iteration=True).batch(args.batch_size)
    val_ds = val_ds.batch(args.batch_size)

    def augment_map_fn_builder(augmenter):
        return lambda images, labels: (tf.py_function(augmenter, [images], images.dtype), labels)

    # apply AutoAugment (data augmentation) on training pipeline
    if args.auto_augment:
        # ensure AutoAugment policy dataset name always starts with "reduced_"
        policy_ds_name = "reduced_" + args.dataset if not args.dataset.startswith("reduced_") else args.dataset
        policy = autoaugment_policy(policy_ds_name)

        # set hyper parameters to size 16 as input size is 32 x 32
        auto_augment = PolicyAugmentation(policy, translate_max=16, cutout_max_size=16)
        augment_map_fn = augment_map_fn_builder(auto_augment)
        train_ds = train_ds.map(augment_map_fn)  # refrain from using AUTOTUNE here, tf.py_func cannot parallel execute

    # apply RandAugment on training pipeline
    if args.rand_augment_n:
        rand_augment = RandAugment(args.rand_augment_m, args.rand_augment_n,
                                   # set hyper parameters to size 16 as input size is 32 x 32
                                   translate_max=16, cutout_max_size=16)
        augment_map_fn = augment_map_fn_builder(rand_augment)
        train_ds = train_ds.map(augment_map_fn)

    # prefetch dataset for faster access in case of larger datasets only (which are not cached)
    if args.dataset in ['svhn', 'imagenet']:
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # calculate steps per epoch for optimizer schedule num steps
    steps_per_epoch = tf.data.experimental.cardinality(train_ds)
    steps_per_epoch = steps_per_epoch.numpy()  # helps model optimizer become JSON serializable

    with strategy.scope():
        # any one of the following:
        # - use an SGD optimizer w/ or w/o weight decay (SGDW / SGD) or just Adam
        # - use a callable learning rate schedule for SGDR or not
        # - use a callable weight decay schedule whenever required
        # - use SGD Nesterov or not
        if args.optimizer == 'sgdr':
            lr = keras.experimental.CosineDecayRestarts(args.init_lr, steps_per_epoch * args.sgdr_t0,
                                                        args.sgdr_t_mul)
            if args.weight_decay_rate:
                weight_decay = keras.experimental.CosineDecayRestarts(args.weight_decay_rate * args.init_lr,
                                                                      steps_per_epoch * args.sgdr_t0, args.sgdr_t_mul)
        elif args.drop_lr_by:
            lr_boundaries = [(steps_per_epoch * epoch) for epoch in sorted(args.drop_lr_every)]
            lr_values = [args.init_lr * (args.drop_lr_by ** idx) for idx in range(len(lr_boundaries) + 1)]
            lr = keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, lr_values)
            if args.weight_decay_rate:
                wd_values = [args.weight_decay_rate * lr_val for lr_val in lr_values]
                weight_decay = keras.optimizers.schedules.PiecewiseConstantDecay(lr_boundaries, wd_values)
        else:
            lr = args.init_lr
            if args.weight_decay_rate:
                weight_decay = args.weight_decay_rate * lr

        if args.optimizer.startswith('sgd'):
            if args.weight_decay_rate:
                opt = tfa.optimizers.SGDW(weight_decay, lr, momentum=0.9, nesterov=args.sgd_nesterov)
            else:
                opt = keras.optimizers.SGD(lr, momentum=0.9, nesterov=args.sgd_nesterov)
        else:  # adam
            if args.weight_decay_rate:
                opt = tfa.optimizers.AdamW(weight_decay, lr)
            else:
                opt = keras.optimizers.Adam(lr)

        metrics = [keras.metrics.SparseCategoricalAccuracy()]
        # use top-5 accuracy metric with ImageNet and reduced-ImageNet only
        if args.dataset.endswith("imagenet"):
            metrics.append(keras.metrics.SparseTopKCategoricalAccuracy(k=5))

        if args.l2_reg != 0:
            for var in model.trainable_variables:
                model.add_loss(lambda: keras.regularizers.L2(args.l2_reg)(var))

        model.compile(opt, loss='sparse_categorical_crossentropy', metrics=metrics)

    # prepare tensorboard logging
    tb_path = args.job_dir + '/tensorboard'
    checkpoint_path = args.job_dir + '/checkpoint'
    callbacks = [keras.callbacks.TensorBoard(tb_path), keras.callbacks.ModelCheckpoint(checkpoint_path)]

    print("Using tensorboard directory as", tb_path)

    # train the model
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # save keras model
    save_path = args.job_dir + '/keras_model'
    keras.models.save_model(model, save_path)
    print("Model exported to", save_path)


if __name__ == '__main__':
    args = get_args()
    main(args)
