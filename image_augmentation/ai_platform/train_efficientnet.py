"""Train EfficientNet(s) on ImageNet using Cloud TPUs."""

import argparse
import logging

import tensorflow as tf
from tensorflow import keras

from image_augmentation.datasets import large_imagenet
from image_augmentation.image import PolicyAugmentation, autoaugment_policy, RandAugment
from image_augmentation.callbacks import TensorBoardLRLogger


def get_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet on ImageNet dataset')

    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints of models and other results')
    parser.add_argument(
        '--epochs',
        type=int,
        default=350,
        help='number of times to go through the data, default=120')
    parser.add_argument(
        '--train-batch-size',
        default=2048,
        type=int,
        help='size of each training batch, default=2048')
    parser.add_argument(
        '--val-batch-size',
        default=64,
        type=int,
        help='size of each validation batch, default=64')
    parser.add_argument(
        '--model-name',
        default='efficientnet-b5',
        choices=['efficientnet-b5', 'efficientnet-b7'],
        type=str,
        help='EfficientNet architecture that is to be used, default=efficientnet-b5')
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
        '--data-dir',
        default=None,
        type=str,
        help='local or GCS location for accessing data with TFDS '
             '(directory for tensorflow_datasets)')
    parser.add_argument(
        '--optimizer',
        default='rmsprop',
        choices=["rmsprop", "sgd", "adam"],
        help='optimizer that is to be used for training, default="rmsprop"')
    parser.add_argument(
        '--optimizer-decay',
        default=0.9,
        type=float,
        help='optimizer decay rate for RMSprop, default=0.9')
    parser.add_argument(
        '--optimizer-momentum',
        default=0.9,
        type=float,
        help='optimizer momentum rate for RMSprop and SGD, default=0.9')
    parser.add_argument(
        '--label-smoothing',
        default=0.1,
        type=int,
        help='amount of label smoothing to be applied on '
             'categorical cross entropy (softmax) loss, default=0.1')
    parser.add_argument(
        '--base-lr',
        default=0.016,
        type=float,
        help='initial learning rate for training with a base batch size of 256, '
             'the value will be scaled according to batch size specified, default=0.016')
    parser.add_argument(
        '--lr-decay-rate',
        default=0.97,
        type=float,
        help='amount of learning rate (exponential) decay to be applied, default=0.97')
    parser.add_argument(
        '--lr-decay-epochs',
        default=2.4,
        type=float,
        help='number of epochs to wait before each exponential decay, default=10')
    parser.add_argument(
        '--warmup-epochs',
        default=5,
        type=float,
        help='number of warmup epochs, default=5')
    parser.add_argument(
        '--l2-regularization',
        default=1e-5,
        type=float,
        help='amount of L2 regularization (weight decay rate) to be applied on all weights '
             'of the network, default=1e-5')
    parser.add_argument(
        '--tpu',
        default=None,
        type=str,
        help='the gRPC URL of the TPU that is to be used, default=None (TPU not used)'
    )
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args = parser.parse_args()
    return args


EFFICIENTNET = {
    'efficientnet-b5': {
        'image_size': 456,
        'model_builder': keras.applications.efficientnet.EfficientNetB5
    },
    'efficientnet-b7': {
        'image_size': 600,
        'model_builder': keras.applications.efficientnet.EfficientNetB7
    }
}
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def main(args):
    # set level of verbosity
    logging.getLogger("tensorflow").setLevel(args.verbosity)

    # display script args
    print(args)

    if args.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
        # connect to the TPU
        tf.config.experimental_connect_to_cluster(resolver)
        # initialize the TPU system
        tf.tpu.experimental.initialize_tpu_system(resolver)
        # obtain TPU distribution strategy
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        # use default distribution strategy
        strategy = tf.distribute.get_strategy()
    print('Number of available devices: {}'.format(strategy.num_replicas_in_sync))

    image_size = EFFICIENTNET[args.model_name]['image_size']
    print('Using image size: {}'.format(image_size))

    with strategy.scope():
        model_builder = EFFICIENTNET[args.model_name]['model_builder']
        model = model_builder(include_top=True,
                                     weights=None)

        # normalize images using (ImageNet) RGB mean normalization
        normalization_layer = model.get_layer('normalization')
        norm_weights = [tf.convert_to_tensor(MEAN_RGB, tf.float32),
                        tf.convert_to_tensor(STDDEV_RGB, tf.float32),
                        tf.convert_to_tensor(0, tf.float32)]
        normalization_layer.set_weights(norm_weights)

        model.summary()

    ds = large_imagenet(args.data_dir)
    train_ds = ds['train_ds']
    val_ds = ds['val_ds']

    # shuffle and batch the dataset
    train_ds = train_ds.shuffle(1024, reshuffle_each_iteration=True).batch(args.train_batch_size)
    val_ds = val_ds.batch(args.val_batch_size)

    def augment_map_fn_builder(augmenter):
        return lambda images, labels: (tf.py_function(augmenter, [images], images.dtype), labels)

    # apply AutoAugment (data augmentation) on training pipeline
    if args.auto_augment:
        # use EfficientNet's AutoAugment policy
        policy = autoaugment_policy('imagenet', efficientnet=True)

        # set hyper parameters to appropriate size
        auto_augment = PolicyAugmentation(policy, translate_max=250, cutout_max_size=100)
        augment_map_fn = augment_map_fn_builder(auto_augment)
        train_ds = train_ds.map(augment_map_fn)  # refrain from using AUTOTUNE here, tf.py_func cannot parallel execute

    # apply RandAugment on training pipeline
    if args.rand_augment_n:
        rand_augment = RandAugment(args.rand_augment_m, args.rand_augment_n,
                                   # set hyper parameters to appropriate size
                                   translate_max=250, cutout_max_size=100)
        augment_map_fn = augment_map_fn_builder(rand_augment)
        train_ds = train_ds.map(augment_map_fn)

    # prefetch dataset for faster access
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    # calculate steps per epoch for optimizer schedule num steps
    steps_per_epoch = tf.data.experimental.cardinality(train_ds)
    steps_per_epoch = steps_per_epoch.numpy()  # helps model optimizer become JSON serializable

    init_lr = args.base_lr * (args.train_batch_size / 256.)
    lr = init_lr

    with strategy.scope():
        if args.optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=lr,
                                           rho=args.optimizer_decay,
                                           momentum=args.optimizer_momentum)
        elif args.optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=lr,
                                       momentum=args.optimizer_momentum)
        else:
            opt = keras.optimizers.Adam(learning_rate=lr)

        crossentropy_loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=args.label_smoothing)

        if args.l2_regularization:
            for var in model.trainable_variables:
                model.add_loss(lambda: keras.regularizers.L2(
                    args.l2_regularization)(var))

    metrics = [keras.metrics.SparseCategoricalAccuracy(),
               keras.metrics.SparseTopKCategoricalAccuracy(k=5)]
    model.compile(opt, crossentropy_loss, metrics)

    # prepare for tensorboard logging and model checkpoints
    tb_path = args.job_dir + '/tensorboard'
    checkpoint_path = args.job_dir + '/checkpoint'
    callbacks = [keras.callbacks.TensorBoard(tb_path),
                 keras.callbacks.ModelCheckpoint(checkpoint_path),
                 TensorBoardLRLogger(tb_path + '/train')]
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