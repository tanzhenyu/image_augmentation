"""Train EfficientNet(s) on ImageNet using Cloud TPUs."""

import argparse
import logging

import tensorflow as tf
from tensorflow import keras


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
        '--l2-reg',
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
        efficientnet = model_builder(include_top=True,
                                     weights=None)

        # normalize images using (ImageNet) RGB mean normalization
        normalization_layer = efficientnet.get_layer('normalization')
        norm_weights = [tf.convert_to_tensor(MEAN_RGB, tf.float32),
                        tf.convert_to_tensor(STDDEV_RGB, tf.float32),
                        tf.convert_to_tensor(0, tf.float32)]
        normalization_layer.set_weights(norm_weights)

        efficientnet.summary()


if __name__ == '__main__':
    args = get_args()
    main(args)