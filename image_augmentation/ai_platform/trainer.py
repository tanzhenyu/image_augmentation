import argparse
import logging

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from matplotlib import pyplot as plt

from image_augmentation.wide_resnet import WideResNet
from image_augmentation.preprocessing import imagenet_standardization, imagenet_baseline_augmentation
from image_augmentation.datasets import reduced_imagenet


def get_args():
    parser = argparse.ArgumentParser(
        description='Train WRN on Google AI Platform')

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


def main(args):
    logging.getLogger("tensorflow").setLevel(args.verbosity)


if __name__ == '__main__':
    args = get_args()
    main(args)
