# Image Augmentation

TensorFlow 2 implementation of curated image augmentation techniques for improving regularization and accuracy of modern classifiers trained using popular CNN architectures.
The data augmentation strategies were automatically searched from the data using policy optimisation methods and discrete state space search opposed to manually designed random data augmentation operations. 
The augmentation policies learned from the data vastly increases validation accuracy of the image classifiers.

In this repository, we provide implementation for the following research papers (both papers are authored by the Google Brain team):
- ["AutoAugment: Learning Augmentation Strategies from Data"](https://arxiv.org/abs/1805.09501) by Ekin D. Cubuk, Barret Zoph et al.
- ["RandAugment: Practical automated data augmentation with a reduced search space"](https://arxiv.org/abs/1909.13719) by Ekin D. Cubuk, Barret Zoph et al.

Some parts of this work have been accelerated with the help of [Cloud TPU](https://cloud.google.com/tpu) resources provided by the [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc).

## Components
This project is currently a work in progress and includes the following components as of now.
- [WideResNet(s)](./image_augmentation/wide_resnet)
    - Wide Residual Networks (https://arxiv.org/abs/1605.07146)
- [Image Ops(s)](./image_augmentation/image#image-ops)
    - AutoContrast
    - Invert 
    - Equalize 
    - Solarize 
    - Posterize 
    - Contrast
    - Color
    - Brightness
    - Sharpness
    - Cutout (https://arxiv.org/abs/1708.04552)
    - Sample Pairing (https://arxiv.org/abs/1801.02929)
- [Pre-processing](./image_augmentation/preprocessing) using Baseline Augmentation and Standardization or Rescaling
- [Image Data Augmentation](./image_augmentation/image#image-data-augmentation)
    - Data augmentation using AutoAugment policies / RandAugment strategy
    - [Policy Augmentation](image_augmentation/image/augmentation.py#L386-L487) (for AutoAugment)
    - [RandAugment](image_augmentation/image/augmentation.py#L522-L635)

## Datasets

The following are the list of datasets that we're currently looking at for augmentation policy search and image classification:
1. [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10), Reduced CIFAR-10
2. [CIFAR-100](https://www.tensorflow.org/datasets/catalog/cifar100)
3. [SVHN](https://www.tensorflow.org/datasets/catalog/svhn_cropped), Reduced SVHN
4. [ImageNet (ILSVRC 2012)](http://image-net.org/), Reduced ImageNet

## Installation

The pre-requisites for this project include installation of TensorFlow v2.3 or tf-nightly package.
Additional dependencies include [TFDS-nightly](https://github.com/tensorflow/datasets) and [TFA-nightly](https://github.com/tensorflow/addons), both of which require pre-release versions.

```bash
git clone https://github.com/tanzhenyu/image_augmentation
cd image_augmentation
pip3 install -r requirements.txt
python3 setup.py install
```

## Tests

Post installation, the tests can be run as follows:

```bash
cd image_augmentation
pipenv install --dev --skip-lock --pre 
pipenv shell pytest
```
