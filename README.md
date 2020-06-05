# Image Augmentation

TensorFlow 2 implementation of curated image augmentation techniques for improving regularization and accuracy of modern classifiers trained using popular CNN architectures. The data augmentation strategies are automatically searched from the data using policy optimisation methods and discrete state space search opposed to manually designed random data augmentation operations. The augmentation policies learned from the data vastly increases validation accuracy of the image classifiers.

In this repository, we provide implementation for the following research papers (both are authored by the Google Brain team):
- ["AutoAugment: Learning Augmentation Strategies from Data"](https://arxiv.org/abs/1805.09501) by Cubuk et al.
- ["RandAugment: Practical automated data augmentation with a reduced search space"](https://arxiv.org/abs/1909.13719) by Cubuk et al.

This project is currently a work in progress.
