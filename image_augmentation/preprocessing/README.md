# Baseline Pre-processing

| Method name | Description | Keras layer(s) used |
| --- | --- | --- |
| cifar_standardization | Standardize image channels to a normal distribution with zero mean and unit variance. | `tf.keras.layers.experimental.preprocessing.Normalization` |
| imagenet_standardization | Rescale image intensities into a range [-1.0, +1.0] instead of [0, 255]. | `tf.keras.layers.experimental.preprocessing.Rescaling` |
| cifar_baseline_augmentation | <ul> <li>Add zero / reflect padding of 4 x 4 pixels to the images on each side.</li> <li>Apply horizontal flip on images with a probability of 0.5.</li> <li>Apply random Cutout of patches with size 16 pixels. (optional)</li> <li>Take random crop of 32 x 32 pixels</li> </ul> | <ul> <li>`tf.keras.layers.ZeroPadding2D` / [`ReflectPadding2D`](../image/layers.py)</li> <li>`tf.keras.layers.experimental.preprocessing.RandomFlip`</li> <li>[`RandomCutout`](../image/layers.py)</li> <li>`tf.keras.layers.experimental.preprocessing.RandomCrop`</li> </ul>  |
| imagenet_baseline_augmentation |  Apply horizontal flip on images with a probability of 0.5. | `tf.keras.layers.experimental.preprocessing.RandomFlip` |

These baseline pre-processing strategies are as per details provided in [AutoAugment](https://arxiv.org/abs/1805.09501) / [WideResNet](https://arxiv.org/abs/1605.07146) / [SGDR](https://arxiv.org/abs/1608.03983) paper. 
In case of baseline augmentation for ImageNet dataset, random distortion of colors have been removed as the paper discusses that removal of this operation does not change the results for AutoAugment.
