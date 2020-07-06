# Image Pipelining

**Image Op(s)**

The following is the list of image processing operations and their current implementation status.

| Image Op Name | Op Description | Status | Test Description | Test Status |
| --- | --- | --- | --- | --- |
| **AutoContrast** | Maximize the the image contrast, by making the darkest pixel black and lightest pixel white. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.autocontrast`.| Passing |
| **Invert** | Invert the pixels of an image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.invert`. | Passing |
| **Equalize** | Equalize the image histogram. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.equalize`. | Failing |
| **Solarize** | Invert all pixels above a threshold value of `magnitude`. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.solarize`. | Passing |
| **Posterize** | Reduce the number of bits for each pixel to `magnitude` bits. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.posterize`. | Passing |
| **Contrast** | Adjust the `magnitude` of contrast of the image. A `magnitude=0` is for gray image whereas `magnitude=1` gives original image. | ✅ Implemented |  Compare results of applying the operation on a random image with `PIL.ImageEnhance.Contrast.enhance`.  | Passing |
| **Color** | Adjust the `magnitude` of color balance of the image. A `magnitude=0` is for grayscale image whereas `magnitude=1` gives original image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageEnhance.Color.enhance`. | Passing |
| **Brightness** | Adjust the `magnitude` of brightness of the image. A `magnitude=0` is for complete black image whereas `magnitude=1` gives original image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageEnhance.Brightness.enhance`. | Passing |
| **Sharpness** | Adjust the `magnitude` of sharpness of the image. A `magnitude=0` is for blurred image whereas `magnitude=1` gives original image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageEnhance.Sharpness.enhance`. | Passing |
| **Cutout** | Set a random square patch of side-length `magnitude` pixels to gray. (https://arxiv.org/abs/1708.04552) | ✅ Implemented | Check the results of the applying the operation, whether a few pixels have gray color values or not. <br/> TODO: Include a test with more rigour. | Passing |
| **Sample Pairing** | Linearly add the image with another image with weight `magnitude`. (https://arxiv.org/abs/1801.02929) | ✅ Implemented | Display two random images paired together. <br/> TODO: Write a test that is good enough to evaluate this. | Passing |

As of now, we're reusing the following image op(s) from the TensorFlow Addons package.

| Image Op Name | Op Description | Equivalent Function |
| --- | --- | --- |
| **Shear** | Shear the image along the horizontal / vertical axis with rate `magnitude`. | `tfa.image.shear_x`, `tfa.image.shear_y` |
| **Translate** | Translate the image in the horizontal / vertical axis with rate `magnitude`. | `tfa.image.translate_xy` |
| **Rotate** | Rotate the image by `magnitude` degrees. | `tfa.image.rotate` |

**Image Data Augmentation**

AutoAugment uses an augmentation policy consisting of multiple sub-policies.
Each of the subpolicy has two image operations associated with a probability and magnitude of effect. (some image operations like `AutoContrast`, `Invert`, `Equalize` do not have any effect of `magnitude` parameter)
For any particular image of the training set, we apply a subpolicy chosen randomly from the AutoAugment policy.
The corresponding image op(s) from the selected subpolicy are then applied on the image basis their individual probabilities.

The AutoAugment paper provides details for policies found on 3 datasets:
1. Reduced ImageNet [[policy](./policy_augmentation.py#L92-L118)]
2. Reduced SVHN [[policy](./policy_augmentation.py#L65-L91)]
3. Reduced CIFAR-10 [[[policy](./policy_augmentation.py#L38-L64)]

