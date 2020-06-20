# Image Pipelining

The following is the list of image processing operations and their current implementation status.

| Image Op Name | Op Description | Status | Test Description | Test Status |
| --- | --- | --- | --- | --- |
| **Invert** | Invert the pixels of an image. | ✅ Implemented |Compare results of applying the operation on a random image with PIL.ImageOps.invert.|PASSING|
| **Cutout** | Set a random square patch of side-length `magnitude` pixels to gray. (https://arxiv.org/abs/1708.04552) | ✅ Implemented |Check the results of the applying the operation, whether a few pixels have gray color values or not.<br />TODO: Include a test with more rigour.|PASSING|
| **Posterize** | Reduce the number of bits for each pixel to `magnitude` bits. | ✅ Implemented |Compare results of applying the operation on a random image with PIL.ImageOps.posterize.|PASSING|
| **Solarize** | Invert all pixels above a threshold value of `magnitude`. | ✅ Implemented |Compare results of applying the operation on a random image with PIL.ImageOps.solarize.|PASSING|
| **Equalize** | Equalize the image histogram. | ✅ Implemented |Compare results of applying the operation on a random image with PIL.ImageOps.equalize.|FAILING|
| **AutoContrast** | Maximize the the image contrast, by making the darkest pixel black and lightest pixel white. | ❌ WIP |Compare results of applying the operation on a random image with PIL.ImageOps.autocontrast.|N/A|
| **Sharpness** | Adjust the sharpness of the image in the `magnitude` range [0, 1]. | ❌ WIP |||
| **Color** | Adjust the color balance of the image in the `magnitude` range [0, 1]. | ❌ WIP |||
| **Shear** | Shear the image along the horizontal / vertical axis with rate `magnitude`. | ❌ WIP |||
| **Sample Pairing** | Linearly add the image with another image (selected at random from the same mini-batch) with weight magnitude, without changing the label. (https://arxiv.org/abs/1801.02929) | ❌ WIP |||
