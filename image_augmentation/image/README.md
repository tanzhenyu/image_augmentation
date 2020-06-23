# Image Pipelining

**Image Op(s)**

The following is the list of image processing operations and their current implementation status.

| Image Op Name | Op Description | Status | Test Description | Test Status |
| --- | --- | --- | --- | --- |
| **Shear** | Shear the image along the horizontal / vertical axis with rate `magnitude`. | ❌ WIP |  | N/A |
| **Translate** | Translate the image in the horizontal / vertical axis with rate `magnitude`. | ❌ WIP |  | N/A |
| **Rotate** | Rotate the image by `magnitude` degrees. | ❌ WIP |  | N/A | 
| **AutoContrast** | Maximize the the image contrast, by making the darkest pixel black and lightest pixel white. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.autocontrast`.| Passing |
| **Invert** | Invert the pixels of an image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.invert`. | Passing |
| **Equalize** | Equalize the image histogram. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.equalize`. | Failing |
| **Solarize** | Invert all pixels above a threshold value of `magnitude`. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.solarize`. | Passing |
| **Posterize** | Reduce the number of bits for each pixel to `magnitude` bits. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageOps.posterize`. | Passing |
| **Contrast** | Adjust the `magnitude` of contrast of the image. A `magnitude=0` is for gray image whereas `magnitude=1` gives original image. | ❌ WIP |  | N/A |
| **Color** | Adjust the `magnitude` of color balance of the image. A `magnitude=0` is for grayscale image whereas `magnitude=1` gives original image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageEnhance.Color.enhance`. | Passing |
| **Brightness** | Adjust the `magnitude` of brightness of the image. A `magnitude=0` is for complete black image whereas `magnitude=1` gives original image. | ❌ WIP |  | N/A |
| **Sharpness** | Adjust the `magnitude` of sharpness of the image. A `magnitude=0` is for blurred image whereas `magnitude=1` gives original image. | ✅ Implemented | Compare results of applying the operation on a random image with `PIL.ImageEnhance.Sharpness.enhance`. | Passing |
| **Cutout** | Set a random square patch of side-length `magnitude` pixels to gray. (https://arxiv.org/abs/1708.04552) | ✅ Implemented | Check the results of the applying the operation, whether a few pixels have gray color values or not. <br/> TODO: Include a test with more rigour. | Passing |
| **Sample Pairing** | Linearly add the image with another image with weight `magnitude`. (https://arxiv.org/abs/1801.02929) | ✅ Implemented | Display two random images paired together. <br/> TODO: Write a test that is good enough to evaluate this. | Passing |

**Image Pipeline**

Coming soon.