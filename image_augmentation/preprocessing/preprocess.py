from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

def preprocess_imagenet(x):
    # scale images to a range (-1, +1)
    x = Rescaling(scale=1 / 127.5, offset=-1, name='rescaling')(x)
    return x


def preprocess_cifar(x, data_samples):
    norm_layer = Normalization(name='mean_normalization')
    norm_layer.adapt(data_samples)

    x = norm_layer(x)
    return x
