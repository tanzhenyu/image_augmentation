from tensorflow import keras
import tensorflow_datasets as tfds

from image_augmentation.wide_resnet.wrn import WideResNet
from image_augmentation.preprocessing.preprocess import preprocess_cifar
from image_augmentation.preprocessing.preprocess import baseline_augmentation

cifar10, info = tfds.load('cifar10', as_supervised=True, with_info=True)
print(info)

images_only = cifar10['train'].map(lambda image, label: image)

inp_shape = info.features['image'].shape
num_classes = info.features['label'].num_classes

wrn_28_10 = WideResNet(inp_shape, depth=28, k=10, num_classes=num_classes)
wrn_28_10.summary()

inp = keras.layers.Input(inp_shape)
x = baseline_augmentation(inp)
x = preprocess_cifar(inp, images_only)
x = wrn_28_10(x)

model = keras.Model(inp, x)
model.summary()

model.compile('adam', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

batch_size = 128
epochs = 40

train_ds = cifar10['train'].cache().shuffle(
        1000, reshuffle_each_iteration=True).batch(batch_size)

val_ds = cifar10['test'].cache().batch(batch_size)

model.fit(train_ds, validation_data=val_ds, epochs=epochs)
