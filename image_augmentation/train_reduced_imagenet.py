import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

from matplotlib import pyplot as plt

from image_augmentation.wide_resnet import WideResNet
from image_augmentation.preprocessing import imagenet_standardization, imagenet_baseline_augmentation
from image_augmentation.datasets import reduced_imagenet

inp_shape = (32, 32, 3)
num_classes = 120

ds = reduced_imagenet()
train_ds, val_ds = ds['train_ds'], ds['val_ds']

train_distro, val_distro = [tf.math.bincount(
                                [label for image, label in curr_ds],
                                minlength=num_classes)
                            for curr_ds in (train_ds, val_ds)]

plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
plt.bar(tf.range(num_classes).numpy(), train_distro.numpy(), color='y')
plt.xlabel("Reduced ImageNet Classes")
plt.ylabel("Number of Samples")
plt.ylim(0, 75)
plt.title("Reduced ImageNet Training Distribution")

plt.subplot(1, 2, 2)
plt.bar(tf.range(num_classes).numpy(), val_distro.numpy(), color='g')
plt.xlabel("Reduced ImageNet Classes")
plt.ylabel("Number of Samples")
plt.ylim(0, 75)
plt.title("Reduced ImageNet Validation Distribution")

plt.savefig("dataset_distribution.pdf")

wrn_40_2 = WideResNet(inp_shape, depth=40, k=2, num_classes=num_classes)
wrn_40_2.summary()

inp = keras.layers.Input(inp_shape, name='image_input')
x = imagenet_baseline_augmentation(inp)

# mean normalization of images require that images be supplied
x = imagenet_standardization(x)
x = wrn_40_2(x)

model = keras.Model(inp, x, name='WRN')
model.summary()

batch_size = 128
epochs = 200
sgdr_t_0 = 10
sgdr_t_mul = 2
init_learn_rate = 0.1
weight_decay = 10e-5

train_ds = train_ds.cache().shuffle(
        1000, reshuffle_each_iteration=True).batch(batch_size)
val_ds = val_ds.cache().batch(batch_size)

lr_schedule = keras.experimental.CosineDecayRestarts(init_learn_rate,
                                                     sgdr_t_0, sgdr_t_mul)
opt = tfa.optimizers.SGDW(weight_decay, lr_schedule,
                          momentum=0.9)

model.compile(opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [keras.callbacks.TensorBoard('./wrn-reduced-imagenet')]

model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
