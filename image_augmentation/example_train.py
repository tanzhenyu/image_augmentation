from tensorflow import keras

from image_augmentation.wide_resnet import WideResNet
from image_augmentation.preprocessing import cifar_standardization, cifar_baseline_augmentation
from image_augmentation.datasets import cifar10, reduced_cifar10

ds = cifar10()

info = ds['info']
inp_shape = info.features['image'].shape
num_classes = info.features['label'].num_classes

ds = reduced_cifar10()
train_ds, val_ds = ds['train_ds'], ds['val_ds']

images_only = train_ds.map(lambda image, label: image)

wrn_40_2 = WideResNet(inp_shape, depth=40, k=2, num_classes=num_classes)
wrn_40_2.summary()

inp = keras.layers.Input(inp_shape)
x = cifar_standardization(inp, images_only)
x = cifar_baseline_augmentation(x)
x = wrn_40_2(x)

model = keras.Model(inp, x)
model.summary()

batch_size = 128
epochs = 200
restart_steps = 10
init_learn_rate = 0.01
weight_decay = 10e-4

train_ds = train_ds.cache().shuffle(
        1000, reshuffle_each_iteration=True).batch(batch_size)
val_ds = val_ds.cache().batch(batch_size)

lr_schedule = keras.experimental.CosineDecayRestarts(init_learn_rate, restart_steps)
opt = keras.optimizers.SGD(lr_schedule, momentum=0.9)

l2 = keras.regularizers.L2(weight_decay)
for layer in wrn_40_2.layers:
    layer.activity_regularizer = l2

model.compile(opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=epochs)
