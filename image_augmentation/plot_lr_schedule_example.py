import tensorflow as tf
from matplotlib import pyplot as plt

sgdr_t0 = 10
sgdr_t_mul = 2

num_images = 50000  # CIFAR-10
batch_size = 128

steps_per_epoch = (num_images // batch_size)
step_size = 1.0 / steps_per_epoch

lr_schedule = tf.keras.experimental.CosineDecayRestarts(0.01, sgdr_t0, sgdr_t_mul)

num_epochs = 200
batches = tf.linspace(step_size, num_epochs, steps_per_epoch)

lrs = tf.map_fn(lr_schedule, batches)

plt.plot(batches, lrs)
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.show()
