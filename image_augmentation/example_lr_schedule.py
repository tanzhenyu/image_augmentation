import tensorflow as tf
from matplotlib import pyplot as plt

num_images = 50000
batch_size = 256

train_steps = 10
step_size = 1.0 / (num_images // batch_size)

lr_schedule = tf.keras.experimental.CosineDecayRestarts(0.01, train_steps)

num_epochs = 200
epochs = tf.range(step_size, num_epochs + 1, step_size)

lrs = [lr_schedule(e) for e in epochs]

plt.plot(epochs, lrs)
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.show()
