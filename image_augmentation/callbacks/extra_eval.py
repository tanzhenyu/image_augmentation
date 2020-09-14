import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class ExtraValidation(Callback):
    def __init__(self, validation_data, tensorboard_path, validation_freq=1):
        super(ExtraValidation, self).__init__()

        self.validation_data = validation_data
        self.tensorboard_path = tensorboard_path

        self.tensorboard_writer = tf.summary.create_file_writer(self.tensorboard_path)

        self.validation_freq = validation_freq

    def on_epoch_end(self, epoch, logs=None):
        # evaluate at an interval of `validation_freq` epochs
        if (epoch + 1) % self.validation_freq == 0:
            metric_names = ['{}_{}'.format('epoch', metric.name)
                            for metric in self.model.metrics]

            scores = self.model.evaluate(self.validation_data, verbose=1)

            with self.tensorboard_writer.as_default():
                for metric_name, score in zip(metric_names, scores):
                    tf.summary.scalar(metric_name, score, step=epoch)
