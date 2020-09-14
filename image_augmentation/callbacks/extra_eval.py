import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class ExtraValidation(Callback):
    """Log evaluation metrics of an extra validation set. This callback
    is useful for model training scenarios where multiple validation sets
    are used for evaluation (as Keras by default, provides functionality for
    evaluating on a single validation set only).

    The evaluation metrics are also logged to TensorBoard.

    Args:
        validation_data: a tf.data.Dataset pipeline used to evaluate the
            model, essentially an extra validation dataset.
        tensorboard_path: path to the TensorBoard logging directory.
        validation_freq: number of epochs to wait before performing
            subsequent evaluations.
    """
    def __init__(self, validation_data, tensorboard_path, validation_freq=1):
        super(ExtraValidation, self).__init__()

        self.validation_data = validation_data
        self.tensorboard_path = tensorboard_path

        self.tensorboard_writer = tf.summary.create_file_writer(self.tensorboard_path)

        self.validation_freq = validation_freq

    def on_epoch_end(self, epoch, logs=None):
        # evaluate at an interval of `validation_freq` epochs
        if (epoch + 1) % self.validation_freq == 0:
            # gather metric names form model
            metric_names = ['{}_{}'.format('epoch', metric.name)
                            for metric in self.model.metrics]
            # TODO: fix `model.evaluate` memory leak on TPU
            # gather the evaluation metrics
            scores = self.model.evaluate(self.validation_data, verbose=2)

            # gather evaluation metrics to TensorBoard
            with self.tensorboard_writer.as_default():
                for metric_name, score in zip(metric_names, scores):
                    tf.summary.scalar(metric_name, score, step=epoch)
