import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class TensorBoardLRLogger(Callback):
    def __init__(self, writer_path):
        super(TensorBoardLRLogger, self).__init__()
        self.writer_path = writer_path
        self.writer = tf.summary.create_file_writer(self.writer_path)
        self.use_weight_decay = False
        self.callable_weight_decay = False
        self.callable_learning_rate = False

    def on_train_begin(self, logs=None):
        if hasattr(self.model.optimizer, 'weight_decay'):
            self.use_weight_decay = True
            if hasattr(self.model.optimizer.weight_decay, '__call__'):
                self.callable_weight_decay = True
        if hasattr(self.model.optimizer.learning_rate, '__call__'):
            self.callable_learning_rate = True

    def on_train_batch_end(self, batch, logs=None):
        with self.writer.as_default():
            if self.callable_learning_rate:
                lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
            else:
                lr = self.model.optimizer.learning_rate
            tf.summary.scalar("learning_rate", lr, step=self.model.optimizer.iterations)

            if self.use_weight_decay:
                if self.callable_weight_decay:
                    wd = self.model.optimizer.weight_decay(self.model.optimizer.iterations)
                else:
                    wd = self.model.optimizer.weight_decay
                tf.summary.scalar("weight_decay", wd, step=self.model.optimizer.iterations)
            self.writer.flush()
