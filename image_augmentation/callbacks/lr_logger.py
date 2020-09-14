import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class TensorBoardLRLogger(Callback):
    """Log updates of the optimizer learning rate to TensorBoard during training.
    Learning rate is logged at the end of each epoch.

    The callback also attempts to log the weight decay applied by the optimizer
    in case the optimizer makes use of a callable weight decay schedule or a fixed
    value of weight decay. (eg. SGDW, AdamW)

    Args:
        writer_path: Path to the TensorBoard logging directory.
    """
    def __init__(self, writer_path):
        super(TensorBoardLRLogger, self).__init__()
        self.writer_path = writer_path
        self.writer = tf.summary.create_file_writer(self.writer_path)
        self.use_weight_decay = False
        self.callable_weight_decay = False
        self.callable_learning_rate = False

    def on_train_begin(self, logs=None):
        # determine if the model's optimizer uses weight decay
        if hasattr(self.model.optimizer, 'weight_decay'):
            self.use_weight_decay = True
            # determine if callable weight decay schedule is used
            if hasattr(self.model.optimizer.weight_decay, '__call__'):
                self.callable_weight_decay = True
        # determine if callable learning rate schedule is used
        if hasattr(self.model.optimizer.learning_rate, '__call__'):
            self.callable_learning_rate = True

    def on_epoch_end(self, batch, logs=None):
        with self.writer.as_default():
            # callable learning rate schedule
            if self.callable_learning_rate:
                lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
            # fixed learning rate
            else:
                lr = self.model.optimizer.learning_rate
            tf.summary.scalar("learning_rate", lr, step=self.model.optimizer.iterations)

            if self.use_weight_decay:
                # callable weight decay schedule
                if self.callable_weight_decay:
                    wd = self.model.optimizer.weight_decay(self.model.optimizer.iterations)
                # fixed weight decay
                else:
                    wd = self.model.optimizer.weight_decay
                tf.summary.scalar("weight_decay", wd, step=self.model.optimizer.iterations)