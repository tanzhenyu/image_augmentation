from tensorflow.keras.callbacks import Callback


class ExtraValidation(Callback):
    def __init__(self, validation_data):
        super(ExtraValidation, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        self.model.evaluate(self.validation_data, verbose=1)
