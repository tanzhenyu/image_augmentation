import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class ExponentialDecayStaircaseIntervals(LearningRateSchedule):
    def __init__(self,
                 initial_learning_rate,
                 decay_steps,
                 decay_rate):
        super(ExponentialDecayStaircaseIntervals, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        decay_steps = tf.convert_to_tensor(self.decay_steps, dtype=tf.float32)
        exp_factor = tf.cast(step >= decay_steps, tf.float32)
        exp_factor = tf.reduce_sum(exp_factor)
        return self.initial_learning_rate * (self.decay_rate ** exp_factor)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate
        }
