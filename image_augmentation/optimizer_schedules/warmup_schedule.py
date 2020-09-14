"""Exponential decay learning rate schedule with support for warmup."""

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class WarmupExponentialDecay(ExponentialDecay):
    """Exponentially decay the learning rate with a few
    starting warmup steps.

    Args:
        initial_learning_rate: The initial learning rate.
        decay_steps: The number of steps to decay over. Must be positive.
        decay_rate: The rate of decay.
        warmup_steps: The number of starting warmup steps to use. Must be positive.
        staircase: If True decay the learning rate at discrete intervals
        name: Optional name of the operation. Defaults to 'WarmupExponentialDecay'.
    """
    def __init__(self, initial_learning_rate, decay_steps,
                 decay_rate, warmup_steps, staircase=False,
                 name=None):
        super(WarmupExponentialDecay, self).__init__(initial_learning_rate,
                                                     decay_steps,
                                                     decay_rate,
                                                     staircase,
                                                     name)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmupExponentialDecay"):
            is_warmup_step = step < self.warmup_steps
            warmup_lr = (
                    self.initial_learning_rate *
                    tf.cast(step, tf.float32) /
                    tf.cast(self.warmup_steps, tf.float32))

            decay_lr = super(WarmupExponentialDecay, self).__call__(step)

            return tf.cond(is_warmup_step,
                           lambda: warmup_lr,
                           lambda: decay_lr)

    def get_config(self):
        config = {'warmup_steps': self.warmup_steps}
        base_config = super(WarmupExponentialDecay, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
