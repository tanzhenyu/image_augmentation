import tensorflow as tf
from tensorflow import keras

from image_augmentation.image import PolicyAugmentation
from image_augmentation.wide_resnet import WideResNet
from image_augmentation.preprocessing import cifar_baseline_augmentation, cifar_standardization


class ChildNetwork:
    def __init__(self, policy, steps_per_epoch):
        self.policy = policy
        self.augmenter = PolicyAugmentation(self.policy, cutout_max_size=16, translate_max=16)
        self.steps_per_epoch = steps_per_epoch

        self.input_shape = [32, 32, 3]
        self.num_classes = 10

        self.wrn_depth = 40
        self.wrn_k = 2

        self.learning_rate = 0.01
        self.l2_reg = 10e-4
        self.epochs = 120

        self.model = self.build_model()

    def build_model(self):
        wrn = WideResNet(self.input_shape, self.wrn_depth, self.wrn_k, self.num_classes)

        inp = keras.layers.Input(self.input_shape, name='image_input')
        x = cifar_baseline_augmentation(inp, cutout=True)
        x = cifar_standardization(x)
        x = wrn(x)
        model = keras.Model(inp, x, name='Child-Network')
        return model
