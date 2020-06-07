import os

from tensorflow.keras.utils import plot_model

from image_augmentation.wide_resnet.wrn import WideResNet


def visualize_model(model):
    net_dig_path = '/tmp/model.png'
    plot_model(net, net_dig_path, show_shapes=True)
    os.system('open ' + net_dig_path)


def test_wrn_28_10():
    inp_shape = (32, 32, 3)

    net = WideResNet(inp_shape, depth=28, k=10)
    net.summary()

    visualize_model(net)

    assert True


def test_wrn_40_2():
    inp_shape = (32, 32, 3)

    net = WideResNet(inp_shape, depth=40, k=2)
    net.summary()

    visualize_model(net)

    assert True


if __name__ == '__main__':
    test_wrn_28_10()
    test_wrn_40_2()
