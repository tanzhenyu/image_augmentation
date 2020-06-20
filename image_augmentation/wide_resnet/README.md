# Wide Residual Networks (WRNs)

https://arxiv.org/abs/1605.07146

Implementation of WideResNet(s) using `tf.keras`.

**Highlights:**

- Deep residual networks with architecture similar to ResNets except that the number of filters in each block gets multiplied by a factor `k`.
- No use any max-pooling layer.
- The usual notation is `WideResNet-n-k`. eg. WideResNet-28-10, WideResNet-40-2 and so on.
    - `n` - being the depth of the network
    - `k` - factor with which conv filters are increased
- Use of BN-ReLU-Conv instead of Conv-BN-ReLU while adding residual connections.
- Optionally, Dropout may be used in between Conv layers within residual blocks.

**WRN-28-10**, `n = 28`, `k = 10`
- number of residual blocks = `(n - 6) / 4` = `4` blocks
- number of filters =
    - conv1: `16`
    - conv2: `16 x 10 = 160`
    - conv3:`32 x 10 = 320`
    - conv4: `64 x 10 = 640`

**WRN-40-2**, `n = 40`, `k = 2`
- number of residual blocks = `(n - 6) / 4` = `(28 - 6) / 4` = `4` blocks
- number of filters =
    - conv1: `16`
    - conv2: `16 x 2 = 32`
    - conv3:`32 x 2 = 64`
    - conv4: `64 x 2 = 128`

