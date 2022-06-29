import tensorflow as tf


def gated_deconv(x, filters, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gated_conv operation.
    Args:
        x: Input.
        filters: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    dim = x.get_shape().as_list()
    x = tf.image.resize(x, [dim[1] * 2, dim[2] * 2], method='nearest')
    x = gated_conv(
        x, filters, 3, 1, padding=padding,
        training=training)
    return x


def gated_conv(x, filters, ksize, stride=1, rate=1, name='conv',
               padding='SAME', activation=tf.keras.activations.elu, training=True):
    """Define conv for generator.
    Args:
        x: Input.
        filters: Channel number.
        ksize: Kernel size.
        stride: Convolution stride.
        rate: Rate for or dilated conv.
        name: Name of layers.
        padding: Default to SYMMETRIC.
        activation: Activation function after convolution.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    assert padding in ['SYMMETRIC', 'SAME', 'REFELECT']
    if padding == 'SYMMETRIC' or padding == 'REFELECT':
        p = int(rate * (ksize - 1) / 2)
        x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode=padding)
        padding = 'VALID'
    x = tf.keras.layers.Conv2D(filters, ksize, stride, dilation_rate=rate,
                               activation=None, padding=padding)(x)

    if filters == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3)
    x = activation(x)
    y = tf.keras.activations.sigmoid(y)
    x = x * y
    return x


def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask
    """
    shape = x.get_shape().as_list()[1:3]
    mask_resize = tf.image.resize(mask, shape, method='nearest')
    return mask_resize


def dilated_residual_block(x, filters, ksize=4, stride=1, dilation_rate=2):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=ksize,
                               strides=stride,
                               padding="same",
                               dilation_rate=dilation_rate)(x)

    # TODO: Change to instance normalization (?)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x

