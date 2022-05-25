import tensorflow as tf


def gen_conv(x, cnum, ksize, stride=1, rate=1, name='conv',
             padding='SAME', activation=tf.keras.activations.elu, training=True):
    """Define conv for generator.
    Args:
        x: Input.
        cnum: Channel number.
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
    x = tf.keras.layers.Conv2D(cnum, ksize, stride, dilation_rate=rate,
        activation=None, padding=padding, name=name)(x)

    if cnum == 3 or activation is None:
        # conv for output
        return x
    x, y = tf.split(x, 2, 3)
    x = activation(x)
    y = tf.keras.activations.sigmoid(y)
    x = x * y
    return x


def gen_deconv(x, cnum, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.
    Args:
        x: Input.
        cnum: Channel number.
        name: Name of layers.
        training: If current graph is for training or inference, used for bn.
    Returns:
        tf.Tensor: output
    """
    dim = x.get_shape().as_list()
    x = tf.image.resize(x, [dim[1] * 2, dim[2] * 2], method='nearest')
    x = gen_conv(
        x, cnum, 3, 1, name=name+'_conv', padding=padding,
        training=training)
    return x


def generator(img_size):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size)

    x = tf.keras.layers.concatenate([input_img, input_mask])

    initializer = tf.random_normal_initializer(0., 0.02)

    cnum = 48
    x = gen_conv(x, cnum, 5, 1, name='conv1')
    x = gen_conv(x, 2 * cnum, 3, 2, name='conv2_downsample')
    x = gen_conv(x, 2 * cnum, 3, 1, name='conv3')
    x = gen_conv(x, 4 * cnum, 3, 2, name='conv4_downsample')
    x = gen_conv(x, 4 * cnum, 3, 1, name='conv5')
    x = gen_conv(x, 4 * cnum, 3, 1, name='conv6')
    x = gen_conv(x, 4 * cnum, 3, rate=2, name='conv7_atrous')
    x = gen_conv(x, 4 * cnum, 3, rate=4, name='conv8_atrous')
    x = gen_conv(x, 4 * cnum, 3, rate=8, name='conv9_atrous')
    x = gen_conv(x, 4 * cnum, 3, rate=16, name='conv10_atrous')
    x = gen_conv(x, 4 * cnum, 3, 1, name='conv11')
    x = gen_conv(x, 4 * cnum, 3, 1, name='conv12')
    x = gen_deconv(x, 2 * cnum, name='conv13_upsample')
    x = gen_conv(x, 2 * cnum, 3, 1, name='conv14')
    x = gen_deconv(x, cnum, name='conv15_upsample')
    x = gen_conv(x, cnum // 2, 3, 1, name='conv16')
    x = gen_conv(x, 1, 3, 1, activation=None, name='conv17')
    outputs = tf.keras.activations.tanh(x)

    model = tf.keras.Model(inputs=[input_img, input_mask], outputs=outputs)
    model.summary()
    return model
