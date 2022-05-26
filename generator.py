import tensorflow as tf

from contextual_attention import contextual_attention


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


def gated_deconv(x, filters, name='upsample', padding='SAME', training=True):
    """Define deconv for generator.
    The deconv is defined to be a x2 resize_nearest_neighbor operation with
    additional gen_conv operation.
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


def st_generator(img_size):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size)

    x = tf.keras.layers.concatenate([input_img, input_mask])

    x = gated_conv(x, filters=64, ksize=7, stride=2)
    x = gated_conv(x, filters=128, ksize=4, stride=2)
    x = gated_conv(x, filters=256, ksize=4, stride=1)

    x = dilated_residual_block(x, filters=256, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)

    #TODO: Contextual attention
    #x = contextual_attention(x, x, input_mask, 3, 1, rate=2)

    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)

    x = gated_conv(x, filters=256, ksize=4, stride=1)
    x = gated_deconv(x, filters=128)
    x = gated_deconv(x, filters=64)
    x = gated_conv(x, filters=1, ksize=7, stride=1, activation=None)

    model = tf.keras.Model(inputs=[input_img, input_mask], outputs=x)
    model.summary()

    return model


def gated_generator(img_size):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size)

    x = tf.keras.layers.concatenate([input_img, input_mask])

    initializer = tf.random_normal_initializer(0., 0.02)

    filters = 48
    x = gated_conv(x, filters, 5, 1, name='conv1')
    x = gated_conv(x, 2 * filters, 3, 2, name='conv2_downsample')
    x = gated_conv(x, 2 * filters, 3, 1, name='conv3')
    x = gated_conv(x, 4 * filters, 3, 2, name='conv4_downsample')
    x = gated_conv(x, 4 * filters, 3, 1, name='conv5')
    x = gated_conv(x, 4 * filters, 3, 1, name='conv6')
    x = gated_conv(x, 4 * filters, 3, rate=2, name='conv7_atrous')
    x = gated_conv(x, 4 * filters, 3, rate=4, name='conv8_atrous')
    x = gated_conv(x, 4 * filters, 3, rate=8, name='conv9_atrous')
    x = gated_conv(x, 4 * filters, 3, rate=16, name='conv10_atrous')
    x = gated_conv(x, 4 * filters, 3, 1, name='conv11')
    x = gated_conv(x, 4 * filters, 3, 1, name='conv12')
    x = gated_deconv(x, 2 * filters, name='conv13_upsample')
    x = gated_conv(x, 2 * filters, 3, 1, name='conv14')
    x = gated_deconv(x, filters, name='conv15_upsample')
    x = gated_conv(x, filters // 2, 3, 1, name='conv16')
    x = gated_conv(x, 1, 3, 1, activation=None, name='conv17')
    outputs = tf.keras.activations.tanh(x)

    model = tf.keras.Model(inputs=[input_img, input_mask], outputs=outputs)
    model.summary()
    return model