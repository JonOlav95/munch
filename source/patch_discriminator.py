import tensorflow as tf
import tensorflow_addons as tfa

from config import FLAGS


def downsample(x, filters, kernel_size=2, stride=2):
    # initializer = tf.random_normal_initializer(0., 0.02)

    x = tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters=filters,
                                                                kernel_size=kernel_size,
                                                                strides=stride,
                                                                padding="same",
                                                                use_bias=False))(x)

    x = tf.keras.layers.LeakyReLU()(x)

    return x


def discriminator(img_size):
    input_missing = tf.keras.layers.Input(shape=img_size)
    input_complete = tf.keras.layers.Input(shape=img_size)

    x = tf.keras.layers.concatenate([input_missing, input_complete])

    for filters in [64, 128, 256, 256, 256, 256]:
        x = downsample(x, filters)

    x = tf.keras.layers.Flatten()(x)

    model = tf.keras.Model(inputs=[input_missing, input_complete], outputs=x)
    # model.summary()
    return model
