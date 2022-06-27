import tensorflow as tf
import tensorflow_addons as tfa


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
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size[:2] + [1])

    x = tf.keras.layers.concatenate([input_img, input_mask])

    for filters in [64, 128, 256, 256, 256, 256]:
        x = downsample(x, filters)

    x = tf.keras.layers.Flatten()(x)

    model = tf.keras.Model(inputs=[input_img, input_mask], outputs=x)
    # model.summary()
    return model
