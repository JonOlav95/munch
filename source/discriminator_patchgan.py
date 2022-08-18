import tensorflow as tf
import tensorflow_addons as tfa

from config import FLAGS


def downsample(x, filters, kernel_size=4, stride=2, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same",
                               kernel_initializer=initializer, use_bias=False)(x)

    if apply_batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.LeakyReLU()(x)

    return x


def discriminator_patchgan(img_size=FLAGS["img_size"]):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_target = tf.keras.layers.Input(shape=img_size)
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.concatenate([input_img, input_target])
    x = downsample(x, filters=64, kernel_size=8, apply_batchnorm=False)

    for filters in [128, 256, 256, 256]:
        x = downsample(x, filters)

    x = tf.keras.layers.ZeroPadding2D()(x)
    x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    x = tf.keras.layers.ZeroPadding2D()(x)
    outputs = tf.keras.layers.Conv2D(1, 4, strides=1, activation="sigmoid", kernel_initializer=initializer)(x)

    model = tf.keras.Model(inputs=[input_img, input_target], outputs=outputs)
    return model
