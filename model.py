import tensorflow as tf


def upsample(x, filters, kernel_size=2, stride=2):
    x = tf.keras.layers.Conv2DTranspose(filters, kernel_size, stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def downsample(x, filters, kernel_size=2, stride=2):
    x = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def get_model(img_size):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size)

    inputs = tf.keras.layers.Concatenate(axis=3)([input_img, input_mask])

    x = tf.keras.layers.Conv2D(32, 1, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    for filters in [64, 128, 256, 512, 512, 512, 512]:
        x = downsample(x, filters)

    for filters in [512, 512, 512, 512, 256, 128, 64]:
        x = upsample(x, filters)

    outputs = tf.keras.layers.Conv2DTranspose(1, 2, 2, padding="same", activation="tanh")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def discriminator(img_size):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size)

    inputs = tf.keras.layers.Concatenate(axis=3)([input_img, input_mask])

    x = tf.keras.layers.Conv2D(64, 1, strides=2, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    for filters in [128, 256, 256, 256, 256]:
        x = downsample(x, filters)

    outputs = tf.keras.layers.Conv2D(1, 4, strides=1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model



