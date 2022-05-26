import tensorflow as tf


def upsample(x, filters, kernel_size=2, stride=2):
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.Conv2DTranspose(filters=filters,
                                        kernel_size=kernel_size,
                                        strides=stride,
                                        padding="same",
                                        kernel_initializer=initializer,
                                        use_bias=False)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def downsample(x, filters, kernel_size=2, stride=2):
    initializer = tf.random_normal_initializer(0., 0.02)

    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=stride,
                               padding="same",
                               kernel_initializer=initializer,
                               use_bias=False)(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


def discriminator(img_size):
    input_img = tf.keras.layers.Input(shape=img_size)
    input_mask = tf.keras.layers.Input(shape=img_size)

    x = tf.keras.layers.concatenate([input_img, input_mask])

    x = tf.keras.layers.Conv2D(64, 1, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    for filters in [128, 256, 256, 256, 512]:
        x = downsample(x, filters)

    x = tf.keras.layers.ZeroPadding2D()(x)

    outputs = tf.keras.layers.Conv2D(1, 4, strides=1, activation="sigmoid", use_bias=False)(x)
    model = tf.keras.Model(inputs=[input_img, input_mask], outputs=outputs)
    #model.summary()
    return model
