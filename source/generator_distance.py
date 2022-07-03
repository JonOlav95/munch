import tensorflow as tf

from config import FLAGS
from contextual_attention import contextual_attention
from model_operations import gated_conv, dilated_residual_block, resize_mask_like, gated_deconv


def generator_distance(img_size):
    input_distance_image = tf.keras.layers.Input(shape=img_size)
    n_channeles = img_size[2]

    x = gated_conv(input_distance_image, filters=64, ksize=7, stride=2)
    x = gated_conv(x, filters=128, ksize=4, stride=2)
    x = gated_conv(x, filters=256, ksize=4, stride=1)

    x = dilated_residual_block(x, filters=256, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)

    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=256, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=128, ksize=4, stride=1)

    x = gated_conv(x, filters=256, ksize=4, stride=1)
    x = gated_deconv(x, filters=128)
    x = gated_deconv(x, filters=64)
    x = gated_conv(x, filters=n_channeles, ksize=4, stride=1, activation=None)
    x = tf.keras.activations.tanh(x)

    model = tf.keras.Model(inputs=input_distance_image, outputs=x)
    model.summary()
    return model
