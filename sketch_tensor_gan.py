import tensorflow as tf

from config import FLAGS
from contextual_attention import contextual_attention
from model_operations import gated_conv, dilated_residual_block, resize_mask_like, gated_deconv


def st_generator(img_size):
    input_img = tf.keras.layers.Input(shape=img_size, batch_size=FLAGS["batch_size"])
    input_mask = tf.keras.layers.Input(shape=img_size[:2] + [1], batch_size=FLAGS["batch_size"])

    x = tf.keras.layers.concatenate([input_img, input_mask])
    one_mask = input_mask[0, ...]
    one_mask = tf.expand_dims(one_mask, axis=0)

    x = gated_conv(x, filters=64, ksize=7, stride=2)
    x = gated_conv(x, filters=128, ksize=4, stride=2)
    x = gated_conv(x, filters=256, ksize=4, stride=1)

    x = dilated_residual_block(x, filters=256, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)

    mask_small = resize_mask_like(one_mask, x)
    x, b = contextual_attention(x, x, mask_small, ksize=3, stride=1, rate=2)

    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)
    x = dilated_residual_block(x, filters=512, ksize=4, stride=1)

    x = gated_conv(x, filters=256, ksize=4, stride=1)
    x = gated_deconv(x, filters=128)
    x = gated_deconv(x, filters=64)
    x = gated_conv(x, filters=img_size[2], ksize=7, stride=1, activation=None)

    model = tf.keras.Model(inputs=[input_img, input_mask], outputs=x)
    model.summary()

    return model
