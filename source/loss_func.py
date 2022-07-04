import tensorflow as tf
from config import FLAGS

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)


def discriminator_loss(disc_real_output, disc_generated_output):
    disc_real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    disc_gen_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    disc_real_loss /= FLAGS["replica_batch_size"]
    disc_gen_loss /= FLAGS["replica_batch_size"]

    total_disc_loss = disc_real_loss + disc_gen_loss

    return total_disc_loss, disc_real_loss, disc_gen_loss


def generator_loss(disc_generated_output, gen_out, target):
    l1_loss = 0
    gan_loss = 0

    if FLAGS["l1_loss"]:
        l1_loss = FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - gen_out))
    if FLAGS["disc_loss"]:
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    gan_loss /= FLAGS["replica_batch_size"]
    l1_loss /= FLAGS["replica_batch_size"]

    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss

