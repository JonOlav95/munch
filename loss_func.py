import tensorflow as tf
from config import FLAGS

hinge = tf.keras.losses.Hinge()

def discriminator_loss(disc_real_output, disc_generated_output):
    disc_real_loss = hinge(tf.ones_like(disc_real_output), disc_real_output)
    disc_gen_loss = hinge(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = disc_real_loss + disc_gen_loss

    return total_disc_loss, disc_real_loss, disc_gen_loss


def generator_loss(disc_generated_output, gen_output, target):
    l1_loss = 0
    gan_loss = 0

    if FLAGS["l1_loss"]:
        l1_loss = FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - gen_output))
    if FLAGS["disc_loss"]:
        gan_loss = hinge(tf.ones_like(disc_generated_output), disc_generated_output)

    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss

