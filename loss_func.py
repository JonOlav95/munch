import tensorflow as tf
from config import FLAGS

hinge = tf.keras.losses.Hinge(reduction=tf.keras.losses.Reduction.SUM)


def discriminator_loss(disc_real_output, disc_generated_output):
    disc_real_loss = hinge(tf.ones_like(disc_real_output), disc_real_output)
    disc_gen_loss = hinge(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = disc_real_loss + disc_gen_loss

    return total_disc_loss, disc_real_loss, disc_gen_loss


def generator_loss(disc_generated_output, stage1_gen, stage2_gen, target):
    l1_loss = 0
    gan_loss = 0

    if FLAGS["l1_loss"]:
        l1_loss = FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - stage1_gen))
        l1_loss += FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - stage2_gen))
    if FLAGS["disc_loss"]:
        gan_loss = hinge(tf.ones_like(disc_generated_output), disc_generated_output)

    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss

