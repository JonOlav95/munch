import tensorflow as tf
from config import FLAGS

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss_single(disc_real_output, disc_generated_output):


    disc_real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    #shape = disc_real_loss.shape
    #current_batch_size = shape[0]
    #num_elements = shape.num_elements()
    #num_elements = num_elements * (FLAGS["global_batch_size"] / current_batch_size)
    #tmp_2 = tf.reduce_sum(disc_real_loss) * (1. / num_elements)
#
    disc_gen_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = disc_real_loss + disc_gen_loss
    return total_disc_loss, disc_real_loss, disc_gen_loss


def generator_loss_single(disc_generated_output, gen_out, target):
    l1_loss = 0
    gan_loss = 0

    if FLAGS["l1_loss"]:
        l1_loss = FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - gen_out))
    if FLAGS["disc_loss"]:
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss
