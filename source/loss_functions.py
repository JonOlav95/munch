import tensorflow as tf
from config import FLAGS

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)


def sum_over_batch_size(loss):
    """Function used to replicate sum over batch size.

    Reduction must be set to NONE when working with multiple GPUs, this function
    makes sure the loss is correct when working with either a single GPU or
    multiple.

    Args:
        loss: The current loss.

    Returns:
        The loss with respect to the batch size.
    """
    shape = loss.shape
    current_batch_size = shape[0]
    num_elements = shape.num_elements()
    num_elements = num_elements * (FLAGS["global_batch_size"] / current_batch_size)
    sum_over_bs_loss = tf.reduce_sum(loss) * (1. / num_elements)

    return sum_over_bs_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    disc_real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    disc_gen_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    disc_real_loss = sum_over_batch_size(disc_real_loss)
    disc_gen_loss = sum_over_batch_size(disc_gen_loss)

    total_disc_loss = disc_real_loss + disc_gen_loss

    return total_disc_loss, disc_real_loss, disc_gen_loss


def generator_loss(disc_generated_output, gen_output, target):
    l1_loss = 0
    gan_loss = 0

    if FLAGS["l1_loss"]:
        l1_loss = FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - gen_output))
    if FLAGS["disc_loss"]:
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    gan_loss = sum_over_batch_size(gan_loss)
    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss
