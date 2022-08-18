import time
import numpy as np
import tensorflow as tf

from config import FLAGS
from loss_functions import generator_loss, discriminator_loss
from global_variables import *
from plotter import plot_one


@tf.function
def train_step(y, x, mask):
    """One train step on a batch from the dataset.

    Args:
        y: The groundtruth.
        x: The masked image.
        mask: The mask.

    Returns:
        The different losses.
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(x, training=True)
        gen_output = gen_output[2]

        disc_real_output = discriminator([x, y], training=True)
        disc_generated_output = discriminator([x, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss


def store_loss(loss_arr, losses):
    """Store the loss as numpy variables in an list.

    Args:
        loss_arr: The list which holds the loss.
        losses: The current epochs losses.
    """
    gen_gan_loss = losses[0].numpy()
    gen_l1_loss = losses[1].numpy()
    disc_real_loss = losses[2].numpy()
    disc_gen_loss = losses[3].numpy()
    loss_arr.append([gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss])


def end_epoch(epoch, loss_arr, start):
    """Info and storage at the end of an epoch, same for single and multi GPU.

    Args:
        epoch: The current epoch number.
        loss_arr: Loss across all epochs.
        start: The start time.
    """
    loss_arr = np.asarray(loss_arr)

    print("Epoch: {}\nGEN GAN Loss: {}\nL1 Loss: {}\nDISC Real Loss: {}\nDISC Gen Loss: {}"
          .format(epoch,
                  np.mean(loss_arr[:, 0]),
                  np.mean(loss_arr[:, 1]),
                  np.mean(loss_arr[:, 2]),
                  np.mean(loss_arr[:, 3])),
          flush=True)

    print(f'Time taken for epoch {epoch:d}: {time.time() - start:.2f} sec', flush=True)

    if (epoch + 1) % FLAGS["checkpoint_nsave"] == 0 & FLAGS["checkpoint_save"]:
        checkpoint.save(file_prefix=FLAGS["checkpoint_prefix"])

    if FLAGS["plotting"]:
        plot_one(ds, generator)
