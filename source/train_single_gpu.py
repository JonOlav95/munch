import tensorflow as tf
import statistics
import numpy as np
import time

from data_handler import load_data
from generator_gated import gated_generator
from generator_standard import generator_standard
from patch_discriminator import *
from loss_logger import make_log, log_loss
from config import FLAGS
from mask import create_mask, mask_image_batch
from plotter import plot_one


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)


def discriminator_loss(disc_real_output, disc_generated_output):


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


def generator_loss(disc_generated_output, gen_out, target):
    l1_loss = 0
    gan_loss = 0

    if FLAGS["l1_loss"]:
        l1_loss = FLAGS["l1_lambda"] * tf.reduce_mean(tf.abs(target - gen_out))
    if FLAGS["disc_loss"]:
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    total_gen_loss = gan_loss + l1_loss

    return total_gen_loss, gan_loss, l1_loss


@tf.function
def train_step(y, x, generator, disc, g_optimizer, d_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(x, training=True)
        gen_output = gen_output[2]

        disc_real_output = disc([x, y], training=True)
        disc_generated_output = disc([x, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, disc.trainable_variables)

    g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    d_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    return gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss


def train_single_gpu():
    ds = load_data(FLAGS["training_samples"])
    epochs = FLAGS["max_iters"]

    generator = generator_standard(FLAGS.get("img_size"))
    disc = discriminator(FLAGS.get("img_size"))

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=disc)

    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    for i in range(epochs):

        loss_arr = []
        start = time.time()

        for batch in ds:

            gr_batch = batch[:, 0, ...]
            masked_batch = batch[:, 1, ...]

            gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss = train_step(gr_batch, masked_batch, generator,
                                                                                  disc, generator_optimizer,
                                                                                  discriminator_optimizer)

            gen_gan_loss = gen_gan_loss.numpy()
            gen_l1_loss = gen_l1_loss.numpy()
            disc_real_loss = disc_real_loss.numpy()
            disc_gen_loss = disc_gen_loss.numpy()
            loss_arr.append([gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss])

        loss_arr = np.asarray(loss_arr)
        print("Epoch: {}\nGEN GAN Loss: {}\nL1 Loss: {}\nDISC Real Loss: {}\nDISC Gen Loss: {}"
              .format(i,
                      np.mean(loss_arr[:, 0]),
                      np.mean(loss_arr[:, 1]),
                      np.mean(loss_arr[:, 2]),
                      np.mean(loss_arr[:, 3])),
              flush=True)

        print(f'Time taken for epoch {i:d}: {time.time() - start:.2f} sec', flush=True)

        if (i + 1) % FLAGS["checkpoint_nsave"] == 0 & FLAGS["checkpoint_save"]:
            checkpoint.save(file_prefix=FLAGS["checkpoint_prefix"])

        if FLAGS["plotting"]:
            plot_one(ds, disc, generator)
