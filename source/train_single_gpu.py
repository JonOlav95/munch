import tensorflow as tf
import statistics
import numpy as np
import time

from data_handler import load_data
from generator_standard import generator_standard
from loss_single_gpu import discriminator_loss_single, generator_loss_single
from patch_discriminator import *
from config import FLAGS
from train_utility import store_loss, end_epoch


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


@tf.function
def train_step(y, x):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(x, training=True)
        gen_output = gen_output[2]

        disc_real_output = disc([x, y], training=True)
        disc_generated_output = disc([x, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_single(disc_generated_output, gen_output, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss_single(disc_real_output,
                                                                                   disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    return [gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss]


def train_single_gpu():
    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    for i in range(epochs):

        loss_arr = []
        start = time.time()

        for batch in ds:
            gr_batch = batch[:, 0, ...]
            masked_batch = batch[:, 1, ...]

            losses = train_step(gr_batch, masked_batch, generator,
                                disc, generator_optimizer,
                                discriminator_optimizer)

            store_loss(loss_arr, losses)

        end_epoch(i, loss_arr, start, checkpoint, ds, generator)
