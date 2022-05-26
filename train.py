import tensorflow as tf
import statistics

from data_handler import load_data
from generator import gated_generator, st_generator
from discriminator import *
from main import FLAGS, generator_optimizer, discriminator_optimizer
from plotter import plot_one

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(disc_real_output, disc_generated_output):
    disc_real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    disc_gen_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = disc_real_loss + disc_gen_loss

    return total_disc_loss, disc_real_loss, disc_gen_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (FLAGS["l1_lambda"] * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


@tf.function
def train_step(model, disc, x, mask, y):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = model([x, mask], training=True)

        disc_real_output = disc([y, mask], training=True)
        disc_generated_output = disc([gen_output, mask], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, model.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    return gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss


def train():
    model = st_generator(FLAGS.get("img_size"))
    disc = discriminator(FLAGS.get("img_size"))
    ds = load_data()
    epochs = FLAGS["max_iters"]

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=model,
                                     discriminator=disc)

    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    for i in range(epochs):

        loss_arr = []

        for batch in ds:
            x, mask, y = tf.split(value=batch, num_or_size_splits=3, axis=3)
            gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss = train_step(model, disc, x, mask, y)

            loss_arr.append((gen_gan_loss.numpy(), gen_l1_loss.numpy(), disc_real_loss.numpy(),
                             disc_gen_loss.numpy(),
                             disc_gen_loss.numpy()))

        print("Epoch: {}\nGEN GAN Loss: {}\nL1 Loss: {}\nDISC Real Loss: {}\nDISC Gen Loss: {}"
              .format(i,
                      statistics.mean(loss_arr[0]),
                      statistics.mean(loss_arr[1]),
                      statistics.mean(loss_arr[2]),
                      statistics.mean(loss_arr[3])))

        if (i + 1) % FLAGS["checkpoint_nsave"] == 0 & FLAGS["checkpoint_save"]:
            checkpoint.save(file_prefix=FLAGS["checkpoint_prefix"])

        if FLAGS["plotting"]:
            plot_one(ds, disc, model)
