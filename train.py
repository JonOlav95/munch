import tensorflow as tf
import statistics
import time

from data_handler import load_data
from gated_generator import gated_generator
from patch_discriminator import *
from loss_func import generator_loss, discriminator_loss
from loss_logger import make_log, log_loss
from config import FLAGS, generator_optimizer, discriminator_optimizer
from mask import create_mask, mask_batch, reiterate_mask
from plotter import plot_one


@tf.function
def train_step(model, disc, x, y, mask):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = model([x, mask], training=True)

        stage1_gen = gen_output[0]
        stage2_gen = gen_output[1]

        disc_real_output = disc([y, mask], training=True)
        disc_generated_output = disc([stage2_gen, mask], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, stage1_gen, stage2_gen, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, model.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    return gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss


def train():
    ds = load_data(FLAGS["training_samples"])
    generator = gated_generator(FLAGS.get("img_size"))
    #generator = st_generator(FLAGS.get("img_size"))
    disc = discriminator(FLAGS.get("img_size"))
    epochs = FLAGS["max_iters"]

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=disc)

    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    filename = None
    if FLAGS["logging"]:
        filename = make_log()

    for i in range(epochs):

        loss_arr = []
        start = time.time()

        for groundtruth_batch in ds:

            if len(groundtruth_batch) != FLAGS["batch_size"]:
                continue

            mask = create_mask(FLAGS["img_size"][:2])
            masked_batch = mask_batch(groundtruth_batch, mask)
            masks = reiterate_mask(mask, len(groundtruth_batch))

            gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss = train_step(generator, disc, masked_batch,
                                                                                  groundtruth_batch, masks)

            loss_arr.append((gen_gan_loss.numpy(),
                             gen_l1_loss.numpy(),
                             disc_real_loss.numpy(),
                             disc_gen_loss.numpy()))

        print(f'Time taken: {time.time() - start:.2f} sec\n', flush=True)
        print("Epoch: {}\nGEN GAN Loss: {}\nL1 Loss: {}\nDISC Real Loss: {}\nDISC Gen Loss: {}"
              .format(i,
                      statistics.mean(loss_arr[0]),
                      statistics.mean(loss_arr[1]),
                      statistics.mean(loss_arr[2]),
                      statistics.mean(loss_arr[3])),
              flush=True)

        if (i + 1) % FLAGS["checkpoint_nsave"] == 0 & FLAGS["checkpoint_save"]:
            checkpoint.save(file_prefix=FLAGS["checkpoint_prefix"])

        if FLAGS["plotting"]:
            plot_one(ds, disc, generator)

        if FLAGS["logging"]:
            log_loss(filename, loss_arr)
