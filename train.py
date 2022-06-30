import tensorflow as tf
import statistics
import time

from data_handler import load_data
from gated_generator import gated_generator
from patch_discriminator import *
from loss_func import generator_loss, discriminator_loss
from loss_logger import make_log, log_loss
from config import FLAGS
from mask import create_mask, mask_image_batch
from plotter import plot_one

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    generator = gated_generator(FLAGS.get("img_size"))
    disc = discriminator(FLAGS.get("img_size"))

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=disc)


@tf.function
def train_step(x, y, mask):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator([x, mask], training=True)

        stage1_gen = gen_output[0]
        stage2_gen = gen_output[1]

        disc_real_output = disc([y, mask], training=True)
        disc_generated_output = disc([stage2_gen, mask], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, stage1_gen, stage2_gen, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    return gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss


def distributed_step_fn(groundtruth_batch):
    if FLAGS["num_gpus"] > 1:
        groundtruth_batch = groundtruth_batch.values
        groundtruth_batch = groundtruth_batch[0]

    if len(groundtruth_batch) < FLAGS["replica_batch_size"]:
        return

    masked_batch, masks = mask_image_batch(groundtruth_batch, n=len(groundtruth_batch))
    gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss = strategy.run(train_step, args=(
        masked_batch, groundtruth_batch, masks,
    ))

    return gen_gan_loss.numpy(), gen_l1_loss.numpy(), disc_real_loss.numpy(), disc_gen_loss.numpy()


def train():
    ds = load_data(FLAGS["training_samples"])
    ds = strategy.experimental_distribute_dataset(ds)
    epochs = FLAGS["max_iters"]

    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    filename = None
    if FLAGS["logging"]:
        filename = make_log()

    for i in range(epochs):

        loss_arr = []
        start = time.time()
        for (losses) in map(distributed_step_fn, ds):
            loss_arr.append(losses)

        print("Epoch: {}\nGEN GAN Loss: {}\nL1 Loss: {}\nDISC Real Loss: {}\nDISC Gen Loss: {}"
              .format(i,
                      statistics.mean(loss_arr[0]),
                      statistics.mean(loss_arr[1]),
                      statistics.mean(loss_arr[2]),
                      statistics.mean(loss_arr[3])),
              flush=True)
        print(f'Time taken for epoch {i:d}: {time.time() - start:.2f} sec', flush=True)

        if (i + 1) % FLAGS["checkpoint_nsave"] == 0 & FLAGS["checkpoint_save"]:
            checkpoint.save(file_prefix=FLAGS["checkpoint_prefix"])

        if FLAGS["plotting"]:
            plot_one(ds, disc, generator)
