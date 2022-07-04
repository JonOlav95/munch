import time

from data_handler import load_data
from generator_standard import generator_standard
from loss_multi_gpu import generator_loss_multi, discriminator_loss_multi
from patch_discriminator import *
from config import FLAGS
from train_utility import store_loss, end_epoch


strategy = tf.distribute.MirroredStrategy()

ds = load_data(FLAGS["training_samples"])
ds = strategy.experimental_distribute_dataset(ds)
epochs = FLAGS["max_iters"]

with strategy.scope():
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

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_multi(disc_generated_output, gen_output, y)
        total_disc_loss, disc_real_loss, disc_gen_loss = discriminator_loss_multi(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(total_disc_loss, disc.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    return gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss


def distributed_step_fn(batch):

    batch = batch.values[0]

    if len(batch) < FLAGS["replica_batch_size"]:
        return

    gr_batch = batch[:, 0, ...]
    masked_batch = batch[:, 1, ...]

    gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss = strategy.run(train_step, args=(
        gr_batch, masked_batch
    ))

    gen_gan_loss = gen_gan_loss.values[0]
    gen_l1_loss = gen_l1_loss.values[0]
    disc_real_loss = disc_real_loss.values[0]
    disc_gen_loss = disc_gen_loss.values[0]

    return [gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss]


def train_multi_gpu():
    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    for i in range(epochs):

        loss_arr = []
        start = time.time()
        for (losses) in map(distributed_step_fn, ds):
            store_loss(loss_arr, losses)

        end_epoch(i, loss_arr, start, checkpoint, ds, generator)