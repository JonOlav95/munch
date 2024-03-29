import time
from config import FLAGS
from global_variables import *
from train_utility import store_loss, end_epoch, train_step


def distributed_step_fn(batch):
    """Additional functionality required before the default train step.

    Args:
        batch: One batch from the dataset.

    Returns:
        The losses as a list.
    """
    batch = batch.values[0]

    if len(batch) < FLAGS["replica_batch_size"]:
        return None

    gr_batch = batch[:, 0, ...]
    masked_batch = batch[:, 1, ...]
    mask_batch = batch[:, 2, ...]

    gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss = strategy.run(train_step, args=(
        gr_batch, masked_batch, mask_batch
    ))

    gen_gan_loss = gen_gan_loss.values[0]
    gen_l1_loss = gen_l1_loss.values[0]
    disc_real_loss = disc_real_loss.values[0]
    disc_gen_loss = disc_gen_loss.values[0]

    return [gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss]


def train_multi_gpu():
    """Train on multiple GPUs

    This is set up to train on multiple GPUs on the same node, not for
    a homogeneous cluster on multiple nodes.
    https://keras.io/guides/distributed_training/
    """
    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    for i in range(FLAGS["max_iters"]):

        loss_arr = []
        start = time.time()
        for (losses) in map(distributed_step_fn, ds):
            if losses:
                store_loss(loss_arr, losses)

        end_epoch(i, loss_arr, start)