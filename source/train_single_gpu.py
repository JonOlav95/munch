import tensorflow as tf
import statistics
import numpy as np
import time

from data_handler import load_data
from generator_gated import gated_generator
from generator_standard import generator_standard
from loss_functions import discriminator_loss, generator_loss, two_stage_generator_loss
from discriminator_patchgan import *
from config import FLAGS
from model_variables import *
from train_utility import store_loss, end_epoch, train_step


def train_single_gpu():
    if FLAGS["checkpoint_load"]:
        checkpoint.restore(tf.train.latest_checkpoint(FLAGS["checkpoint_dir"]))

    for i in range(FLAGS["max_iters"]):

        loss_arr = []
        start = time.time()

        for batch in ds:

            if len(batch) != FLAGS["global_batch_size"]:
                continue

            gr_batch = batch[:, 0, ...]
            masked_batch = batch[:, 1, ...]
            mask_batch = batch[:, 2, ..., 0]

            losses = train_step(gr_batch, masked_batch, mask_batch)

            store_loss(loss_arr, losses)

        end_epoch(i, loss_arr, start, checkpoint, ds, generator)
