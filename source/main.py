import os
import json
import tensorflow as tf

from test import test
from train import train
from config import FLAGS
from train_multi_gpu import train_multi_gpu
from train_single_gpu import train_single_gpu

if __name__ == "__main__":

    assert int(FLAGS["replica_batch_size"] * FLAGS["num_gpus"]) == FLAGS["global_batch_size"]

    if FLAGS["debug"]:
        tf.config.run_functions_eagerly(True)

    if FLAGS["train"]:
        if FLAGS["num_gpus"] == 1:
            train_single_gpu()
        else:
            train_multi_gpu()

    elif FLAGS["test"]:
        test()
    else:
        print("No mode selected.")
