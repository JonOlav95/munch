import tensorflow as tf

from test import test
from train import train
from config import FLAGS


if __name__ == "__main__":
    assert FLAGS["test"] or FLAGS["train"]

    if FLAGS["debug"]:
        tf.config.run_functions_eagerly(True)

    if FLAGS["train"]:
        train()
    elif FLAGS["test"]:
        test()
    else:
        print("No mode selected.")
