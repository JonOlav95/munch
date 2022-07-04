import time
import numpy as np

from config import FLAGS
from plotter import plot_one


def store_loss(loss_arr, losses):
    gen_gan_loss = losses[0].numpy()
    gen_l1_loss = losses[1].numpy()
    disc_real_loss = losses[2].numpy()
    disc_gen_loss = losses[3].numpy()
    loss_arr.append([gen_gan_loss, gen_l1_loss, disc_real_loss, disc_gen_loss])
    return


def end_epoch(epoch, loss_arr, start, checkpoint, ds, generator):
    loss_arr = np.asarray(loss_arr)

    print("Epoch: {}\nGEN GAN Loss: {}\nL1 Loss: {}\nDISC Real Loss: {}\nDISC Gen Loss: {}"
          .format(epoch,
                  np.mean(loss_arr[:, 0]),
                  np.mean(loss_arr[:, 1]),
                  np.mean(loss_arr[:, 2]),
                  np.mean(loss_arr[:, 3])),
          flush=True)

    print(f'Time taken for epoch {epoch:d}: {time.time() - start:.2f} sec', flush=True)

    if (epoch + 1) % FLAGS["checkpoint_nsave"] == 0 & FLAGS["checkpoint_save"]:
        checkpoint.save(file_prefix=FLAGS["checkpoint_prefix"])

    if FLAGS["plotting"]:
        plot_one(ds, generator)
