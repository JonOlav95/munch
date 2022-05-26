import csv
from datetime import datetime


def make_log():
    filename = str(datetime.now())
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow(["gen_gan_loss", "gen_l1_loss", "disc_real_loss", "disc_gen_loss"])


def log(filename, loss):
    with open(filename, "a") as f:
        writer = csv.writer(f)
        writer.writerow([str(loss[0]), str(loss[1]), str(loss[2]), str(loss[3])])
