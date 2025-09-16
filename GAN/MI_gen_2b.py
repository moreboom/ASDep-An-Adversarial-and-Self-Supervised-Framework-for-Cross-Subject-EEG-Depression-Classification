

from preprocess_2b import *
# from WGAN_GP_for_csp import wgan  # test the structure of discriminator
from GAN_2b import wgan  # discriminate the eeg and the csp meanwhile
# from WGAN_GP_for_csp_just_gen import wgan
import torch
import numpy as np
import time
import random
import argparse
import matplotlib.pyplot as plt
import torch.distributed as dist
import os

sub_index = 9
gen_model = "WGAN"


seed_n = np.random.randint(500)
print(seed_n)

random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)

gen_time = 0


data_train, csp_data_train, label_train, data_test, csp_data_test, label_test, Cov, Dis_mean, Dis_std, P, B, Wb \
    = split_subject(sub_index)

for nclass in range(0, 2):

    class_sample_index = np.argwhere(label_train == nclass)
    data_train_class = data_train[class_sample_index, :, :]
    csp_data_train_class = csp_data_train[class_sample_index, :, :]
    label_train_class = label_train[class_sample_index]

    data_train_class = np.squeeze(data_train_class)
    csp_data_train_class = np.squeeze(csp_data_train_class)
    data_train_class = data_train_class.swapaxes(1, 2)
    csp_data_train_class = csp_data_train_class.swapaxes(1, 2)
    label_train_class = np.squeeze(label_train_class)

    # generative model
    print("*********Training Generative Model*********")
    start = time.time()

    print(gen_model)
    data_train_class = np.expand_dims(data_train_class, axis=1)
    csp_data_train_class = np.expand_dims(csp_data_train_class, axis=1)
    gen_data = wgan(data_train_class, csp_data_train_class, label_train_class, nclass, seed_n, sub_index,
                    Cov[:, :, nclass], Dis_mean[:, nclass], Dis_std[:, nclass], P[:, :, nclass], B[:, :, nclass], Wb)

    gen_time = gen_time + (time.time() - start)

    # save generated data

    np.save("path"
            "S" + str(sub_index) + "_class" + str(nclass) + ".npy", gen_data)


print("time for generative model: %f" % gen_time)






