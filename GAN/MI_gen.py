
import numpy as np
import torch
import time
import random
import os
from preprocess import import_data
from GAN import wgan

for subiiii in [29,30]:
    sub_index = subiiii

    gen_model = "WGAN"


    seed_n = np.random.randint(500)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)


    data, csp_data, label, Cov, Dis_mean, Dis_std, P, B, Wb = import_data(sub_index)


    data = np.expand_dims(np.transpose(data, (0, 2, 1)), axis=1)
    csp_data = np.expand_dims(np.transpose(csp_data, (0, 2, 1)), axis=1)

    gen_time = 0
    save_dir = "generated_data_2"
    os.makedirs(save_dir, exist_ok=True)

    for cls in range(2):

        print(f"\n====== Training Class {cls} Generator ======")
        idx = np.where(label == cls)[0]
        data_cls = data[idx]
        csp_cls = csp_data[idx]
        label_cls = label[idx]
        if len(label_cls) == 0:
            print(f"kind no {cls} skip")
            continue

        start = time.time()
        gen_data = wgan(data_cls, csp_cls, label_cls, cls, seed_n, sub_index,
                        Cov[:, :, cls], Dis_mean[:, cls], Dis_std[:, cls], P[:, :, cls], B[:, :, cls], Wb)
        gen_time += time.time() - start

        np.save(os.path.join(save_dir, f"S{sub_index}_class{cls}_19.npy"), gen_data)

    print(f"\n Generation complete! Total time: {gen_time:.2f} seconds")
