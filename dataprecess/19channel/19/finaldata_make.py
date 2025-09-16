import os
import numpy as np


in_dir = 'final_dataset_gen'
out_file = 'MDD3.npy'

all_data = {}


for fname in os.listdir(in_dir):
    if fname.endswith('.npy'):
        key = fname.replace('.npy', '')
        ke = key.replace('MDDH_','')
        path = os.path.join(in_dir, fname)
        print(f"[read] {fname}")
        all_data[ke] = np.load(path, allow_pickle=True)

np.save(out_file, all_data)
print(f"\nsave as {out_file}")
