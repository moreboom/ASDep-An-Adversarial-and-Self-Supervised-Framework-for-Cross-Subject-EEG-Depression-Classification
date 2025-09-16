import os
import numpy as np

def merge_and_save(filenames, out_path):
    all_data = []
    for fname in filenames:
        data = np.load(fname)
        print(f"[load] {fname}, shape={data.shape}")
        all_data.append(data)
    merged = np.concatenate(all_data, axis=0)
    np.save(out_path, merged)
    print(f"→ save to  {out_path}，shape={merged.shape}")

if __name__ == "__main__":
    base_dir = "merged_EC_2"       # save data to hug
    out_dir = "final_dataset_gen"    # final dir
    os.makedirs(out_dir, exist_ok=True)


    train_files = [
        os.path.join(base_dir, "H_S1~24_EC.npy"),
        os.path.join(base_dir, "MDD_S1~27_EC.npy"),
        os.path.join(base_dir, "s class 0.npy"),
        os.path.join(base_dir, "s class 1.npy")
    ]
    test_files = [
        os.path.join(base_dir, "H_S25~27_EC.npy"),
        os.path.join(base_dir, "MDD_S28~30_EC.npy")
    ]
    val_files = [
        os.path.join(base_dir, "H_S28~30_EC.npy"),
        os.path.join(base_dir, "MDD_S31~34_EC.npy")
    ]

    merge_and_save(train_files, os.path.join(out_dir, "train_data.npy"))
    merge_and_save(test_files, os.path.join(out_dir, "test_data.npy"))
    merge_and_save(val_files, os.path.join(out_dir, "val_data.npy"))
