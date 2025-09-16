import os
import numpy as np

def merge_npy_files(file_list, out_path):
    all_data = []
    for f in file_list:
        arr = np.load(f)
        print(f"[load] {f}, shape={arr.shape}")
        all_data.append(arr)
    merged = np.concatenate(all_data, axis=0)
    np.save(out_path, merged)
    print(f"→ save {out_path}, shape={merged.shape}")

if __name__ == "__main__":
    base_dir = "final_dataset_gen"

    # collect path
    data_files = [
        os.path.join(base_dir, "train_data.npy"),
        os.path.join(base_dir, "test_data.npy"),
        os.path.join(base_dir, "val_data.npy")
    ]

    label_files = [
        os.path.join(base_dir, "train_label.npy"),
        os.path.join(base_dir, "test_label.npy"),
        os.path.join(base_dir, "val_label.npy")
    ]

    # collect and hug
    merge_npy_files(data_files, os.path.join(base_dir, "All_train_data.npy"))
    merge_npy_files(label_files, os.path.join(base_dir, "All_train_label.npy"))
