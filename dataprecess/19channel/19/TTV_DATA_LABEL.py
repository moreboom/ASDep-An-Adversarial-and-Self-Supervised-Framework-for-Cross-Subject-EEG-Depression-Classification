import os
import numpy as np

def merge_labels_and_save(label_files, out_path):
    all_labels = []
    for fname in label_files:
        labels = np.load(fname)
        print(f"[load label] {fname}, shape={labels.shape}")
        all_labels.append(labels)
    merged_labels = np.concatenate(all_labels, axis=0)
    np.save(out_path, merged_labels)
    print(f"→ save label to  {out_path}，shape={merged_labels.shape}")

if __name__ == "__main__":
    base_dir = "merged_EC_2"       # save npy after combines
    out_dir = "final_dataset_gen"    # final output
    os.makedirs(out_dir, exist_ok=True)

    # each label document
    train_label_files = [
        os.path.join(base_dir, "H_S1~24_EC_label.npy"),
        os.path.join(base_dir, "MDD_S1~27_EC_label.npy"),
        os.path.join(base_dir, "s class 0_label.npy"),
        os.path.join(base_dir, "s class 1_label.npy")

    ]
    test_label_files = [
        os.path.join(base_dir, "H_S25~27_EC_label.npy"),
        os.path.join(base_dir, "MDD_S28~30_EC_label.npy")
    ]
    val_label_files = [
        os.path.join(base_dir, "H_S28~30_EC_label.npy"),
        os.path.join(base_dir, "MDD_S31~34_EC_label.npy")
    ]

    # cooperate
    merge_labels_and_save(train_label_files, os.path.join(out_dir, "train_label.npy"))
    merge_labels_and_save(test_label_files, os.path.join(out_dir, "test_label.npy"))
    merge_labels_and_save(val_label_files, os.path.join(out_dir, "val_label.npy"))
