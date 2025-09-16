import os
import numpy as np

def generate_labels(data_path, label_path, label_value):
    data = np.load(data_path)
    num_samples = data.shape[0]
    labels = np.full((num_samples,), fill_value=label_value, dtype=np.int64)
    np.save(label_path, labels)
    print(f"→ save：{label_path}，label={label_value}，number={num_samples}")

if __name__ == "__main__":
    # dir
    data_dir = "merged_EC_3"

    for fname in os.listdir(data_dir):
        if fname.endswith(".npy") and not fname.endswith("_label.npy"):
            data_path = os.path.join(data_dir, fname)
            label_path = os.path.join(data_dir, fname.replace(".npy", "_label.npy"))

            if fname.startswith("s class 0"):
                label_value = 0
            elif fname.startswith("s class 1"):
                label_value = 1
            else:
                print(f"[skip] failed recognition：{fname}")
                continue

            generate_labels(data_path, label_path, label_value)

    print("finish ！")
