import os
import numpy as np

def merge_npy_files(input_dir, output_file):
    all_data = []

    for fname in os.listdir(input_dir):
        if fname.endswith(".npy"):
            fpath = os.path.join(input_dir, fname)
            data = np.load(fpath)
            all_data.append(data)
            print(f"→ load {fname}, shape = {data.shape}")

    # hug all
    merged = np.concatenate(all_data, axis=0)
    np.save(output_file, merged)
    print(f"hug at：{output_file}, whole shape = {merged.shape}")

if __name__ == "__main__":
    # input
    input_dirs = [
        "s class 1",
        "s class 0"

    ]

    # out put
    output_root = "merged_EC_3"
    os.makedirs(output_root, exist_ok=True)

    for folder in input_dirs:
        input_path = folder
        output_path = os.path.join(output_root, folder + ".npy")
        merge_npy_files(input_path, output_path)

    print("hug finished")
