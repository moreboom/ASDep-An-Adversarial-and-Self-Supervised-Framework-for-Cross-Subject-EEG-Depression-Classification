import numpy as np

data = np.load("MDD2.npy", allow_pickle=True).item()
print(data.keys())
