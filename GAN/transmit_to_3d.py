
import numpy as np
data = np.load('generated_data/S10_class0.npy')  # (7530, 1, 6, 1280)
data = np.squeeze(data, axis=1)  # -> (7530, 6, 1280)
np.save('transdata/S10_class0_squeezed.npy', data)