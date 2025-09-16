# preprocess.py
"""
Preprocess EEG data from .mat files under CSP constraints for cross-validation.
"""

import scipy.io
import numpy as np
import os


def import_data(sub_index):
    path = 'for_gan_mat_19'
    file_path = os.path.join(path, f'H_S{sub_index}_EC_19.mat') #change H TO MDD
    mat = scipy.io.loadmat(file_path)

    data = np.transpose(mat['data'], (0, 2, 1))         # (trial, time, channel)
    csp_data = np.transpose(mat['csp_data'], (0, 2, 1)) # (trial, time, channel)
    label = mat['label'].squeeze()                  # label: shape (trial,), 0-based

    Cov = mat['Cov']
    Dis_mean = mat['Dis_mean']
    Dis_std = mat['Dis_std']
    P = mat['PP']
    B = mat['BB']
    Wb = mat['Wb']

    return data, csp_data, label, Cov, Dis_mean, Dis_std, P, B, Wb


def split_subject(sub_index, standardize=True):
    data, csp_data, label, Cov, Dis_mean, Dis_std, P, B, Wb = import_data(sub_index)

    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    split = int(0.7 * num_samples)
    train_idx, test_idx = indices[:split], indices[split:]

    data_train, data_test = data[train_idx], data[test_idx]
    csp_train, csp_test = csp_data[train_idx], csp_data[test_idx]
    label_train, label_test = label[train_idx], label[test_idx]

    if standardize:
        mean = data_train.mean(axis=0)
        std = data_train.std(axis=0)
        data_train = (data_train - mean) / std
        data_test = (data_test - mean) / std

    data_train = np.transpose(data_train, (0, 2, 1))     # (trial, channel, time)
    data_test = np.transpose(data_test, (0, 2, 1))
    csp_train = np.transpose(csp_train, (0, 2, 1))
    csp_test = np.transpose(csp_test, (0, 2, 1))

    return data_train, csp_train, label_train, data_test, csp_test, label_test, Cov, Dis_mean, Dis_std, P, B, Wb


def split_half(sub_index, standardize=True):
    data_train, label_train = [], []

    for i in range(1, 25):
        if i != sub_index:
            data_i, _, label_i, *_ = import_data(i)
            data_train.append(data_i)
            label_train.append(label_i)

    data_train = np.concatenate(data_train, axis=0)
    label_train = np.concatenate(label_train, axis=0)

    data_test, _, label_test, *_ = import_data(sub_index)

    if standardize:
        mean = data_train.mean(axis=0)
        std = data_train.std(axis=0)
        data_train = (data_train - mean) / std
        data_test = (data_test - mean) / std

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    return data_train, label_train, data_test, label_test


def split_cross(sub_index, standardize=True):
    data_train, label_train = [], []

    for i in range(1, 25):
        if i != sub_index:
            data_i, _, label_i, *_ = import_data(i)
            data_train.append(data_i)
            label_train.append(label_i)

    data_train = np.concatenate(data_train, axis=0)
    label_train = np.concatenate(label_train, axis=0)

    data_test, _, label_test, *_ = import_data(sub_index)

    if standardize:
        mean = data_train.mean(axis=0)
        std = data_train.std(axis=0)
        data_train = (data_train - mean) / std
        data_test = (data_test - mean) / std

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    return data_train, label_train, data_test, label_test
