"""
preprocess for csp data
import data and split for three kinds of experiments
the input data has been processed using MATLAB
sample size: 22*1000
frequency band: 4-40 Hz
"""


import scipy.io
import numpy as np
import matplotlib.pyplot as plt


# get the data from .m files
# standardize

def import_data(sub_index, datatype):
    data, label = [], []

    path = 'path'

    tmp = scipy.io.loadmat(path + str(sub_index) + datatype + '.mat')
    data_one_subject = tmp['csp_data']
    # data_one_subject = tmp['data']
    data = np.transpose(data_one_subject, (2, 1, 0))
    # data.append(data_one_subject)

    label_one_subject = tmp['label']
    label = label_one_subject.T[0]
    # label.append(label_one_subject)

    return data, label  # (288, 22, 1000) (288,)


# to get the mixed data (subject-independent data, 100 subject-specific data and augmented data)
# used for tri data
def aug_data(sub_index, standardize=True):
    t_data, t_label = import_data(sub_index, datatype='T')
    e_data, e_label = import_data(sub_index, datatype='E')

    # shuffle the train data
    num = np.random.permutation(len(t_data))

    data_train = t_data[num, :, :]
    label_train = t_label[num]

    data_test = e_data
    label_test = e_label

    num = np.random.permutation(len(data_train))
    data_train = data_train[num, :, :]
    label_train = label_train[num]

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    # standardize to [-1, 1]
    '''
    k = 2 / (np.max(data_train) - np.min(data_train))
    d_tr = -1 + k * (data_train - np.min(data_train))
    d_te = -1 + k * (data_test - np.min(data_train))
    '''

    # change to 0-3 for train
    label_train -= 1
    label_test -= 1

    return data_train, label_train, data_test, label_test


# one subject: split the data of one subject to train set and val set (actually, T and E)
def split_subject(sub_index, standardize=True):
    t_data, t_label = import_data(sub_index, datatype='T')
    e_data, e_label = import_data(sub_index, datatype='E')

    # ------
    # to get the source data of 8 subjects and the W
    path = 'path'
    tmp = scipy.io.loadmat(path + str(sub_index) + 'T' + '.mat')
    ss_data = tmp['data']
    ss_data = np.transpose(ss_data, (2, 1, 0))

    source_label = tmp['label']
    source_label = source_label.T[0]

    aug_number = 750   # the number of augmentation data

    path = 'path'
    tmp = scipy.io.loadmat(path + str(sub_index) + 'T' + '.mat')
    Wb = tmp['Wb']
    source_data = np.zeros([ss_data.shape[0], 16, 1000])
    augment_data = np.zeros([aug_number*4, 16, 1000])
    aug_data = []
    aug_label = []
    # for S1_class0.npy
    for cla_index in range(4):
        path = 'your path' + str(sub_index) + '_class' + str(cla_index+1) + '.npy'

        aug_data_oneclass = np.load(path)
        aug_label_oneclass = np.zeros([aug_number])
        aug_label_oneclass[:] = cla_index + 1
        aug_data.append(aug_data_oneclass[0:aug_number, :, :])
        aug_label.append(aug_label_oneclass)
    aug_label = np.concatenate(aug_label)
    aug_data = np.concatenate(aug_data)
    aug_data = np.squeeze(aug_data)


    for a_in in range(aug_data.shape[0]):
        augment_data[a_in, :, :] = Wb.transpose().dot(aug_data[a_in, :, :])

    for s_in in range(ss_data.shape[0]):
        source_data[s_in, :, :] = Wb.transpose().dot(ss_data[s_in, :, :])

    t_data = np.concatenate((t_data, source_data, augment_data), axis=0)
    t_label = np.concatenate((t_label, source_label, aug_label), axis=0)
    # t_data = np.concatenate((source_data, augment_data), axis=0)
    # t_label = np.concatenate((source_label, aug_label), axis=0)
    # ------


    # shuffle the train data
    num = np.random.permutation(len(t_data))

    data_train = t_data[num, :, :]
    label_train = t_label[num]

    data_test = e_data
    label_test = e_label



    num = np.random.permutation(len(data_train))
    data_train = data_train[num, :, :]
    label_train = label_train[num]
    # -------------------------------------------------------------

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    # standardize to [-1, 1]
    '''
    k = 2 / (np.max(data_train) - np.min(data_train))
    d_tr = -1 + k * (data_train - np.min(data_train))
    d_te = -1 + k * (data_test - np.min(data_train))
    '''


    # change to 0-3 for train
    label_train -= 1
    label_test -= 1

    return data_train, label_train, data_test, label_test


# half cross subject: all T as the training data, one E as the test data
def split_half(sub_index, standardize=True):
    data_train = []
    label_train = []
    for i in range(9):
        t_data, t_label = import_data(i+1, datatype='T')
        data_train.append(t_data)
        label_train.append(t_label)

    data_train = np.concatenate((data_train[0], data_train[1], data_train[2], data_train[3], data_train[4],
                                data_train[5], data_train[6], data_train[7], data_train[8]))
    label_train = np.concatenate((label_train[0], label_train[1], label_train[2], label_train[3], label_train[4],
                                 label_train[5], label_train[6], label_train[7], label_train[8]))

    num = np.random.permutation(len(data_train))
    data_train = data_train[num, :, :]
    label_train = label_train[num]

    data_test, label_test = import_data(sub_index, datatype='E')



    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    label_train -= 1
    label_test -= 1

    return data_train, label_train, data_test, label_test


# cross subject: 8 T data of 8 subjects as the training data, one E data of another subject as the test
def split_cross(sub_index, standardize=True):
    data_train = []
    label_train = []
    for i in range(9):
        if i != sub_index-1:
            t_data, t_label = import_data(i+1, datatype='T')
            data_train.append(t_data)
            label_train.append(t_label)

    data_train = np.concatenate((data_train[0], data_train[1], data_train[2], data_train[3],
                                 data_train[4], data_train[5], data_train[6], data_train[7]))
    label_train = np.concatenate((label_train[0], label_train[1], label_train[2], label_train[3],
                                  label_train[4], label_train[5], label_train[6], label_train[7]))

    num = np.random.permutation(len(data_train))
    data_train = data_train[num, :, :]
    label_train = label_train[num]

    data_test, label_test = import_data(sub_index, datatype='E')

    data_train = np.transpose(data_train, (0, 2, 1))
    data_test = np.transpose(data_test, (0, 2, 1))

    label_train -= 1
    label_test -= 1

    return data_train, label_train, data_test, label_test


def main():
    split_cross()


if __name__ == "__main__":
    main()
