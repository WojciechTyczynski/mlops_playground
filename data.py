import torch
from torch.utils.data import TensorDataset, DataLoader
from numpy import load
import numpy as np
import glob

def mnist():
    # exchange with the corrupted mnist dataset
    files_train = glob.glob('/Users/wojtek/DTU/mlops/dtu_mlops/data/corruptmnist/train*.npz')
    files_test = glob.glob('/Users/wojtek/DTU/mlops/dtu_mlops/data/corruptmnist/test*.npz')
    images_train = []
    labels_tain = []
    for file in files_train:
        data_tmp = load(file)
        images_train.append(data_tmp['images'])
        labels_tain.append(data_tmp['labels'])
    train_x = torch.from_numpy(np.concatenate(images_train)).float()
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[1])
    train_y = torch.from_numpy(np.concatenate(labels_tain))
    test_data = load(files_test[0])
    test_x = torch.from_numpy(test_data['images']).float()
    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[1])
    test_y = torch.from_numpy(test_data['labels'])
    trainDataset = TensorDataset(train_x, train_y)
    testDataset = TensorDataset(test_x, test_y)
    train = DataLoader(trainDataset, batch_size=64, shuffle=True)
    test = DataLoader(testDataset)
    return train, test
