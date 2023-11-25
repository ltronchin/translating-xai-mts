import numpy as np
import os
import os.path
import random
import pandas as pd
import copy
import aeon
from aeon.datasets import load_from_tsfile

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from src.utils import util_general

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def loader(path, mode, decide_backend):
    x = np.load(path)
    if decide_backend == '.npy':
        x = x.astype(np.float32)
    else:
        x = torch.tensor(x, dtype=torch.float32) # To Tensor
        if mode == 'acc':
            x = torch.transpose(x, dim0=0, dim1=1)
        else:
            pass
    return x

def load_as_df(data):
    pass

class DatasetGeneraliZeros(torch.utils.data.Dataset):
    """ Characterizes a dataset for PyTorch Dataloader to train images """
    def __init__(self, data, mode, decide_backend='.npy'):
        """ Initialization """
        self.data = data
        self.decide_backend = decide_backend
        if mode == 'acc':
            self.shape = (2490, 3)
        else:
            self.shape = (41, 1)

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.data)

    def __getitem__(self, index):
        """ Generates one sample of data """
        if self.decide_backend == '.npy':
            x = np.zeros(shape=self.shape, dtype=np.float32)
        else:
            x = torch.zeros(self.shape ,dtype=torch.float32)

        y = np.nan
        id_car = np.nan

        return x, y, id_car


class DatasetGenerali(torch.utils.data.Dataset):
    """ Characterizes a dataset for PyTorch Dataloader to train images """
    def __init__(self, data_dir, data, mode, class_to_idx=None, decide_backend='.npy'):
        """ Initialization """
        if class_to_idx is None:
            class_to_idx = {'crash': 1, 'non_crash': 0}
        self.data_dir = data_dir
        self.data = data
        self.mode = mode
        self.decide_backend = decide_backend
        self.class_to_idx = class_to_idx

    def idx_to_class_generali(self, class_id):
        if class_id == 1:
            return 'crash'
        elif class_id == 0:
            return 'non_crash'

    def class_to_idx_generali(self, class_name):
        if class_name == 'crash':
            return 1
        elif class_name == 'non_crash':
            return 0

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.data)

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Select sample
        row = self.data.iloc[index]
        id_car = row.name
        y = self.class_to_idx_generali(row.label)

        # Load data and get label
        path = os.path.join(self.data_dir, id_car+'.npy')
        x = loader(path=path, mode=self.mode, decide_backend=self.decide_backend)

        return x, y, id_car

class MultimodalDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataset_mode_1, dataset_mode_2):
        'Initialization'
        self.dataset_mode_1 = dataset_mode_1
        self.dataset_mode_2 = dataset_mode_2

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.dataset_mode_1)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x1, label1, idx1 = self.dataset_mode_1[index]
        x2, label2, idx2 = self.dataset_mode_2[index]
        if label1 is not np.nan:
            return x1, x2, label1, idx1
        else:
            return x1, x2, label2, idx2

# basic motions
def transform_labels(y_train, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] ---> [0,1,2]


    Parameters
    ----------
    y_train: array
        Labels of the train set

    y_test: array
        Labels of the test set


    Returns
    -------
    new_y_train: array
        Transformed y_train array

    new_y_test: array
        Transformed y_test array
    """

    # Initiate the encoder
    encoder = LabelEncoder()

    # Concatenate train and test to fit
    y_train_test = np.concatenate((y_train, y_test), axis=0)

    # Fit the encoder
    encoder.fit(y_train_test.ravel())

    # Transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_train_test.ravel())

    # Resplit the train and test
    new_y_train = new_y_train_test[0 : len(y_train)]
    new_y_test = new_y_train_test[len(y_train) :]

    return new_y_train, new_y_test


def import_data(data_dir, dataset_name, class_to_idx, randomize=False):
    """
    Load and preprocess train and test sets


    Parameters
    ----------
    data_dir: string
        Path to the data directory
    dataset_name: string
        Name of the dataset
    class_to_idx: dict
        Dictionary with the classes and the corresponding labels

    Returns
    -------
    X_train: array
        Train set without labels
    y_train: array
        Labels of the train set encoded
    X_test: array
        Test set without labels
    y_test: array
        Labels of the test set encoded
    y_train_nonencoded: array
        Labels of the train set non-encoded
    y_test_nonencoded: array
        Labels of the test set non-encoded
    """

    # Load train and test sets
    #X_train = np.load(basedir+ "/X_train.npy")
    #y_train = np.load(basedir + "/y_train.npy")
    #X_test = np.load(basedir + "/X_test.npy")
    #y_test = np.load(basedir + "/y_test.npy")
    # Transform to continuous labels
    #y_train, y_test = transform_labels(y_train, y_test)

    # Check this page for tutorial -> https://github.com/aeon-toolkit/aeon/blob/main/examples/datasets/data_loading.ipynb
    X_train, y_train = load_from_tsfile(os.path.join(data_dir, f'{dataset_name}_TRAIN.ts'))
    X_test, y_test = load_from_tsfile(os.path.join(data_dir, f'{dataset_name}_TEST.ts'))
    y_train = np.asarray([class_to_idx[y] for y in y_train])
    y_test =  np.asarray([class_to_idx[y] for y in y_test])

    if randomize:
        import random
        indexes_train = list(range(len(X_train)))
        indexes_test = list(range(len(X_test)))
        random.shuffle(indexes_train)
        random.shuffle(indexes_test)
        X_train = np.asarray([X_train[i] for i in indexes_train], dtype=np.float32)
        y_train = np.asarray([y_train[i] for i in indexes_train], dtype=np.float32)
        X_test = np.asarray([X_test[i] for i in indexes_test],  dtype=np.int32)
        y_test = np.asarray([y_test[i] for i in indexes_test], dtype=np.int32)

    y_train_nonencoded = copy.deepcopy(y_train)
    y_test_nonencoded = copy.deepcopy(y_test)

    # One hot encoding of the labels
    enc = OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # Reshape data to match 2D convolution filters input shape
    X_train = np.reshape(
        np.array(X_train),
        (X_train.shape[0], X_train.shape[2], X_train.shape[1], 1),
        order="C",
    )
    X_test = np.reshape(
        np.array(X_test),
        (X_test.shape[0], X_test.shape[2], X_test.shape[1], 1),
        order="C",
    )

    return X_train, y_train, X_test, y_test, y_train_nonencoded, y_test_nonencoded

