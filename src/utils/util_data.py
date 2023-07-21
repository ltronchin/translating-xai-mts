import numpy as np
import os
import random
import pandas as pd

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
    def __init__(self, data_dir, data, mode, decide_backend='.npy'):
        """ Initialization """
        self.data_dir = data_dir
        self.data = data
        self.mode = mode
        self.decide_backend = decide_backend

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.data)

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Select sample
        row = self.data.iloc[index]
        id_car = row.name
        y = util_general.class_to_idx_generali(row.label)

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
