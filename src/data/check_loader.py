import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import yaml
import os
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

import tensorflow as tf

from src.utils import util_general
from src.utils import util_data

print("Upload configuration file")
with open('./configs/cnn.yaml') as file:
   cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
print("Parameters")
modes = list(cfg['data']['modes'].keys())
steps = ['train', 'valid', 'test']

# Device
device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
print("Files and directories")
fold_dir = os.path.join(cfg['data']['fold_dir'])
data_dir = os.path.join(cfg['data']['data_dir'])

# Data
print("Create dataloader")
fold_data = {step: pd.read_csv(os.path.join(fold_dir, '%s.txt' % step), delimiter=" ", index_col=0)  for step in steps}
datasets_acc = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[0], step), data=fold_data[step], mode=modes[0]) for step in steps}
datasets_pos = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[1], step), data=fold_data[step], mode=modes[1]) for step in steps}
datasets = {step: util_data.MultimodalDataset(dataset_mode_1=datasets_acc[step], dataset_mode_2=datasets_pos[step]) for step in steps}

# Data loaders
data_loaders = {
    'train': torch.utils.data.DataLoader(datasets['train'], batch_size=1, shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=1, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)
}



for step in steps:
    print(step)
    n_crash = 0
    n_non_crash = 0
    for X_acc, X_pos, y, file_names in tqdm(data_loaders[step]):
        if y.numpy()[0] == 1:
            n_crash += 1
        else:
            n_non_crash += 1

    print('crash')
    print(n_crash)
    print('non crash')
    print(n_non_crash)

print("May be the force with you.")