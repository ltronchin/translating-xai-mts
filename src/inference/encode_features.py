import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import yaml
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import tensorflow as tf
from sklearn.metrics import confusion_matrix, average_precision_score

from src.utils import util_general
from src.utils import util_data
from src.utils import util_model

from src.utils import util_autoencoder


debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise\n")

print("Upload configuration file")
if debug == 'develop':
    with open('./configs/cae.yaml') as file:
       cfg = yaml.load(file, Loader=yaml.FullLoader)
    worker = cfg['device']['worker']
    device_type = cfg['device']['device_type']
else:
    args = util_general.get_args_cae_generali()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    worker = args.gpu
    device_type = args.device_type

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
print("Parameters")
model_name = cfg['model']['model_name']
modes = list(cfg['data']['modes'].keys())
steps = ['train', 'valid', 'test']
classes = cfg['data']['classes']

# Device
if device_type == "cpu":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device("cpu")
else:
    device = torch.device(worker if torch.cuda.is_available() else "cpu")
num_workers = 0 if device.type == "cpu" else cfg['device']['gpu_num_workers']
print(f'device: {device}')

# Files and Directories
print("Files and directories")
fold_dir = os.path.join(cfg['data']['fold_dir'])
data_dir = os.path.join(cfg['data']['data_dir'])
interim_dir = os.path.join(cfg['data']['interim_dir'])
util_general.create_dir(interim_dir)
# Model dir
model_dir = os.path.join(cfg['model']['model_dir'])
util_general.create_dir(model_dir)
# Report dir
reports_dir = os.path.join(cfg['reports']['reports_dir']) # folder to save results
util_general.create_dir(reports_dir)
plot_training_dir = os.path.join(reports_dir, "training_plot")
util_general.create_dir(plot_training_dir)
general_reports_dir = os.path.join(reports_dir, "general")
util_general.create_dir(general_reports_dir)

# Load architectures
print("Build encoder and decoder architecture and load weights")
encoder = util_autoencoder.Encoder()
decoder = util_autoencoder.Decoder()

# Upload pretrained models
print("Load pretrained cnn")
encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pt"), map_location=device))
decoder.load_state_dict(torch.load(os.path.join(model_dir, "decoder.pt"), map_location=device))

encoder.to(device)
decoder.to(device)

# Data
print("Create dataloader")
fold_data = {step: pd.read_csv(os.path.join(fold_dir, '%s.txt' % step), delimiter=" ", index_col=0)  for step in steps}
datasets_acc = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[0], step), data=fold_data[step], mode=modes[0], decide_backend='.torch') for step in steps}
datasets_pos = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[1], step), data=fold_data[step], mode=modes[1], decide_backend='.torch') for step in steps}
datasets = {step: util_data.MultimodalDataset(dataset_mode_1=datasets_acc[step], dataset_mode_2=datasets_pos[step]) for step in steps}

# Data loaders
data_loaders = {
    'train': torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True, drop_last=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)
}

# Extract features and save results
features = util_autoencoder.encode_features(encoder=encoder, dataloader=data_loaders['valid'], device=device)
# Save results
features = pd.DataFrame.from_dict(features, orient='index')
features.to_csv(os.path.join(interim_dir, "embedded_features.csv"))