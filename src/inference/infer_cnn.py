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


print("Upload configuration file")
with open('./configs/cnn.yaml') as file:
   cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
print("Parameters")
worker = cfg['device']['worker']
model_name = cfg['model']['model_name']
modes = list(cfg['data']['modes'].keys())
steps = ['train', 'valid', 'test']
classes = cfg['data']['classes']
thr_cnn = cfg['trainer']['thr_cnn']

# Device
if cfg['device']['device_type'] == "cpu":
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

# Upload pretrained models
print("Load pretrained model")
model = util_model.build_model(
    model_dir=model_dir, model_name=model_name, samples_acc=cfg['data']['modes']['acc']['timestamp'], samples_pos=cfg['data']['modes']['pos']['timestamp']
)

# Data
print("Create dataloader")
fold_data = {step: pd.read_csv(os.path.join(fold_dir, '%s.txt' % step), delimiter=" ", index_col=0)  for step in steps}
datasets_acc = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[0], step), data=fold_data[step]) for step in steps}
datasets_pos = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[1], step), data=fold_data[step]) for step in steps}
datasets = {step: util_data.MultimodalDataset(dataset_mode_1=datasets_acc[step], dataset_mode_2=datasets_pos[step]) for step in steps}

# Data loaders
data_loaders = {
    'train': torch.utils.data.DataLoader(datasets['train'], batch_size=64, shuffle=True, drop_last=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)
}

i = 0
running_pred, running_y = [], []
for X_acc, X_pos, y, file_names in tqdm(data_loaders['valid']):
    if i >= len(data_loaders['valid'].dataset):
        break
    X_acc = tf.convert_to_tensor(X_acc, dtype=tf.float32)
    X_pos = tf.convert_to_tensor(X_pos, dtype=tf.float32)

    pred = model.predict([X_acc, X_pos])
    running_pred.append(pred)
    running_y.append(y.detach().numpy())
    i += 1

average_precision = average_precision_score(np.concatenate(running_y).ravel(), np.concatenate(running_pred).ravel())
print(f"Average precision: {average_precision}")

print("May the force be with you")