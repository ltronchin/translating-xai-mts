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
import pickle

import tensorflow as tf
from sklearn.metrics import confusion_matrix, average_precision_score

import dice_ml
from dice_ml.utils import helpers  # helper functions

from src.utils import util_general
from src.utils import util_data
from src.utils import util_model


print("Upload configuration file")
with open('./configs/counterfactual.yaml') as file:
   cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed everything
util_general.seed_all(cfg['seed'])

# Parameters
print("Parameters")
worker = cfg['device']['worker']
id_run = cfg['id_run']
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
reports_dir = os.path.join(cfg['reports']['reports_dir'], str(id_run)) # folder to save results
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
    'train': torch.utils.data.DataLoader(datasets['train'], batch_size=1, shuffle=True, drop_last=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=1, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=1, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)
}

# Create train and valid dataframe
print("Create dataframe for DiCE")
# Column names
features_acc_name = [f'{axis}_acc_{i}' for axis in ['x', 'y', 'z'] for i in range(2490)]
features_pos_name = [f'pos_{i}' for i in range(41)]
features_name = features_acc_name + features_pos_name + ['event']
for step in ['train', 'valid']:
    print(step)
    try:
        if step == 'train':
           with open(os.path.join(interim_dir, 'dict_train_min_max'), 'rb') as f:
               dict_train_min_max = pickle.load(f)
        else:
            df_valid =pd.read_csv(os.path.join(interim_dir, f'df_{step}.csv'), index_col=0)
    except FileNotFoundError:
        print('create it!')
        # Construct each row
        running_features = []
        i = 0
        for X_acc, X_pos, y, file_names in tqdm(data_loaders[step]):
            if i >= 100: # len(data_loaders[step].dataset):
                break
            X_acc = X_acc[0].numpy()
            X_pos = X_pos[0].numpy()

            features_acc = np.concatenate((X_acc[:, 0], X_acc[:, 1], X_acc[:, 2]))
            running_features.append(np.concatenate((features_acc, X_pos,  y.numpy())))
            i += 1
        if step == 'train':
            df_train = pd.DataFrame(running_features, columns = features_name)
            df_train.to_csv(os.path.join(interim_dir, 'df_train'))

            # Compute min max
            running_min_max = []
            for feature_name in tqdm(features_name[:-1]):
                running_min_max.append([np.min(df_train[feature_name]), np.max(df_train[feature_name])])
            dict_train_min_max = dict(zip(features_name[:-1], running_min_max))
            with open(os.path.join(interim_dir, 'dict_train_min_max'), 'wb') as f:
                pickle.dump(dict_train_min_max, f)

        else:
            df_valid = pd.DataFrame(running_features, columns = features_name)
            df_valid.to_csv(os.path.join(interim_dir, 'df_valid.csv'))

# Step 1: dice_ml.Data
print("Prepare data for DiCE")
d = dice_ml.Data(features=dict_train_min_max, type_and_precision={feature: ['float', 8] for feature in features_name}, outcome_name='event')
backend = 'TF'+tf.__version__[0]
# Step 2: dice_ml.Model
print("Prepare model for DiCE")
m = dice_ml.Model(model=model, backend=backend)
# Step 3: initiate DiCE
print("Create DiCE explainer")
exp = dice_ml.Dice(d, m)

# Provide a query instance
# Query instance in the form of a dictionary or a dataframe; keys: feature name, values: feature value
query_instance = df_valid.iloc[0][:-1].to_dict()
# generate counterfactuals
dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_range=None, desired_class="opposite", features_to_vary="all") # Query instance should be a dict, a pandas dataframe, a list, or a list of dicts
dice_exp.visualize_as_dataframe(show_only_changes=True)

imp = exp.local_feature_importance(query_instance, cf_examples_list=dice_exp.cf_examples_list)