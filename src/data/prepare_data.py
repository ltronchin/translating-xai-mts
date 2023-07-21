# Function to create folder splits at slices level with a fixed seed
import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import collections
import os
from tqdm import tqdm

import yaml
import scipy.io as sio
import numpy as np
import src.utils.util_general as util_general

# Configuration file
with open('./configs/prepare_data.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Seed Everything
util_general.seed_all(cfg['seed'])

# Files and Directories
data_source = cfg['data']['data_source']
data_fold = cfg['data']['data_fold']
data_dest =  cfg['data']['data_dest']
util_general.create_dir(data_fold)
modes = list(cfg['data']['modes'].keys())

# define split
splits = ['train', 'valid', 'test']

# Load label
label_split = collections.defaultdict(lambda: {})
for split in splits:
    with open(os.path.join(data_source, f'y_{split}.npy'), 'rb') as f:
        label_split[split] = np.load(f)

# Save label file for each split
print("Save label")
for split in splits:
    with open(os.path.join(data_fold, '%s.txt' % split), 'w') as file:
        file.write("id label\n")
        for id_signal, label in enumerate(tqdm(label_split[split])):
            label = "%s\n" % util_general.idx_to_class_generali(label)
            row = "C_%s %s" % (id_signal, label)
            file.write(row)

# Save data file for each split
print("Save data")
data_split = collections.defaultdict(
    lambda: collections.defaultdict(dict)
)
for mode in modes:
    for split in splits:
        with open(os.path.join(data_source, f'X_{mode}_{split}_scaled.npy'), 'rb') as f:
            data_split[mode][split] = np.load(f)

        print(f'{mode}, {split}')
        data_dest_split_mode = os.path.join(data_dest, mode, split)
        util_general.create_dir(data_dest_split_mode)
        for id_signal, x in enumerate(tqdm(data_split[mode][split])):
            np.save(file=os.path.join(data_dest_split_mode, "C_%s.npy" % id_signal), arr=x)

