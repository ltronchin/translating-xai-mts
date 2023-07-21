"""
Script to train a convolutional autoencoder (CAE) on Time series dataset from Generali.
"""

import sys
print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(["./"])
import yaml
import os
import pandas as pd
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision

from src.utils import util_general
from src.utils import util_data
from src.utils import util_model
from src.utils import util_autoencoder
from src.utils import util_report

debug = 'develop'
debug = input(f"Enter '{debug}' to run a debug session press enter otherwise")

print("Upload configuration file")
if debug == 'develop':
    with open('./configs/cae.yaml') as file:
       cfg = yaml.load(file, Loader=yaml.FullLoader)
    worker = cfg['device']['worker']
    exp_name = cfg['exp_name']
    device_type = cfg['device']['device_type']
else:
    args = util_general.get_args_cae_generali()
    with open(args.cfg_file) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)
    worker = args.gpu
    exp_name = cfg['exp_name']
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
model_dir = os.path.join(cfg['model']['model_dir'], exp_name)
util_general.create_dir(model_dir)
# Report dir
reports_dir = os.path.join(cfg['reports']['reports_dir'], exp_name) # folder to save results
util_general.create_dir(reports_dir)
plot_training_dir = os.path.join(reports_dir, "training_plot")
util_general.create_dir(plot_training_dir)
general_reports_dir = os.path.join(reports_dir, "general")
util_general.create_dir(general_reports_dir)

# Data
print("Create dataloader")
fold_data = {step: pd.read_csv(os.path.join(fold_dir, '%s.txt' % step), delimiter=" ", index_col=0)  for step in steps}
datasets_acc = {step: util_data.DatasetGenerali(data_dir=os.path.join(data_dir, modes[0], step), data=fold_data[step], mode=modes[0], decide_backend='torch') for step in steps}

# Data loaders
data_loaders = {
    'train': torch.utils.data.DataLoader(datasets_acc['train'], batch_size=64, shuffle=True, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'valid': torch.utils.data.DataLoader(datasets_acc['valid'], batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker),
    'test': torch.utils.data.DataLoader(datasets_acc['test'], batch_size=64, shuffle=False, num_workers=num_workers, worker_init_fn=util_data.seed_worker)
}

# Architecture
print("Build architecture")
overall_history = []
elapsed_time_dict = {'fold': [], 'training_time': [], 'overall_time': 0.0}
# Initialize the two networks
encoder = util_autoencoder.Encoder()
decoder = util_autoencoder.Decoder()

if cfg['model_autoencoder']['pretrained']:
    try:
        encoder.load_state_dict(torch.load(os.path.join(cfg['model_autoencoder']['pretrained'], exp_name, "encoder.pt")))
        decoder.load_state_dict(torch.load(os.path.join(cfg['model_autoencoder']['pretrained'], exp_name, "decoder.pt")))
    except FileNotFoundError:
        pass

encoder.to(device)
decoder.to(device)

# Sanity encoder and decoder
if debug == 'develop':
    x_sanity_check = torch.randn(64, 3, 2490).to(device)
    print(summary(encoder, x_sanity_check[0].shape)) # (channel, timestamp)
    h_sanity_check= encoder(x_sanity_check)
    print(summary(decoder, h_sanity_check[0].shape))
    x_sanity_check_rec = decoder(h_sanity_check)
    assert x_sanity_check_rec.shape == x_sanity_check.shape

# Optimizer and loss
loss = torch.nn.MSELoss() # Define the loss function
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optim = torch.optim.Adam(params_to_optimize, lr=cfg['model_autoencoder']['trainer']['optimizer']['lr'],  weight_decay=1e-05) # Define an optimizer

# Training
print("Train model")
since = time.time()
util_general.notification_ifttt({"Training CAE START"})

writer_loss = SummaryWriter(general_reports_dir + "/logs/loss")
writer_rec = SummaryWriter(general_reports_dir + "/logs/rec")
valid_iterator = iter(data_loaders['valid'])
fixed_x = next(valid_iterator)[0]
fixed_x = fixed_x.to(device)

history = {'train_loss': [], 'val_loss': []}

for epoch in range(cfg['model_autoencoder']['trainer']['epochs']):
    print(f"Epoch: {epoch}")
    train_loss, encoder, decoder = util_autoencoder.train_epoch(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=data_loaders['train'],
        loss_fn=loss,
        optimizer=optim,
    )

    val_loss, encoder, decoder = util_autoencoder.test_epoch(
        encoder=encoder,
        decoder=decoder,
        device=device,
        dataloader=data_loaders['valid'],
        loss_fn=loss,
    )

    # Print Validation loss
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print('\nEPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, cfg['model_autoencoder']['trainer']['epochs'], train_loss, val_loss))

    # Print original and reconstructed signals
    util_autoencoder.plot_ae_outputs(
        val_dataset=datasets_acc['valid'], encoder=encoder, decoder=decoder, device=device, general_reports_dir=general_reports_dir, n_img=3)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # plot same fake images to tensorboard (no gradient needed)
        rec_img = decoder(encoder(fixed_x[0].unsqueeze(0)))
        util_report.plot_acc(general_reports_dir, fixed_x[0], info='original', channel=cfg['data']['modes']['acc']['channel'])
        util_report.plot_acc(general_reports_dir, rec_img[0], info='reconstructed', channel=cfg['data']['modes']['acc']['channel'])

    util_report.plot_training(history=history, plot_training_dir=plot_training_dir)

    writer_loss.add_scalar('train', train_loss.item(), global_step=epoch)
    writer_loss.add_scalar('val', val_loss.item(), global_step=epoch)

    # Save model
    torch.save(encoder.state_dict(), os.path.join(model_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(model_dir, "decoder.pt"))

time_elapsed = time.time() - since
util_general.notification_ifttt({f"Training CAE END: {time_elapsed}"})

print("May the force be with you.")
