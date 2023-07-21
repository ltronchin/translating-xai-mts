import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder architecture (input 3x2490)
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=8, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):  # Define the forward pass
        x = self.encoder_cnn(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional section
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=128, kernel_size=7, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=7, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=7, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=7, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=7, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=3, kernel_size=7, stride=2, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_conv(x)
        return x


# Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()

    train_loss = []
    i = 0
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for X, _, _ in tqdm(dataloader):  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        if i >= len(dataloader.dataset):
            break
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero.

        X = X.to(device) # Move tensor to the proper device

        encoded_data = encoder(X)  # Encode data
        decoded_data = decoder(encoded_data)  # Decode data

        loss = loss_fn(decoded_data, X)  # Evaluate loss
        loss.backward()  # Backward pass (Computes the gradient)
        optimizer.step()  # Update the weights

        train_loss.append(loss.detach().cpu().numpy())
        i += 1

    return np.mean(train_loss), encoder, decoder

# Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()

    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        i = 0
        for X, _, _ in tqdm(dataloader):
            if i>=len(dataloader.dataset):
                break
            # Move tensor to the proper device
            X = X.to(device)

            encoded_data = encoder(X)  # Encode data
            decoded_data = decoder(encoded_data)  # Decode data

            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(X.cpu())
            i += 1

        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)

        val_loss = loss_fn(conc_out, conc_label)  # Evaluate global loss
    return val_loss.data, encoder, decoder

def plot_ae_outputs(val_dataset, encoder, decoder, device, general_reports_dir, n_img=3):
    plt.figure()
    for i in range(n_img):

        ax = plt.subplot(2, n_img, i + 1)
        x = val_dataset[i][0].unsqueeze(0)
        x = x.to(device)

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            rec_img = decoder(encoder(x))

        x = torch.transpose(x, dim0=1, dim1=2)
        rec_img = torch.transpose(rec_img, dim0=1, dim1=2)
        plt.plot(x.cpu().squeeze().numpy(), linewidth=0.7)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n_img // 2:
            ax.set_title('Original images')

        ax = plt.subplot(2, n_img, i + 1 + n_img)
        plt.plot(rec_img.cpu().squeeze().numpy(), linewidth=0.7)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n_img // 2:
            ax.set_title('Reconstructed images')

    plt.savefig(os.path.join(general_reports_dir, "img_loss.png"), dpi=400, format='png')
    plt.show()

def encode_features(encoder, dataloader, device):
    # Results
    results = {}
    # Loop
    encoder.eval()
    with torch.no_grad():
        for inputs, _, _, file_names in tqdm(dataloader):
            inputs = inputs.to(device)
            # Prediction
            outputs = encoder(inputs.float())
            outputs = outputs.view([outputs.shape[0], outputs.shape[1] * outputs.shape[2]])
            for file_name, output in zip(file_names, outputs):
                results[file_name] = output.detach().cpu().numpy()
    return results


def decode_features(decoder, dataloader, device):
    # Results
    results = {}
    # Loop
    decoder.eval()
    with torch.no_grad():
        for inputs, _, _, file_names in tqdm(dataloader):
            inputs = inputs.to(device)
            # Prediction
            outputs = decoder(inputs.float())
            for file_name, outputs in zip(file_names, outputs):
                results[file_name] = (outputs.detach().cpu().numpy())
    return results