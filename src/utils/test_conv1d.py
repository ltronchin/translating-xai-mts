import torch

# Conv1d expects inputs of shape [batch, channels, features] (where features can be some timesteps and can vary)
data = torch.randn(64, 3, 2490)  # [batch, channels, timesteps]

enc1 = torch.nn.Conv1d(in_channels=3, out_channels=8, kernel_size=7, stride=2)
enc2 = torch.nn.Conv1d(in_channels=8, out_channels=16, kernel_size=7, stride=2)
enc3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=2)
enc4 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=2)
enc5 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=2)
enc6 = torch.nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=2)

data1 = enc1(data)
print(data1.shape)
data2 = enc2(data1)
print(data2.shape)
data3 = enc3(data2)
print(data3.shape)
data4 = enc4(data3)
print(data4.shape)
data5 = enc5(data4)
print(data5.shape)
data6 = enc6(data5)
print(data6.shape)

# layerflatten = torch.nn.Flatten(start_dim=1)
# dataflatten = layerflatten(data6)
# print(dataflatten.shape) # torch.Size([64, 9920])

##############
# DECODER
data_dec = torch.randn(1, 32, 34)  # [batch, channels, timesteps]

dec1 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=128, kernel_size=7, stride=2, output_padding=1)
dec2 = torch.nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=7, stride=2, output_padding=1)
dec3 = torch.nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=7, stride=2, output_padding=1)
dec4 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=7, stride=2, output_padding=1)
dec5 = torch.nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=7, stride=2, output_padding=1)
dec6 = torch.nn.ConvTranspose1d(in_channels=8, out_channels=3, kernel_size=7, stride=2, output_padding=1)

data1 = dec1(data6)
print(data1.shape)
data2 = dec2(data1)
print(data2.shape)
data3 = dec3(data2)
print(data3.shape)
data4 = dec4(data3)
print(data4.shape)
data5 = dec5(data4)
print(data5.shape)
data6 = dec6(data5)
print(data6.shape)
