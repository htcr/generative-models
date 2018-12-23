import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

data_root = './data'
if not os.path.exists(data_root):
    os.makedirs(data_root)

train_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

train_dataset = torchvision.datasets.MNIST(
    root=data_root, train=True, transform=train_data_transform, 
    download=True)

train_batch_size = 4

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, 
    shuffle=True)


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_size = 28**2
        self.hidden_size = 300
        self.latent_size = 2

        self.fc1 = nn.Linear(
            in_features=self.input_size, 
            out_features=self.hidden_size
        )
        self.fc_mu = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.latent_size
        )
        self.fc_sigma = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.latent_size
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma


class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # force p(x|z) be of unit variance
        self.latent_size = 2
        self.hidden_size = 300
        self.img_edge = 28
        self.reconstruct_size = self.img_edge**2

        self.fc1 = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_size
        )
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.reconstruct_size
        )

    def forward(self, x):
        # x.shape should be (batch_size*sample_num, latent_size)
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        mu = mu.view(x.shape[0], 1, self.img_edge, self.img_edge)
        return mu


for data, label in train_loader:
    pass

    '''
    print('datatype', type(data))
    print('labeltype', type(label))
    print('datashape', data.shape)
    print('labelshape', label.shape)

    plt.imshow(data[0, 0, :, :])
    plt.show()
    '''