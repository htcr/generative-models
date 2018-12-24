import pdb
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math

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

train_batch_size = 100
latent_size = 2

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, 
    shuffle=True)


class Encoder(Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.input_size = 28**2
        self.hidden_size = 300
        self.latent_size = latent_size

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
        sigma = F.relu(self.fc_sigma(x))
        return mu, sigma


class Decoder(Module):
    def __init__(self, latent_size):
        super(Decoder, self).__init__()
        # force p(x|z) be of unit variance
        self.latent_size = latent_size
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sample_num = 1
encoder, decoder = Encoder(latent_size), Decoder(latent_size)
encoder, decoder = encoder.to(device), decoder.to(device)


params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params)

epsilon_sampler = torch.distributions.MultivariateNormal(
    torch.zeros(latent_size).to(device), 
    torch.eye(latent_size).to(device)
)

encoder.train()
decoder.train()

max_epochs = 100
eps = 1e-4

for epoch in range(max_epochs):
    mean_rec_loss = 0
    mean_kl_loss = 0
    mean_loss = 0

    for data, label in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # get q(z|x)
        mu_z, sigma_z = encoder(data) # (batch_size, latent_size)
        # (batch_size, latent_size)
        epsilon = epsilon_sampler.sample(torch.Size([train_batch_size])) 
        # sample z ~ q(z|x)
        z = mu_z + epsilon * sigma_z # (batch_size, latent_size)

        # get p(x|z)
        mu_x = decoder(z)

        # get loss terms
        rec_loss = torch.sum((data - mu_x)**2)
        kl_loss = torch.sum(
            mu_z**2 + sigma_z**2 - 2*torch.log(sigma_z+eps)
        )

        mean_rec_loss += rec_loss.item()
        mean_kl_loss += kl_loss.item()

        if math.isnan(mean_kl_loss):
            print('nan encountered')
            pdb.set_trace()

        loss = (rec_loss + kl_loss) / train_batch_size
        # backprop
        loss.backward()
        optimizer.step()

    mean_rec_loss /= len(train_dataset)
    mean_kl_loss /= len(train_dataset)
    mean_loss = mean_rec_loss + mean_kl_loss

    print(
        'Epoch {}/{} Loss: {} Rec: {} KL: {}'.format(
            epoch, max_epochs, mean_loss, 
            mean_rec_loss, mean_kl_loss
        )
    )

# save decoder weights
decoder_params = decoder.state_dict()
torch.save(decoder_params, 'decoder.pth')