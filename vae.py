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
import datetime
from vae_model import Encoder, Decoder

data_root = './data'
if not os.path.exists(data_root):
    os.makedirs(data_root)

exp_name = 'baseline'
exp_time = str(datetime.datetime.now())
exp_record_name = '_'.join(['exp', exp_name, exp_time])
exp_record_dir = './exps'
exp_record_path = os.path.join(exp_record_dir, exp_record_name)
if not os.path.exists(exp_record_path):
    os.makedirs(exp_record_path)

train_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

train_dataset = torchvision.datasets.MNIST(
    root=data_root, train=True, transform=train_data_transform, 
    download=True
)

train_batch_size = 100
latent_size = 2

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=train_batch_size, 
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sample_num = 1
encoder, decoder = Encoder(latent_size), Decoder(latent_size)
encoder, decoder = encoder.to(device), decoder.to(device)


params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)

epsilon_sampler = torch.distributions.MultivariateNormal(
    torch.zeros(latent_size).to(device), 
    torch.eye(latent_size).to(device)
)

encoder.train()
decoder.train()

max_epochs = 5
eps = 1e-4

sample_seen = 0
loss_record_path = os.path.join(exp_record_path, 'loss.txt')
kl_loss_record_path = os.path.join(exp_record_path, 'kl_loss.txt')
rec_loss_record_path = os.path.join(exp_record_path, 'rec_loss.txt')

f_loss = open(loss_record_path, 'a+')
f_kl_loss = open(kl_loss_record_path, 'a+')
f_rec_loss = open(rec_loss_record_path, 'a+')

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

        batch_rec_loss, batch_kl_loss = rec_loss.item(), kl_loss.item()
        record_rec_loss, record_kl_loss = batch_rec_loss / train_batch_size, batch_kl_loss / train_batch_size
        record_loss = record_rec_loss + record_kl_loss
        sample_seen += train_batch_size
        f_rec_loss.write('{} {}\n'.format(sample_seen, record_rec_loss))
        f_kl_loss.write('{} {}\n'.format(sample_seen, record_kl_loss))
        f_loss.write('{} {}\n'.format(sample_seen, record_loss))

        mean_rec_loss += batch_rec_loss
        mean_kl_loss += batch_kl_loss

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

f_loss.close()
f_kl_loss.close()
f_rec_loss.close()


# save decoder weights
decoder_params = decoder.state_dict()
decoder_save_path = os.path.join(exp_record_path, 'decoder.pth')
torch.save(decoder_params, decoder_save_path)

