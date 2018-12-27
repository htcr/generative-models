import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

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


class Encoder_Big(Module):
    def __init__(self, latent_size):
        super(Encoder_Big, self).__init__()
        self.input_size = 28**2
        self.hidden_size = 300
        self.hidden_size_2 = 100
        self.latent_size = latent_size

        self.fc1 = nn.Linear(
            in_features=self.input_size, 
            out_features=self.hidden_size
        )

        self.fc2 = nn.Linear(
            in_features=self.hidden_size, 
            out_features=self.hidden_size_2
        )

        self.fc_mu = nn.Linear(
            in_features=self.hidden_size_2, 
            out_features=self.latent_size
        )
        self.fc_sigma = nn.Linear(
            in_features=self.hidden_size_2,
            out_features=self.latent_size
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        sigma = F.relu(self.fc_sigma(x))
        return mu, sigma


class Decoder_Big(Module):
    def __init__(self, latent_size):
        super(Decoder_Big, self).__init__()
        # force p(x|z) be of unit variance
        self.latent_size = latent_size
        self.hidden_size_1 = 100
        self.hidden_size = 300
        self.img_edge = 28
        self.reconstruct_size = self.img_edge**2

        self.fc0 = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_size_1
        )
        self.fc1 = nn.Linear(
            in_features=self.hidden_size_1,
            out_features=self.hidden_size
        )
        self.fc2 = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.reconstruct_size
        )

    def forward(self, x):
        # x.shape should be (batch_size*sample_num, latent_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        mu = self.fc2(x)
        mu = mu.view(x.shape[0], 1, self.img_edge, self.img_edge)
        return mu

class Encoder_Conv(Module):
    def __init__(self, latent_size):
        super(Encoder_Conv, self).__init__()
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(1, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.fc1   = nn.Linear(16*4*4, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc_mu   = nn.Linear(64, self.latent_size)
        self.fc_sigma   = nn.Linear(64, self.latent_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        mu = self.fc_mu(out)
        sigma = F.relu(self.fc_sigma(out))

        return mu, sigma


class Decoder_Conv(Module):
    def __init__(self, latent_size):
        super(Decoder_Conv, self).__init__()
        # force p(x|z) be of unit variance
        self.latent_size = latent_size

        self.fc3 = nn.Linear(self.latent_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(128, 256)
        self.conv2 = nn.ConvTranspose2d(16, 8, 5)
        self.conv1 = nn.ConvTranspose2d(8, 1, 5)

    def forward(self, x):
        # x.shape should be (batch_size*sample_num, latent_size)
        out = F.relu(self.fc3(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc1(out))
        out = out.view(out.size(0), 16, 4, 4)
        out = F.upsample(out, scale_factor=2)
        out = self.conv2(out)
        out = F.upsample(out, scale_factor=2)
        mu = self.conv1(out)
        return mu