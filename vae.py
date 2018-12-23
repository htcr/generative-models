import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt

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

for data, label in train_loader:
    print('datatype', type(data))
    print('labeltype', type(label))
    print('datashape', data.shape)
    print('labelshape', label.shape)

    plt.imshow(data[0, 0, :, :])
    plt.show()