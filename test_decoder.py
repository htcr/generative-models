import torch
from vae_model import Decoder
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

latent_size = 2
sample_per_dim = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder_param = torch.load('decoder.pth')
decoder = Decoder(latent_size)
decoder.load_state_dict(decoder_param)
decoder.to(device)
decoder.eval()

# build latent code grid
percentages = np.linspace(1.0/(sample_per_dim+1), 1.0, sample_per_dim, endpoint=False)
values = norm.ppf(percentages)
z0s, z1s = np.meshgrid(values, values)
z0s, z1s = z0s.reshape(-1), z1s.reshape(-1)
z = np.stack((z0s, z1s), axis=1) # (sample_num, 2)
z = torch.Tensor(z).to(device)

mu_x = decoder(z) # (sample_num, 1, 28, 28)
mu_x_np = mu_x.detach().cpu().numpy()

img_edge = 28
vis_edge = img_edge*sample_per_dim

manifold_vis = np.zeros(
    (vis_edge, vis_edge)
)

for i in range(sample_per_dim):
    for j in range(sample_per_dim):
        manifold_vis[i*img_edge:(i+1)*img_edge, j*img_edge:(j+1)*img_edge] = \
            mu_x_np[i*sample_per_dim+j, 0, :, :]

plt.imshow(manifold_vis)
plt.show()