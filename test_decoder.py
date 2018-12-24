import torch
from vae_model import Decoder
import matplotlib.pyplot as plt

latent_size = 2

decoder_param = torch.load('decoder.pth')
decoder = Decoder(latent_size)
decoder.load_state_dict(decoder_param)
decoder.eval()

code_sampler = torch.distributions.MultivariateNormal(
    torch.zeros(latent_size),
    torch.eye(latent_size)
)

for i in range(100):
    z = code_sampler.sample(torch.Size([1]))
    mu_x = decoder(z)
    mu_x_np = mu_x.detach().numpy()
    plt.imshow(mu_x_np[0, 0, :, :])
    plt.show()