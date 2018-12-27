import os
from test_decoder import vis_manifold
import matplotlib.pyplot as plt

def read_curve(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(' ') for line in lines]
    x = [int(line[0]) for line in lines]
    y = [float(line[1]) for line in lines]
    
    return x, y

def get_exp_data(exp_path):
    f_loss_path = os.path.join(exp_path, 'rec_loss.txt')
    sample_seen, loss_val = read_curve(f_loss_path)
    decoder_path = os.path.join(exp_path, 'decoder.pth')
    decoder_type_path = os.path.join(exp_path, 'decoder_type.txt')
    if os.path.exists(decoder_type_path):
        with open(decoder_type_path, 'r') as f:
            content = [line.strip() for line in f.readlines()]
        if len(content) == 1:
            decoder_type = content[0]
            latent_size = 2
        else:
            decoder_type = content[0]
            latent_size = int(content[1])
    else:
        decoder_type = 'Decoder'
        latent_size = 2

    manifold_vis = vis_manifold(decoder_path, decoder_type=decoder_type, latent_size=latent_size)
    return sample_seen, loss_val, manifold_vis

exp_record_dir = './exps'

exp1_path = os.path.join(exp_record_dir, 
    'exp_latent8_1e_3_50epoch_2018-12-26 18:32:01.672492'
)
exp2_path = os.path.join(exp_record_dir, 
    'exp_latent8conv_1e_3_50epoch_2018-12-26 18:43:05.966052'
)
data1 = get_exp_data(exp1_path)
data2 = get_exp_data(exp2_path)

plt.figure(1, figsize=(18, 5))

plt.subplot(131)
plt.imshow(data1[2])

plt.subplot(132)
plt.imshow(data2[2])

plt.subplot(133)
plt.plot(data1[0], data1[1], color='r')
plt.plot(data2[0], data2[1], color='g')

plt.show()