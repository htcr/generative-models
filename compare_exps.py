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
    f_loss_path = os.path.join(exp_path, 'loss.txt')
    sample_seen, loss_val = read_curve(f_loss_path)
    decoder_path = os.path.join(exp_path, 'decoder.pth')
    decoder_type_path = os.path.join(exp_path, 'decoder_type.txt')
    if os.path.exists(decoder_type_path):
        with open(decoder_type_path, 'r') as f:
            decoder_type = f.readline()
    else:
        decoder_type = 'Decoder'

    manifold_vis = vis_manifold(decoder_path, decoder_type=decoder_type)
    return sample_seen, loss_val, manifold_vis

exp_record_dir = './exps'

exp1_path = os.path.join(exp_record_dir, 
    'exp_baseline_1e_3_50epoch_2018-12-24 21:53:55.751986'
)
exp2_path = os.path.join(exp_record_dir, 
    'exp_adameps_1e_3_50epoch_2018-12-25 23:50:45.361881'
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