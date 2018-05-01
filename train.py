from __future__ import print_function

from builtins import range
from tensorflow.examples.tutorials.mnist import input_data

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import utils.general_utils as utils

mnist = input_data.read_data_sets('./MNIST', one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='gan', help='select model from models directory')
parser.add_argument('--bsize', type=int, default=64, help='input batch size')
parser.add_argument('--zdim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--hdim', type=int, default=128, help='dimension of hidden layers')
parser.add_argument('--nepochs', type=int, default=50, help='number of training iterations')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')

opt = parser.parse_args()

# TO DO: Is there a better way to do this?
if model == 'gan':
	from models.gan import Generator, Discriminator
elif model == 'dcgan':
	from models.dgan import Generator, Discriminator

batch_size = int(opt.bsize)
sample_size = mnist.train.images.shape[0]
n_iterations = int(int(opt.nepochs) * (sample_size / batch_size))
lr = int(opt.lr)

x_dim = mnist.train.images.shape[1]
z_dim = int(opt.zdim)
hidden_dim = int(opt.hdim)

G = Generator(in_dim=z_dim, hidden_dim=hidden_dim, out_dim=x_dim)
D = Discriminator(in_dim=x_dim, hidden_dim=hidden_dim, out_dim=1)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=lr)
d_optimizer = optim.Adam(D.parameters(), lr=lr)

ones = Variable(torch.ones(batch_size, 1))
zeros = Variable(torch.zeros(batch_size, 1))

for iteration in range(n_iterations):
    D.zero_grad()

    z = Variable(torch.randn(batch_size, z_dim))

    x, _ = mnist.train.next_batch(batch_size)
    x = Variable(torch.from_numpy(x))

    # Dicriminator steps: forward-loss-backward-update
    generated_sample = G(z)
    discriminate_real = D(x)
    discriminate_fake = D(generated_sample)

    d_loss_for_real = criterion(discriminate_real, ones)
    d_loss_for_fake = criterion(discriminate_fake, zeros)
    d_loss_total = d_loss_for_real + d_loss_for_fake

    d_loss_total.backward()
    d_optimizer.step()

    G.zero_grad()

    # Generator forward-loss-backward-update
    z = Variable(torch.randn(batch_size, z_dim))
    generated_sample = G(z)
    discriminate_fake = D(generated_sample)

    # Generate closest to reals
    g_loss_total = criterion(discriminate_fake, ones)

    g_loss_total.backward()
    g_optimizer.step()

    print('Step {} D loss: {}, G loss: {}'.format(iteration + 1, d_loss_total.data.numpy(), g_loss_total.data.numpy()))

print ("Finished training. Saving model ...")
save_dir = utils.save(G, D, opt.model)
print ("Saved at location: {}".format(save_dir))

gen_ckpt = os.path.join(save_dir, opt.model + '_generator.ckpt')
utils.render(G, gen_ckpt)