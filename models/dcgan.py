from __future__ import print_function

from builtins import range
from tensorflow.examples.tutorials.mnist import input_data

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import utils.general_utils as utils

mnist = input_data.read_data_sets('./MNIST', one_hot=True)

batch_size = 64
n_iterations = 10000

x_dim = mnist.train.images.shape[1]

z_dim = 100
hidden_dim = 128

class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Generator, self).__init__()
        self.seq_block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim, out_dim, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.seq_block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Discriminator, self).__init__()
        self.seq_block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, out_dim, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.seq_block(x)
        return x

G = Generator(in_dim=z_dim, hidden_dim=hidden_dim, out_dim=x_dim)
D = Discriminator(in_dim=x_dim, hidden_dim=hidden_dim, out_dim=1)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=0.001)
d_optimizer = optim.Adam(D.parameters(), lr=0.001)

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
save_dir = utils.save(G, D)
print ("Saved at location: {}".format(save_dir))

gen_ckpt = os.path.join(save_dir, 'dcgan_generator.ckpt')
utils.render(G, gen_ckpt)
