from __future__ import print_function
from builtins import range

import sys
sys.path.append('../')

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dataset
import torchvision.transforms as transforms

import utils.general_utils as utils

parser = argparse.ArgumentParser()

parser.add_argument('--bsize', type=int, default=64, help='input batch size')
parser.add_argument('--zdim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--hdim', type=int, default=128, help='dimension of hidden layers')
parser.add_argument('--nepochs', type=int, default=10, help='number of training iterations')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')

opt = parser.parse_args()

batch_size = int(opt.bsize)
n_iterations = int(opt.nepochs)
lr = int(opt.lr)

dataset = dataset.MNIST(root='./data/',
                     transform=transforms.ToTensor()]),
                     download = True)

loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = batch_size,
                                     shuffle = True)

x_dim = 28 * 28
z_dim = int(opt.zdim)
h_dim = int(opt.hdim)

class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x

G = Generator(in_dim=z_dim, hidden_dim=h_dim, out_dim=x_dim)
D = Discriminator(in_dim=x_dim, hidden_dim=h_dim, out_dim=1)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=lr)
d_optimizer = optim.Adam(D.parameters(), lr=lr)

ones = Variable(torch.ones(batch_size, 1))
zeros = Variable(torch.zeros(batch_size, 1))

for iteration in range(n_iterations):
    for step, (x,_) in enumerate(loader):
        D.zero_grad()

        z = Variable(torch.randn(batch_size, z_dim))

        # Dicriminator steps: forward-loss-backward-update
        generated_samples = G(z)
        discriminate_real = D(x)
        discriminate_fake = D(generated_samples)

        d_loss_for_real = criterion(discriminate_real, ones)
        d_loss_for_fake = criterion(discriminate_fake, zeros)
        d_loss_total = d_loss_for_real + d_loss_for_fake

        d_loss_total.backward()
        d_optimizer.step()

        G.zero_grad()

        # Generator forward-loss-backward-update
        z = Variable(torch.randn(batch_size, z_dim))
        generated_samples = G(z)
        discriminate_fake = D(generated_samples)

        # Generate closest to reals
        g_loss_total = criterion(discriminate_fake, ones)

        g_loss_total.backward()
        g_optimizer.step()

        print("Step {} D loss: {}, G loss: {}".format(step + 1, d_loss_total.data.numpy(), g_loss_total.data.numpy()))
    print ("End of Epoch {}".format(iteration + 1))

    print ("Finished training. Saving model ...")
    save_dir = utils.save(G, D, 'gan')
    print ("Saved at location: {}".format(save_dir))

gen_ckpt = os.path.join(save_dir, 'gan_generator.ckpt')
utils.render(G, gen_ckpt)
