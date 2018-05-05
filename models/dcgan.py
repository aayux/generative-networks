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
from torch.autograd import Variable

import torchvision.datasets as dataset
import torchvision.transforms as transforms

import utils.general_utils as utils

parser = argparse.ArgumentParser()

parser.add_argument('--bsize', type=int, default=64, help='input batch size')
parser.add_argument('--idim', type=int, default=32, help='size of the image, default=32')
parser.add_argument('--zdim', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--hdim', type=int, default=128, help='dimension of hidden layers')
parser.add_argument('--nepochs', type=int, default=10, help='number of training iterations')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')

opt = parser.parse_args()

batch_size = int(opt.bsize)
n_iterations = int(opt.nepochs)
lr = int(opt.lr)

dataset = dataset.MNIST(root='./data/',transform=
                     transforms.Compose([transforms.Resize(int(opt.idim)),
                                         transforms.ToTensor()]),
                     download = True)

loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = batch_size,
                                     shuffle = True)

x_dim = int(opt.idim)
z_dim = int(opt.zdim)
h_dim = int(opt.hdim)

class Generator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Generator, self).__init__()           
        self.deconv_1 = nn.Sequential(                      
                            nn.ConvTranspose2d(in_dim, hidden_dim * 4, 4, bias=False),
                            nn.BatchNorm2d(hidden_dim * 4),
                            nn.ReLU(True))
        
        self.deconv_2 = nn.Sequential(                      
                            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(hidden_dim * 2),
                            nn.ReLU(True))
        
        self.deconv_3 = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(hidden_dim),
                            nn.ReLU(True))
        
        self.deconv_4 = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dim, out_dim, 4, 2, 1, bias=False),
                            nn.Tanh())
        
    def forward(self, x):
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(
                            nn.Conv2d(in_dim, hidden_dim, 4, 2, 1, bias=False),
                            nn.LeakyReLU(0.2, inplace=True))

        self.conv_2 = nn.Sequential(
                            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(hidden_dim * 2),
                            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv_3 = nn.Sequential(
                            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(hidden_dim * 4),
                            nn.LeakyReLU(0.2, inplace=True))
        
        self.conv_4 = nn.Sequential(
                            nn.Conv2d(hidden_dim * 4, 1, 4, bias=False),
                            nn.Sigmoid())    
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x

G = Generator(in_dim=z_dim, hidden_dim=h_dim, out_dim=1)
D = Discriminator(in_dim=1, hidden_dim=h_dim, out_dim=1)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(G.parameters(), lr=lr)
d_optimizer = optim.Adam(D.parameters(), lr=lr)

ones = Variable(torch.ones(batch_size, 1))
zeros = Variable(torch.zeros(batch_size, 1))

for iteration in range(n_iterations):
    for step, (x,_) in enumerate(loader):
        D.zero_grad()

        z = Variable(torch.randn(batch_size, z_dim, 1, 1))

        # Dicriminator steps: forward-loss-backward-update
        generated_samples = G(z.detach())
        discriminate_real = D(x)
        discriminate_fake = D(generated_samples)

        d_loss_for_real = criterion(discriminate_real, ones)
        d_loss_for_fake = criterion(discriminate_fake, zeros)
        d_loss_total = d_loss_for_real + d_loss_for_fake

        d_loss_total.backward()
        d_optimizer.step()

        G.zero_grad()

        # Generator forward-loss-backward-update
        z = Variable(torch.randn(batch_size, z_dim, 1, 1))
        generated_samples = G(z)
        discriminate_fake = D(generated_samples)

        # Generate closest to reals
        g_loss_total = criterion(discriminate_fake, ones)

        g_loss_total.backward()
        g_optimizer.step()

        print("Step {} D loss: {}, G loss: {}".format(step + 1, d_loss_total.data.numpy(), g_loss_total.data.numpy()))
    print ("End of Epoch {}".format(iteration + 1))

    print ("Finished training. Saving model ...")
    save_dir = utils.save(G, D, 'dcgan')
    print ("Saved at location: {}".format(save_dir))

gen_ckpt = os.path.join(save_dir, 'dcgan_generator.ckpt')
utils.render(G, gen_ckpt)
