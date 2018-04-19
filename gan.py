from __future__ import print_function

from builtins import range
from tensorflow.examples.tutorials.mnist import input_data

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

batch_size = 64
n_epochs = 100

x_dim = mnist.train.images.shape[1]

z_dim = 100
hidden_dim = 128

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

G = Generator(in_dim=z_dim, hidden_dim=hidden_dim, out_dim=x_dim)
D = Discriminator(in_dim=x_dim, hidden_dim=hidden_dim, out_dim=1)

criterion = nn.BCELoss()
g_optimizer = optim.SGD(G.parameters(), lr=0.001, momentum=0.9)
d_optimizer = optim.SGD(D.parameters(), lr=0.001, momentum=0.9)

ones = Variable(torch.ones(batch_size, 1))
zeros = Variable(torch.zeros(batch_size, 1))

# temporary hack
n_steps = n_epochs
for step in range(n_steps):
	D.zero_grad()

	z = Variable(torch.randn(batch_size, z_dim))

	# NOTE: Use utils batch iterator
	x, _ = mnist.train.next_batch(batch_size)
	x = Variable(torch.from_numpy(x))

	# Dicriminator steps: forward-loss-backward-update
	generated_sample = G(z).detach()
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

	print('Step {} D loss: {}, G loss: {}'.format(step + 1, d_loss_total.data.numpy()[0], g_loss_total.data.numpy()[0]))
