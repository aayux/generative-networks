from __future__ import print_function

from builtins import range

import os
import time
import torch

import matplotlib.pyplot as plt
from torch.autograd import Variable

def save(G, D, model):
    stamp = model + "_" + str(int(time.time()))
    save_dir = os.path.join(os.path.curdir, "runs", stamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(G.state_dict(), os.path.join(save_dir, model + '_generator.ckpt'))
    torch.save(D.state_dict(), os.path.join(save_dir, model + '_discriminator.ckpt'))
    return save_dir

def render(G, gen_path):
        G.load_state_dict(torch.load(gen_path))
        
        img = G(Variable(torch.randn(1, 100)))
        img = img.data.numpy().reshape((28, 28))
        
        plt.imshow(img, cmap='gray')
        plt.savefig("./sample.png", bbox_inches="tight")
        return
