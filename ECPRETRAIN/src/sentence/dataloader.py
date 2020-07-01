import copy
import json
import operator
import pickle as pkl
import numpy as np
from collections import OrderedDict
import time
import torch
from torch.autograd import Variable
#from torch.utils.serialization import load_lua
from torchfile import load as load_lua

from util import *

random = np.random
random.seed(1234)

def next_batch_joint(images, batch_size, num_dist, tt):
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = [], [], [], [], []
    total_indices = []
    keys = range(len(images))
    assert len(keys) >= num_dist
    #a = time.time()
    for batch_idx in range(batch_size):

        img_indices = random.permutation(len(images))[:num_dist]

        which = random.randint(0, num_dist) # (1)

        spk_img = img_indices[which]
        spk_imgs.append(spk_img) # (batch_size, 2048)
        
        lsn_imgs += list(img_indices)  # batch_size * num_dist

        whichs.append(which) # (batch_size)
    spk_imgs = torch.index_select(images,0,torch.tensor(spk_imgs)).numpy()
    lsn_imgs = torch.index_select(images,0,torch.tensor(lsn_imgs)).view(batch_size, num_dist,-1).numpy()
    whichs = np.array(whichs)
    spk_imgs = Variable(torch.from_numpy(spk_imgs), requires_grad=False).view(batch_size, -1)
    lsn_imgs = torch.from_numpy(lsn_imgs)
    lsn_imgs = Variable(lsn_imgs, requires_grad=False).view(batch_size, num_dist, -1)

    whichs = Variable(torch.LongTensor(whichs), requires_grad=False).view(batch_size)
    if tt == torch.cuda:
        spk_imgs = spk_imgs.cuda()
        lsn_imgs = lsn_imgs.cuda()
        whichs = whichs.cuda()
    return (spk_imgs, lsn_imgs, 0, 0, 0,0,0, whichs)


def weave_out(caps_out):
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans
