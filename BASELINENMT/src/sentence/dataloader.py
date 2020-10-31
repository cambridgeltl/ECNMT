# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
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
random.seed()

def weave_out(caps_out):
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans

def next_batch_nmt(src_lab_org, trg_lab_org, batch_size, tt):
    image_ids = random.choice( range(len(src_lab_org)), batch_size, replace=False ) # (num_dist)
    src_cap_ids = [random.randint(0, len(src_lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object
    trg_cap_ids = [random.randint(0, len(trg_lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object

    src_caps = np.array([src_lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, src_cap_ids)])
    trg_caps = np.array([trg_lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, trg_cap_ids)])

    src_sorted_idx = sort_per_len(src_caps)

    src_caps = src_caps[src_sorted_idx]
    trg_caps = trg_caps[src_sorted_idx]

    src_caps_in = [x[1:-1] for x in src_caps]
    src_caps_in_lens = [len(x) for x in src_caps_in]
    src_seq_len = max(src_caps_in_lens)

    trg_sorted_idx = sort_per_len(trg_caps)
    trg_caps = trg_caps[ trg_sorted_idx ]
    trg_sorted_idx = Variable(torch.LongTensor(trg_sorted_idx), requires_grad=False)

    trg_caps_out = [x[1:] for x in trg_caps]
    trg_caps_in = [x[:-1] for x in trg_caps]
    trg_caps_in_lens = [len(x) for x in trg_caps_in]
    trg_seq_len = max(trg_caps_in_lens)

    src_caps_in = [ np.lib.pad( cap, (0, src_seq_len - ln), 'constant', constant_values=(0,0) ) for (cap, ln) in zip(src_caps_in, src_caps_in_lens) ]
    src_caps_in = np.array(src_caps_in)
    src_caps_in = Variable(torch.LongTensor(src_caps_in), requires_grad=False)

    trg_caps_in = [ np.lib.pad( cap, (0, trg_seq_len - ln), 'constant', constant_values=(0,0) ) for (cap, ln) in zip(trg_caps_in, trg_caps_in_lens) ]
    trg_caps_in = np.array(trg_caps_in)
    trg_caps_in = Variable(torch.LongTensor(trg_caps_in), requires_grad=False)

    trg_caps_out = weave_out(trg_caps_out)
    trg_caps_out = np.array(trg_caps_out)
    trg_caps_out = Variable(torch.LongTensor(trg_caps_out), requires_grad=False)

    if tt == torch.cuda:
        src_caps_in = src_caps_in.cuda()
        trg_sorted_idx = trg_sorted_idx.cuda()
        trg_caps_in = trg_caps_in.cuda()
        trg_caps_out = trg_caps_out.cuda()

    return src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens, trg_caps_out
