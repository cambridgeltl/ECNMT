# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time
import random
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from util import idx_to_emb, logit_to_acc

from dataloader import next_batch_nmt

millis = int(round(time.time() * 10000)) % 10000
random.seed(millis)

def forward_nmt(labels, model, loss_dict, args, loss_fn, tt, valid=False):
    if valid == False:
        batch = args.batch_size
    else:
        batch = args.valid_batch_size
    src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens, trg_caps_out = next_batch_nmt(labels["src"], labels["trg"], batch, tt)
    dec_logits = model(src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens)

    loss = loss_fn['xent'](dec_logits, trg_caps_out)
    loss_dict['loss'].update(loss.data)
    return loss


