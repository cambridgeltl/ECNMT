from math import sqrt
import time
import random
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

from util import idx_to_emb, logit_to_acc, cal_new_loss

from dataloader import next_batch_joint, next_batch_naka_enc, next_batch_nmt

millis = int(round(time.time() * 10000)) % 10000
random.seed(millis)

def forward_nmt(labels, model, loss_dict, args, loss_fn, tt, epoch):
    src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens, trg_caps_out = next_batch_nmt(labels["src"], labels["trg"], args.batch_size, tt)
    dec_logits = model(src_caps_in, src_caps_in_lens, trg_sorted_idx, trg_caps_in, trg_caps_in_lens)

    if args.dif_loss:
        current_dict = {}
        for name, para in model.named_parameters():
            current_dict[name] = para
        loss_aux = cal_new_loss(current_dict, args.pretrained_dict)
        loss_dict['loss_aux'].update(loss_aux.data)
    else:
        loss_dict['loss_aux'].update(0.0)
    loss_seq = loss_fn['xent'](dec_logits, trg_caps_out) 

    loss_dict['loss_seq'].update(loss_seq.data)
 
    if args.dif_loss:

        alpha = pow(0.998,epoch)*5.0
#        alpha = 5.0/(epoch+1.0)

        loss = loss_seq + alpha * loss_aux 
    else:
        loss = loss_seq

    loss_dict['loss'].update(loss.data)
    return loss
