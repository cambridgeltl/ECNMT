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

from dataloader import next_batch_joint

millis = int(round(time.time() * 10000)) % 10000
random.seed(millis)

def forward_joint(images, model, loss_dict_, args, loss_fn, num_dist, tt):

    en_batch = next_batch_joint(images, args.batch_size, num_dist, tt)
    l2_batch = en_batch
    output_en, output_l2, comm_actions, end_loss_, len_info = model(en_batch[:4],  args.sample_how)

    final_loss = 0
    lenlen = False
    if lenlen:
        en_spk_loss = loss_fn['xent'](torch.index_select(output_en.reshape(output_en.size(0)*output_en.size(1),-1),0,end_loss_[0]), end_loss_[1])
    else:
        en_spk_loss = torch.tensor(0).float().cuda()
    loss_dict_["average_len"].update(len_info[1].data)
    if args.loss_type == "xent":
        l2_diff_dist = torch.mean( torch.pow(output_l2[0] - output_l2[1], 2), 2).view(-1, num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        l2_lsn_loss = loss_fn['xent']( l2_logits, l2_batch[7] )
        l2_lsn_acc = logit_to_acc(l2_logits, l2_batch[7]) * 100
        final_loss += l2_lsn_loss 
    elif args.loss_type == "mse":
        en_diff_dist = torch.mean( torch.pow(output_en[1][0] - output_en[1][1], 2), 2).view(-1, args.num_dist)
        en_logits = 1 / (en_diff_dist + 1e-10)
        en_lsn_acc = logit_to_acc(en_logits, en_batch[7]) * 100

        en_diff_dist = torch.masked_select(en_diff_dist, idx_to_emb( en_batch[7].cpu().data.numpy(), args.num_dist, tt ))
        en_lsn_loss = loss_fn['mse']( en_diff_dist, Variable( tt.FloatTensor( en_diff_dist.size()).fill_(0) , requires_grad = False) )

        l2_diff_dist = torch.mean( torch.pow(output_l2[1][0] - output_l2[1][1], 2), 2).view(-1, args.num_dist)
        l2_logits = 1 / (l2_diff_dist + 1e-10)
        l2_lsn_acc = logit_to_acc(l2_logits, l2_batch[7]) * 100

        l2_diff_dist = torch.masked_select(l2_diff_dist, idx_to_emb( l2_batch[7].cpu().data.numpy(), args.num_dist, tt ))
        l2_lsn_loss = loss_fn['mse']( l2_diff_dist, Variable( tt.FloatTensor( l2_diff_dist.size()).fill_(0) , requires_grad = False) )

        final_loss += en_lsn_loss
        final_loss += l2_lsn_loss
    loss_dict_["accuracy"].update(l2_lsn_acc)
    loss_dict_["loss"].update(l2_lsn_loss.data) 
    return final_loss
