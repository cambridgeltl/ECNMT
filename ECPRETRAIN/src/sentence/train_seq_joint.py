import sys
#import commands
import subprocess as commands
import codecs
import copy
import argparse
import math
import pickle as pkl
import os
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
#from torch.utils.serialization import load_lua
from torchfile import load as load_lua

from util import *
from models import *
from dataloader import *
from forward import *
#from time import time
random = np.random
random.seed(1234)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--gpuid", type=int, default=0,
                    help="Which GPU to run")
    parser.add_argument("--dataset", type=str, default="coco",
                    help="Which Image Dataset To Use EC Pretraining")
    parser.add_argument("--len_loss", type=int, default=False,
                    help="Which GPU to run")
    parser.add_argument("--vocab_size", type=int, default=4035, #The EC vocab_size should be in line with the vocab_size in NMT fine-tuning. 
                    help="EC vocab size")                    
    parser.add_argument("--alpha", type=float, default=1.0,
                    help="Which GPU to run")
    parser.add_argument("--two_fc", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--num_games", type=int, default=30000,
                    help="Total number of batches to train for")
    parser.add_argument("--batch_size", type=int, default=256,
                    help="Batch size For Training")
    parser.add_argument("--valid_batch_size", type=int, default=128,
                    help="Batch size For Validation")
    parser.add_argument("--num_dist", type=int, default=256,
                    help="Number of Distracting Images For Training")

    parser.add_argument("--num_dist_", type=int, default=128,
                    help="Number of Distracting Images For Validation")

    parser.add_argument("--D_img", type=int, default=2048,
                    help="ResNet feature dimensionality")
    parser.add_argument("--D_hid", type=int, default=512,
                    help="Token embedding dimensionality")
    parser.add_argument("--D_emb", type=int, default=256,
                    help="Token embedding dimensionality")
    parser.add_argument("--seq_len", type=int, default=15,
                    help="Max Len")
    parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1,
                    help="Dropout keep probability")

    parser.add_argument("--temp", type=float, default=1.0,
                    help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--TransferH", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--print_every", type=int, default=50,
                    help="Save model output.")
    parser.add_argument("--ECemb", type=int, default=5000,
                    help="Set The EC Embedding Size")                    
    parser.add_argument("--valid_every", type=int, default=250,
                    help="Validate model every k batches")
    parser.add_argument("--translate_every", type=int, default=2000,
                    help="Validate model every k batches")
    parser.add_argument("--save_every", type=int, default=4000,
                    help="Save model output.")

    parser.add_argument("--stop_after", type=int, default=30,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--num_directions", type=float, default=1,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--num_layers", type=float, default=1,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--unit_norm", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--cpu", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--pretrain_spk", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--which_loss", type=str, default="joint",
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--loss_type", type=str, default="xent",
                    help="Stop if validation loss doesn't improve after k iterations.")

    parser.add_argument("--fix_spk", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--fix_bhd", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--no_share_bhd", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--decode_how", type=str, default="greedy",
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--sample_how", type=str, default="gumbel",
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--beam_width", type=int, default=12,
                    help="Which GPU to run")
    parser.add_argument("--norm_pow", type=float, default=0.0,
                    help="Which GPU to run")

    parser.add_argument("--re_load", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--no_write", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--no_terminal", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    #torch.backends.cudnn.benachmark = False
    start_time = time.time()
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == []
    args_dict = vars(args)
    print("Entering Main")
    if args.dataset == "coco":
        feat_path = coco_path()
        data_path = coco_path()
        task_path = args.dataset
        args.l2 = "jp"
    else:
        print("image dataset should be set as coco")
        #here to insert alternative imgae data set


    (train_img1, train_img2, valid_img, test_img) = [torch.load('{}/half_feats/{}'.format(feat_path, x)) \
        for x in "train_en_feats train_{}_feats valid_feats test_feats".format(args.l2).split() ]


 
    print("Dataset Loaded")
    fixed, learned = [], ["lsn"]
    if args.fix_spk:      #False
        fixed.append("spk")
    else:
        learned.append("spk")

    if args.fix_bhd:      #False
        fixed.append("bhd")
    else:
        learned.append("bhd")

    fixed, learned = "_".join(sorted(fixed)), "_".join(sorted(learned))

    assert args.which_loss in "joint lsn".split() #which_loss = 'joint'
    model_str = "fixed_{}.learned_{}.{}_loss/".format(fixed, learned, args.which_loss) #fixed_.learned_bhd_lsn_spk.joint_loss/
    if args.pretrain_spk: #False
        model_str = "pretrain_spk." + model_str
    if args.no_share_bhd: #False
        model_str = "no_share_bhd." + model_str

    mill = int(round(time.time() * 1000)) % 1000

    big = "{}sentence_level/{}".format(saved_results_path(), task_path)
    path = "{}sentence_level/{}/joint_model/".format(saved_results_path(), task_path)
    hyperparam_str = "{}_dropout_{}.alpha_{}.lr_{}.temp_{}.D_hid_{}.D_emb_{}.num_dist_{}.vocab_size_{}_{}.hard_{}/".format(mill, args.dropout, args.alpha, args.lr, args.temp, args.D_hid, args.D_emb, args.num_dist, args.vocab_size, args.vocab_size, args.hard )
    path_dir = path + model_str + hyperparam_str

    if not args.no_write: #not_write = False
        recur_mkdir(path_dir)

    sys.stdout = Logger(path_dir, no_write=args.no_write, no_terminal=args.no_terminal)
    print(args)
    print(model_str)
    print(hyperparam_str)
    dir_dic = {"feat_path":feat_path, "data_path":data_path, "task_path":task_path, "path":path, "path_dir":path_dir}
    data = torch.cat([train_img1, train_img2, valid_img, test_img],dim=0)
    train_data, valid_data = remove_duplicate(data)
    train_data = train_data[:50000]
    valid_data = valid_data[:5000]   
 
    print('train_img :', type(train_data), train_data.shape)
    print('valid_img :', type(valid_data), valid_data.shape)


    model = SingleAgent(args)

    print(model)
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

    in_params, out_params = [], []
    in_names, out_names = [], []
    for name, param in model.named_parameters():
        if ("speaker" in name and args.fix_spk) or\
           ("beholder" in name and args.fix_bhd):
            out_params.append(param)
            out_names.append(name)
        else:
            in_params.append(param)
            in_names.append(name)

    in_size, out_size = [x.size() for x in in_params], [x.size() for x in out_params]
    in_sum, out_sum = sum([np.prod(x) for x in in_size]), sum([np.prod(x) for x in out_size])

    print("IN    : {} params".format(in_sum))
    print("OUT   : {} params".format(out_sum))
    print("TOTAL : {} params".format(in_sum + out_sum))

    loss_fn = {'xent':nn.CrossEntropyLoss(), 'mse':nn.MSELoss(), 'mrl':nn.MarginRankingLoss(), 'mlml':nn.MultiLabelMarginLoss(), 'mml':nn.MultiMarginLoss()}
    tt = torch
    if not args.cpu:
        loss_fn = {k:v.cuda() for (k,v) in loss_fn.items()}
        tt = torch.cuda

    optimizer = torch.optim.Adam(in_params, lr=args.lr)

    best_epoch = -1
    train_loss_dict_ = get_log_loss_dict_()
    for epoch in range(args.num_games):
        loss = forward_joint(train_data, model, train_loss_dict_, args, loss_fn, args.num_dist, tt)
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
        optimizer.step()

        if epoch % args.print_every == 0:
            avg_loss_dict_ = get_avg_from_loss_dict_(train_loss_dict_)
            print(print_loss_(epoch, args.alpha, avg_loss_dict_, "train"))
            train_loss_dict_ = get_log_loss_dict_()             

        with torch.no_grad():
            model.eval()
            if epoch % args.valid_every == 0:
                valid_loss_dict_ = get_log_loss_dict_()
                for idx in range(args.print_every):
                    _ = forward_joint(valid_data, model, valid_loss_dict_, args, loss_fn, args.num_dist_, tt)
                avg_loss_dict_ = get_avg_from_loss_dict_(valid_loss_dict_)
                s_new = print_loss_(epoch, args.alpha, avg_loss_dict_, "valid")
                print(s_new)
                if float(s_new.split()[-6][:-2]) > 99.0:
                    path_model = open( path_dir + "model_{}_{}_{}.pt".format(float(s_new.split()[-6][:-2]),epoch,args.vocab_size), "wb" )
                    torch.save(model.state_dict(), path_model)
                    print("Epoch :", epoch, "Prediction Accuracy =", float(s_new.split()[-6][:-2]), "Saved to Path :", path_dir)
                    if args.TransferH:
                        args.hard=True


        model.train()
    end_time = time.time()
    print("Total Runtime :", end_time-start_time)
