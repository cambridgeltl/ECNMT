import sys
import subprocess as commands
import codecs
import copy
import argparse
import math
import pickle as pkl
import os
import numpy as np

import torch
import torch.nn.functional as F
from torchfile import load as load_lua
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable

from util import *
from models import NMT
from dataloader import next_batch_nmt
from forward import forward_nmt

random = np.random
random.seed(1234)

def translate(args, agent, labels, i2w, batch_size, which, tt):
    src_lab_org, trg_lab_org = labels["src"], labels["trg"]
    image_ids = range(batch_size)
    src_cap_ids = [random.randint(0, len(src_lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object
    trg_cap_ids = [random.randint(0, len(trg_lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object

    src_caps = np.array([src_lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, src_cap_ids)])
    trg_caps = np.array([trg_lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, trg_cap_ids)])

    src_sorted_idx = sort_per_len(src_caps)

    src_caps = [ x[1:-1] for x in src_caps[src_sorted_idx] ]
    trg_caps = [ x[1:-1] for x in trg_caps[src_sorted_idx] ]

    l2_src = print_captions(src_caps, i2w["src"], " ")
    en_ref = print_captions(trg_caps, i2w["trg"], " ")
    en_hyp = agent.translate(src_caps, args)

    print("---------------- {} TRANSLATION ----------------".format(which))
    for idx in range(len(en_hyp)):
        print(u"{} src {} | {}".format(args.src.upper(), idx+1, l2_src[idx] ))
        print(u"{} ref {} | {}".format(args.trg.upper(), idx+1, en_ref[idx].strip() ))
        print(u"{} hyp {} | {}".format(args.trg.upper(), idx+1, en_hyp[idx] ))
        print("")
    print("---------------------------------------------")

def valid_bleu(valid_labels, model, args, tt, dir_dic, which_dataset="valid"):
    src = valid_labels["src"]
    batch_size = 200
    num = 1 
    num_imgs = len(src)
    model_gen = [[] for x in range(num)]
    for cap_idx in range(num):
        for batch_idx in range( int( math.ceil( float(num_imgs) / batch_size ) ) ):
            start_idx = batch_idx * batch_size
            end_idx = min( num_imgs, (batch_idx + 1) * batch_size )
            l2_caps = np.array([src[img_idx][cap_idx][1:-1] for img_idx in range(start_idx, end_idx)])

            l2_cap_lens = sort_per_len(l2_caps)
            inverse = np.argsort(l2_cap_lens)

            l2_caps = l2_caps[l2_cap_lens]
            en_hyp = model.translate(l2_caps, args)
            en_hyp = [en_hyp[idx] for idx in inverse]
            model_gen[cap_idx].extend( en_hyp )

    final_out = []
    for idx in range(num_imgs):
        for i2 in range(num):
            final_out.append(model_gen[i2][idx])

    destination = dir_dic["path_dir"] + "{}_hyp_{}".format(which_dataset, args.decode_how)
    f = codecs.open(destination, 'wb', encoding="utf8")
    f.write( u'\r\n'.join( final_out ) )
    f.close()



    if (args.src == "en" and args.trg == "de") or (args.src == "de" and args.trg == "en"):
        command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref/{}_many_{}'.format(dir_dic["data_path"], args.trg, which_dataset), destination )
    elif (args.src == "en" and args.trg == "cs") or (args.src == "cs" and args.trg == "en"):
        command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref_encs/{}_many_{}'.format(dir_dic["data_path"], args.trg, which_dataset), destination )
    elif (args.src == "en" and args.trg == "ro") or (args.src == "ro" and args.trg == "en"):
        command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref_enro/{}_many_{}'.format(dir_dic["data_path"], args.trg, which_dataset), destination )
    elif (args.src == "en" and args.trg == "fr") or (args.src == "fr" and args.trg == "en"):
        command = 'perl {}/multi-bleu.perl {} < {}'.format(scr_path(), '{}/ref_enfr/{}_many_{}'.format(dir_dic["data_path"], args.trg, which_dataset), destination )

    bleu = commands.getstatusoutput(command)[1]
    print(which_dataset, bleu[ bleu.find("BLEU"): ])
    bleu_score = float(bleu[ bleu.find("=")+1: bleu.find(",", bleu.find("=")+1) ] )
    return bleu_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation from scratch')

    parser.add_argument("--gpuid", type=int, default=0,
                    help="Which GPU to run")
    parser.add_argument("--w2v", type=int, default=False,
                    help="Which GPU to run")
    parser.add_argument("--dif_loss", type=int, default=True,
                    help="Which GPU to run")

    parser.add_argument("--proj", type=int, default=True,
                    help="Which GPU to run")
    parser.add_argument("--dataset", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")

    parser.add_argument("--src", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")
    parser.add_argument("--trg", type=str,  # NOTE beam / greedy
                    help="Which GPU to run")
    parser.add_argument("--decode_how", type=str, default="beam",  # NOTE beam / greedy
                    help="Which GPU to run")
    parser.add_argument("--beam_width", type=int, default=12,
                    help="Which GPU to run")
    parser.add_argument("--norm_pow", type=float, default=0.0,
                    help="Which GPU to run")

    parser.add_argument("--num_games", type=int, default=10000000000,
                    help="Total number of batches to train for")
    parser.add_argument("--batch_size", type=int, default=128,
                    help="Batch size")

    parser.add_argument("--D_hid", type=int, default=512,
                    help="Token embedding dimensionality")
    parser.add_argument("--D_emb", type=int, default=256,
                    help="Token embedding dimensionality")

    parser.add_argument("--seq_len_src", type=int, default=80,
                    help="Token embedding dimensionality")
    parser.add_argument("--seq_len_trg", type=int, default=80,
                    help="Token embedding dimensionality")

    parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout keep probability")

    parser.add_argument("--temp", type=float, default=1.0,
                    help="Gumbel temperature")
    parser.add_argument("--hard", action="store_true", default=True,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--print_every", type=int, default=100,
                    help="Save model output.")
    parser.add_argument("--valid_every", type=int, default=250,
                    help="Validate model every k batches")
    parser.add_argument("--translate_every", type=int, default=2000,
                    help="Validate model every k batches")
    parser.add_argument("--save_every", type=int, default=2000,
                    help="Save model output.")

    parser.add_argument("--stop_after", type=int, default=60,
                    help="Stop if validation loss doesn't improve after k iterations.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                    help="Stop if validation loss doesn't improve after k iterations.")

    parser.add_argument("--cpu", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--unit_norm", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--no_write", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--no_terminal", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    parser.add_argument("--drop_img", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--drop_bhd", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--drop_emb", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--drop_out", action="store_true", default=False,
                    help="Hard Gumbel-Softmax Sampling.")

    args, remaining_args = parser.parse_known_args()
    print(remaining_args)
    assert remaining_args == []
    check_dataset_sanity(args)
    args_dict = vars(args)

    assert args.trg != None and args.src != None

    if args.dataset == "coco":
        data_path = coco_path()
        task_path = args.dataset
        args.l2 = "jp"
    elif args.dataset == "multi30k":
        data_path = multi30k_reorg_path() + "task{}/".format(1)
        task_path = "{}_task{}".format(args.dataset, 1)
        args.l2 = "de"

    args.to_drop = []
    if args.drop_img:
        args.to_drop.append("img")
    if args.drop_emb:
        args.to_drop.append("emb")
    if args.drop_out:
        args.to_drop.append("out")
    if args.drop_bhd:
        args.to_drop.append("bhd")
    args.to_drop.sort()

    print("LOAD BPE TOKENS AND DICT")


    if (args.src == "en" and args.trg == "de") or (args.src == "de" and args.trg == "en"):
        (w2i_src, i2w_src, w2i_trg, i2w_trg) = [torch.load(data_path + 'dics/{}'.format(x)) for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format(args.src, args.src, args.trg, args.trg).split()]
    elif (args.src == "en" and args.trg == "cs") or (args.src == "cs" and args.trg == "en"):
        (w2i_src, i2w_src, w2i_trg, i2w_trg) = [torch.load(data_path + 'dics_encs/{}'.format(x)) for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format(args.src, args.src, args.trg, args.trg).split()]
    elif (args.src == "en" and args.trg == "ro") or (args.src == "ro" and args.trg == "en"):
        (w2i_src, i2w_src, w2i_trg, i2w_trg) = [torch.load(data_path + 'dics_enro/{}'.format(x)) for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format(args.src, args.src, args.trg, args.trg).split()]
    elif (args.src == "en" and args.trg == "fr") or (args.src == "fr" and args.trg == "en"):
        (w2i_src, i2w_src, w2i_trg, i2w_trg) = [torch.load(data_path + 'dics_enfr/{}'.format(x)) for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format(args.src, args.src, args.trg, args.trg).split()]
    else:
        (w2i_src, i2w_src, w2i_trg, i2w_trg) = [torch.load(data_path + 'dics_entr/{}'.format(x)) for x in "{}_w2i {}_i2w {}_w2i {}_i2w".format(args.src, args.src, args.trg, args.trg).split()]




    print("LOAD BPE DATA")
    if (args.src == "en" and args.trg == "de") or (args.src == "de" and args.trg == "en"):
        (train_org_src_, valid_org_src, test_org_src) = [torch.load(data_path + 'full_labs/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.src, args.src, args.src).split()]
        (train_org_trg_, valid_org_trg, test_org_trg) = [torch.load(data_path + 'full_labs/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.trg, args.trg, args.trg).split()]
    elif (args.src == "en" and args.trg == "cs") or (args.src == "cs" and args.trg == "en"):
        (train_org_src_, valid_org_src, test_org_src) = [torch.load(data_path + 'full_labs_encs/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.src, args.src, args.src).split()]
        (train_org_trg_, valid_org_trg, test_org_trg) = [torch.load(data_path + 'full_labs_encs/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.trg, args.trg, args.trg).split()]
    elif (args.src == "en" and args.trg == "ro") or (args.src == "ro" and args.trg == "en"):
        (train_org_src_, valid_org_src, test_org_src) = [torch.load(data_path + 'full_labs_enro/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.src, args.src, args.src).split()]
        (train_org_trg_, valid_org_trg, test_org_trg) = [torch.load(data_path + 'full_labs_enro/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.trg, args.trg, args.trg).split()]
    elif (args.src == "en" and args.trg == "fr") or (args.src == "fr" and args.trg == "en"):
        (train_org_src_, valid_org_src, test_org_src) = [torch.load(data_path + 'full_labs_enfr/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.src, args.src, args.src).split()]
        (train_org_trg_, valid_org_trg, test_org_trg) = [torch.load(data_path + 'full_labs_enfr/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.trg, args.trg, args.trg).split()]
    else:
        (train_org_src_, valid_org_src, test_org_src) = [torch.load(data_path + 'full_labs_entr/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.src, args.src, args.src).split()]
        (train_org_trg_, valid_org_trg, test_org_trg) = [torch.load(data_path + 'full_labs_entr/{}'.format(x)) \
            for x in "{}_train_org {}_valid_org {}_test_org".format(args.trg, args.trg, args.trg).split()]




    print("mean length valid src:",sum([len(s[0]) for s in valid_org_src]) / len(valid_org_src))
    print("mean length test src:",sum([len(s[0]) for s in test_org_src]) / len(test_org_src))
    print("mean length valid trg:",sum([len(s[0]) for s in valid_org_trg]) / len(valid_org_trg))
    print("mean length test trg:",sum([len(s[0]) for s in test_org_trg]) / len(test_org_trg))

    print("max length valid src:",max([len(s[0]) for s in valid_org_src]))
    print("max length test src:",max([len(s[0]) for s in test_org_src]))
    print("max length valid trg:",max([len(s[0]) for s in valid_org_trg]))
    print("max length test trg:",max([len(s[0]) for s in test_org_trg]))

    print("min length valid src:",min([len(s[0]) for s in valid_org_src]))
    print("min length test src:",min([len(s[0]) for s in test_org_src]))
    print("min length valid trg:",min([len(s[0]) for s in valid_org_trg]))
    print("min length test trg:",min([len(s[0]) for s in test_org_trg]))

    model = "nmt"
    indices_ = random.permutation(len(train_org_src_))[:500]
    # Define the amount of training data if using 500 training samples, set indices_ = random.permutation(len(train_org_src_))[:500] instead
    print("train data:", indices_[:10])
    print(indices_[:10])
    print(indices_[-10:])
    train_org_src = [train_org_src_[i] for i in indices_]
    train_org_trg = [train_org_trg_[i] for i in indices_]
    print("data_path :", data_path)
    args.vocab_size_src = len(w2i_src)
    args.vocab_size_trg = len(w2i_trg)
    assert len(train_org_src) == len(train_org_trg)

    path = "{}{}/{}/".format(saved_results_path(), task_path, model)
    model_str = "{}-{}/".format(args.src, args.trg)
    hyperparam_str = "lr_{}.dropout_{}.D_hid_{}.D_emb_{}.vocab_size_{}_{}/".format(args.lr, args.dropout, args.D_hid, args.D_emb, args.vocab_size_src, args.vocab_size_trg)
    path_dir = path + model_str + hyperparam_str
    [check_mkdir(xx) for xx in [saved_results_path() + task_path, path, path + model_str, path_dir]]

    sys.stdout = Logger(path_dir, no_write=args.no_write, no_terminal=args.no_terminal)
    print("{} EN images, {} {} images".format(len(train_org_src), len(train_org_trg), args.l2.upper()))

    print(args)
    print(model_str)
    print(hyperparam_str)
    dir_dic = {"data_path":data_path, "task_path":task_path, "path":path, "path_dir":path_dir}

    args.vocab_size = {"src":args.vocab_size_src, "trg":args.vocab_size_trg}
    args.num_layers = {"spk" : { "src":1, "trg":1 }, \
                  "lsn" : { "src":1, "trg":1} }
    args.num_directions = {"lsn" : { "src":1, "trg":1 } }
    args.w2i = { "src":w2i_src, "trg":w2i_trg }
    args.i2w = { "src":i2w_src, "trg":i2w_trg }
    args.seq_len = { "src":args.seq_len_src, "trg":args.seq_len_trg }

    train_org_src = trim_caps(train_org_src, 3, args.seq_len_src * 20)
    train_org_trg = trim_caps(train_org_trg, 3, args.seq_len_trg * 20)

    if args.w2v:
        print("Loading Pretrained W2V EMB")
        en_embed, l2_embed, en_oow, l2_oow  =  pretrained_emb(i2w_en,i2w_l2)
        args.en_embed = torch.tensor(en_embed).cuda(args.gpuid)
        args.l2_embed = torch.tensor(l2_embed).cuda(args.gpuid)

        print("en_oow, l2_oow", len(en_oow), len(l2_oow))


    train_labels = { "src":train_org_src, "trg":train_org_trg }
    valid_labels = { "src":valid_org_src, "trg":valid_org_trg }
    test_labels = { "src":test_org_src, "trg":test_org_trg }



    if args.src == "en" and args.trg == "de":
     
        path_src = "" #enter your root
        path_trg = "" #enter your root

    elif args.src == "de" and args.trg == "en":
 
        path_src = "" #enter your root
        path_trg = "" #enter your root

    elif args.src == "en" and args.trg == "cs":
        path_src = "" #enter your root
        path_trg = "" #enter your root
    elif args.src == "cs" and args.trg == "en":
        path_src = "" #enter your root
        path_trg = "" #enter your root

    elif args.src == "en" and args.trg == "tr":
        path_src = "" #enter your root
        path_trg = "" #enter your root
    elif args.src == "tr" and args.trg == "en":
              
        path_src = "" #enter your root
        path_trg = "" #enter your root
    elif args.src == "en" and args.trg == "ro":
        path_src = "" #enter your root
        path_trg = "" #enter your root
    elif args.src == "ro" and args.trg == "en":
        path_src = "" #enter your root
        path_trg = "" #enter your root

    elif args.src == "en" and args.trg == "fr":
        path_src = "" #enter your root
        path_trg = "" #enter your root
    elif args.src == "fr" and args.trg == "en":
        path_src = "" #enter your root
        path_trg = "" #enter your root
    model_src_dict = torch.load(path_src)
    model_trg_dict = torch.load(path_trg)


 

    if not args.w2v:
        print("BPE EMBEDDING FROM PRETRAINED EMERGENT MODEL")
        args.src_embed = model_src_dict['listener.emb.weight'].cuda(args.gpuid)
        args.trg_embed = model_trg_dict['speaker.emb.weight'].cuda(args.gpuid)


    pretrained_dict={'decoder.rnn.weight_ih_l0' : model_trg_dict['speaker.rnn.weight_ih_l0'].cuda(args.gpuid), 'decoder.rnn.weight_hh_l0': model_trg_dict['speaker.rnn.weight_hh_l0'].cuda(args.gpuid), 'decoder.rnn.bias_ih_l0':model_trg_dict['speaker.rnn.bias_ih_l0'].cuda(args.gpuid), 'decoder.rnn.bias_hh_l0':model_trg_dict['speaker.rnn.bias_hh_l0'].cuda(args.gpuid), 'decoder.hid_to_voc.weight':model_trg_dict['speaker.hid_to_voc.weight'].cuda(args.gpuid), 'decoder.hid_to_voc.bias':model_trg_dict['speaker.hid_to_voc.bias'].cuda(args.gpuid) ,'encoder.rnn.weight_ih_l0':model_src_dict['listener.rnn.weight_ih_l0'].cuda(args.gpuid), 'encoder.rnn.weight_hh_l0':model_src_dict['listener.rnn.weight_hh_l0'].cuda(args.gpuid), 'encoder.rnn.bias_ih_l0':model_src_dict['listener.rnn.bias_ih_l0'].cuda(args.gpuid), 'encoder.rnn.bias_hh_l0':model_src_dict['listener.rnn.bias_hh_l0'].cuda(args.gpuid),'encoder.hid_to_hid.weight':model_src_dict['listener.hid_to_hid.weight'].cuda(args.gpuid), 'encoder.hid_to_hid.bias':model_src_dict['listener.hid_to_hid.bias'].cuda(args.gpuid)}

    if args.dif_loss:
        args.pretrained_dict = pretrained_dict
    model = NMT(args.src, args.trg, args)

###LOAD PRETRAINED MODEL
    model_dict=model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    del pretrained_dict, model_dict, model_src_dict, model_trg_dict
    torch.cuda.empty_cache()

###LOAD PRETRAINED MODEL
    print("PRETRAINED MODEL LOADED")

    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

    in_params, in_names = [], []
    for name, param in model.named_parameters():
        in_params.append(param)
        in_names.append(name)
        print(name, param.size())
    in_size = [x.size() for x in in_params]
    in_sum = sum([np.prod(x) for x in in_size])

    print("IN    : {} params".format(in_sum))

    loss_fn = {'xent':nn.CrossEntropyLoss(), 'mse':nn.MSELoss(), 'mrl':nn.MarginRankingLoss(), 'mlml':nn.MultiLabelMarginLoss(), 'mml':nn.MultiMarginLoss()}
    tt = torch
    if not args.cpu:
        loss_fn = {k:v.cuda() for (k,v) in loss_fn.items()}
        tt = torch.cuda

    optimizer = torch.optim.Adam(in_params, lr=args.lr)

    out_data = {'train':{'x':[], 'y':[] }, \
                'valid':{'x':[], 'y':[] }, \
                'bleu':{'x':[], 'y':[] }, \
                'best_valid':{'x':[], 'y':[] } }

    best_epoch = -1
    best_bleu = {"valid":-1, "test":-1}

    train_loss_dict = {"loss":AverageMeter(), "acc":AverageMeter(),"loss_seq":AverageMeter(), "loss_aux":AverageMeter()}
    for epoch in range(args.num_games):
        loss = forward_nmt(train_labels, model, train_loss_dict, args, loss_fn, tt, epoch)
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
        optimizer.step()

        if epoch % args.print_every == 0:
            print("epoch {} | train {:.3f} norm {:.3f} | loss_seq {:.3f} | loss_aux {:.9f}".format(epoch, train_loss_dict['loss'].avg, total_norm, train_loss_dict['loss_seq'].avg, train_loss_dict['loss_aux'].avg))
            out_data['train']['x'].append(epoch)
            out_data['train']['y'].append( train_loss_dict['loss'].avg )

            train_loss_dict['loss'].reset()
            train_loss_dict['loss_seq'].reset()
            train_loss_dict['loss_aux'].reset()

        model.eval()

        if epoch % args.valid_every == 0:
            valid_loss_dict = {"loss":AverageMeter(), "acc":AverageMeter(),"loss_seq":AverageMeter(), "loss_aux":AverageMeter()}
            for idx in range(args.print_every):
                _ = forward_nmt(valid_labels, model, valid_loss_dict, args, loss_fn, tt, epoch)

            cur_val_loss = valid_loss_dict['loss'].avg
            bleu_score = valid_bleu(valid_labels, model, args, tt, dir_dic, "valid")

            out_data['valid']['x'].append(epoch)
            out_data['valid']['y'].append(cur_val_loss)

            out_data['bleu']['x'].append(epoch)
            out_data['bleu']['y'].append(bleu_score)

            print("epoch {} | valid {:.3f} bleu {}".format(epoch, cur_val_loss, bleu_score))

            if bleu_score > best_bleu["valid"]:
                translate(args, model, valid_labels, args.i2w, 10, "VALID", tt)
                best_bleu["valid"] = bleu_score
                best_epoch = epoch
                path_model = open( path_dir + "best_model.pt", "wb" )
                torch.save(model.state_dict(), path_model)
                path_model.close()

                print("best model saved : {} BLEU".format(bleu_score))
                test_bleu = valid_bleu(test_labels, model, args, tt, dir_dic, "test")
                best_bleu["test"] = test_bleu
            else:
                if epoch - best_epoch >= args.stop_after * args.valid_every:
                    print("Validation BLEU not improving after {} iterations, early stopping".format( args.stop_after * args.valid_every ))
                    print("Best BLEU {}".format(best_bleu))
                    break


        if epoch > 0 and epoch % args.save_every == 0:
            path_results = open( path_dir + "results", "wb" )
            pkl.dump(out_data, path_results)
            path_results.close()
            print("results saved.")

        model.train()

    path_results = open( path_dir + "results", "wb" )
    pkl.dump(out_data, path_results)
    path_results.close()
    print("results saved.")

    result = open( path + model_str + "result", "a")
    result.write( "{}\t{}\t{}\n".format(best_bleu['valid'], best_bleu['test'], hyperparam_str) )
    result.close()
