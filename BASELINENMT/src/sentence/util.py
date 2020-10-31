# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pprint
import codecs
import os
import sys
import time
import pickle as pkl
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
from numpy import random
random.seed(1234)





def gen_ref(path_org, num, path_new):
    try:
        os.remove(path_new)
    except OSError:
        pass
    g = open(path_org,"r")
    lines = g.readlines()
    g.close()
    
    num = set(num)
    with open(path_new, "w") as f:
        for k in range(len(lines)):
            if k in num:
                f.write(lines[k])





def filter_bad(src,trg):
    note = []
    s_ = set([])
    for i in range(len(src)):
        s = str(src[i]) + str(trg[i])
        if s in s_:
            note.append(i)
        else:
            s_.add(s)
    return set(note)
def return_valid_test(src,trg,bad,w2i_trg):

    print("src trg :", len(src),len(trg),len(src[0]),len(trg[0]))
    valid_list = []
    for i in range(len(trg)):
        if i not in bad:
            if (trg[i][0][-2] == w2i_trg['.']) or (trg[i][0][-2] == w2i_trg['?']):
                len_trg = len(trg[i][0])
                len_src = len(src[i][0])
                if len_src > 3 and len_trg > 3 and len_src < 26 and len_trg < 26: 
                    valid_list.append(i)
    return valid_list   
def return_train(src,trg,bad,w2i_trg,valid_list,test_list): 
    train_list = []
    for i in range(len(trg)):
        if (i not in bad) and (i not in valid_list) and (i not in test_list):
            if trg[i][0][-2] == w2i_trg['.'] or trg[i][0][-2] == w2i_trg['?']:
                len_trg = len(trg[i][0])
                len_src = len(src[i][0])
                if len_src > 3 and len_trg > 3 and len_src < 26 and len_trg < 26: 
                    train_list.append(i)
    return train_list   
def make_ref(valid_list, test_list, path_valid, path_test, path_train):
    path_valid = path_valid 
    path_test = path_test 
    try:
        os.remove(path_valid)
    except OSError:
        pass
    try:
        os.remove(path_test)
    except OSError:
        pass
    g = open(path_train,"r")
    lines = g.readlines()
    g.close()
    with open(path_valid, "w") as f:
        for k in valid_list:
            sen = lines[k]
            final = " ".join(["".join(w.split()) for w in sen.split("▁")]).strip()
            f.write(final+'\n')
    with open(path_test, "w") as f:
        for k in test_list:
            sen = lines[k]
            final = " ".join(["".join(w.split()) for w in sen.split("▁")]).strip()           
            f.write(final+'\n')


def restore(i2w_en,i2w_l2,s_en,s_l2):
    sen = [i2w_en[i] for i in s_en]
    sl2 = [i2w_l2[i] for i in s_l2]
    sen = " ".join(sen)
    sl2 = " ".join(sl2)
    print(len(s_en)," ".join(["".join(w.split()) for w in sen.split("▁")]).strip())
    print(len(s_l2)," ".join(["".join(w.split()) for w in sl2.split("▁")]).strip())



def make_ref_tmp(valid_data, path_valid, i2w):
    try:
        os.remove(path_valid)
    except OSError:
        pass
    with open(path_valid, "w") as f:
        for sen in valid_data:
            seq = [i2w[i] for i in sen[0][1:-1]]
            seq = " ".join(seq)
            final = " ".join(["".join(w.split()) for w in seq.split("▁")]).strip()
            f.write(final+'\n')



def return_work_list_and_count(path, voc_size):
    word_list = []
    count = []
    w2i = {}
    with open(path) as f:
        idx = 0
        for line in f:
            toks = line.split()
            for tok in toks:
                if tok in w2i:
                    count[w2i[tok]] += 1
                else:
                    w2i[tok] = idx
                    word_list.append(tok)
                    count.append(1)
                    idx += 1
    count_idx = sorted(range(len(count)), key=lambda k: count[k],reverse=True)
    new_word_list = [word_list[i] for i in count_idx]
    return new_word_list[:voc_size-4], sorted(count,reverse=True)[:voc_size-4]

def return_dic(w2i, i2w, path, voc_size):
    word_list, _ = return_work_list_and_count(path, voc_size)
    k = 4
    for w in word_list:
        w2i[w] = k
        i2w[k] = w
        k += 1
    return w2i, i2w

def return_data(w2i, i2w, path):
    data = []
    with open(path) as f:
        for line in f:
            sent = [w2i['<BOS>']]
            toks = line.split()
            for tok in toks:
                if tok in w2i:
                    sent.append(w2i[tok])
                else:
                    sent.append(w2i['<UNK>'])
            sent.append(w2i['<EOS>'])
            data.append([sent])
    return data    

def final_return_w2i_i2w():
    train_en_path = "/workspace/multi30k/dataset/data/task1/tok/train.lc.norm.tok.en" 
    valid_en_path = "/workspace/multi30k/dataset/data/task1/tok/val.lc.norm.tok.en"
    test_en_path = "/workspace/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.en"
    train_de_path = "/workspace/multi30k/dataset/data/task1/tok/train.lc.norm.tok.de" 
    valid_de_path = "/workspace/multi30k/dataset/data/task1/tok/val.lc.norm.tok.de"
    test_de_path = "/workspace/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.de"
    voc_size_en = 4035
    voc_size_de = 5445
    w2i_en = {'<PAD>':0, '<UNK>':1, '<BOS>':2, '<EOS>':3}
    i2w_en = {0: '<PAD>', 1:'<UNK>', 2:'<BOS>',3:'<EOS>'}
    w2i_l2 = {'<PAD>':0, '<UNK>':1, '<BOS>':2, '<EOS>':3}
    i2w_l2 = {0: '<PAD>', 1:'<UNK>', 2:'<BOS>',3:'<EOS>'}
    w2i_en, i2w_en = return_dic(w2i_en, i2w_en, train_en_path, voc_size_en)
    w2i_l2, i2w_l2 = return_dic(w2i_l2, i2w_l2, train_de_path, voc_size_de)
    return w2i_en, i2w_en, w2i_l2, i2w_l2

def final_return_data(w2i_en, i2w_en, w2i_l2, i2w_l2):
    train_en_path = "/workspace/multi30k/dataset/data/task1/tok/train.lc.norm.tok.en" 
    valid_en_path = "/workspace/multi30k/dataset/data/task1/tok/val.lc.norm.tok.en"
    test_en_path = "/workspace/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.en"
    train_de_path = "/workspace/multi30k/dataset/data/task1/tok/train.lc.norm.tok.de" 
    valid_de_path = "/workspace/multi30k/dataset/data/task1/tok/val.lc.norm.tok.de"
    test_de_path = "/workspace/multi30k/dataset/data/task1/tok/test_2016_flickr.lc.norm.tok.de"

    train_org_en = return_data(w2i_en, i2w_en, train_en_path)
    valid_org_en = return_data(w2i_en, i2w_en, valid_en_path)
    test_org_en = return_data(w2i_en, i2w_en, test_en_path)
    train_org_l2 = return_data(w2i_l2, i2w_l2, train_de_path)
    valid_org_l2 = return_data(w2i_l2, i2w_l2, valid_de_path)
    test_org_l2 = return_data(w2i_l2, i2w_l2, test_de_path)
    return train_org_en, valid_org_en, test_org_en, train_org_l2, valid_org_l2, test_org_l2


def pretrained_emb(i2w_en,i2w_l2):
    path_emb_en = "/media/data/enwiki_20180420_300d.txt"
    path_emb_l2 = "/media/data/dewiki_20180420_300d.txt"
    en_vectors = {}
    l2_vectors = {}
    print("Reading EN W2V EMB")
    with open(path_emb_en) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            if i > 0:
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ')+1:]
                en_vectors[word] = vector
    print("Reading L2 W2V EMB")
    with open(path_emb_l2) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            if i > 0:
                s = line.strip()
                word = s[:s.find(' ')]
                vector = s[s.find(' ')+1:]
                l2_vectors[word] = vector


    en_embed = []
    en_oow = []
    l2_embed = []
    l2_oow = []
    print("Constructing EN EMB LAYER")
    for i in range(len(i2w_en)):
        word = i2w_en[i]
        if word in en_vectors:
            vector = list(map(float, en_vectors[word].split()))
        elif word.strip("@") in en_vectors:
            vector = list(map(float, en_vectors[word.strip("@")].split()))
        else:
            en_oow.append(i)
            vector = [0.0]*300
        en_embed.append(vector)
    en_embed = np.array(en_embed, dtype=np.float32) 

    print("Constructing L2 EMB LAYER")
    for i in range(len(i2w_l2)):
        word = i2w_l2[i]
        if word in l2_vectors:
            vector = list(map(float, l2_vectors[word].split()))
        elif word.strip("@") in l2_vectors:
            vector = list(map(float, l2_vectors[word.strip("@")].split()))
        else:
            l2_oow.append(i)
            vector = [0.0]*300
        l2_embed.append(vector)
    l2_embed = np.array(l2_embed, dtype=np.float32) 
    return en_embed, l2_embed, en_oow, l2_oow



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def sum_num_captions(org):
    return sum([len(x) for x in org])

def get_coco_idx():
    a, b = 56644, 56643
    a_, b_ = [], []

    cand = 0
    while len(a_) < 14500:
        if not cand in a_:
            a_.append(cand)
            cand = cand + 4
        else:
            cand += 1
        cand = cand % 56644

    cand = 0
    while len(b_) < 14500:
        if not cand in b_:
            b_.append(cand)
            cand += 4
        else:
            cand += 1
        cand = cand % 56643

    assert( len(set(a_)) == 14500 )
    assert( len(set(b_)) == 14500 )

    return a_, b_

def recur_mkdir(dir):
    ll = dir.split("/")
    ll = [x for x in ll if x != ""]
    for idx in range(len(ll)):
        ss = "/".join(ll[0:idx+1])
        check_mkdir("/"+ss)

class Logger(object):
    def __init__(self, path, no_write=False, no_terminal=False):
        self.no_write = no_write
        if self.no_write:
            print("Don't write to file")
        else:
            self.log = codecs.open(path+"log.log", "wb", encoding="utf8")

        self.no_terminal = no_terminal
        self.terminal = sys.stdout

    def write(self, message):
        if not self.no_write:
            self.log.write(message)
        if not self.no_terminal:
            self.terminal.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def check_dataset_sanity(args):
    assert args.dataset == "coco" or args.dataset == "multi30k"
    if args.dataset == "coco":
        assert (args.src, args.trg) == ("en", "jp") or (args.src, args.trg) == ("jp", "en")
    elif args.dataset == "multi30k":
       
        assert (args.src, args.trg) == ("en", "de") or (args.src, args.trg) == ("de", "en") or (args.src, args.trg) == ("cs", "en") or (args.src, args.trg) == ("en", "cs") or (args.src, args.trg) == ("en", "tr") or (args.src, args.trg) == ("tr", "en") or (args.src, args.trg) == ("en", "ro") or (args.src, args.trg) == ("ro", "en") or (args.src, args.trg) == ("fr", "en") or (args.src, args.trg) == ("en", "fr")

def scr_path():
    return "" #enter your root

def saved_results_path():
    return "" #enter your root

def multi30k_reorg_path():
    return "" #enter your root

def coco_path():
    return "" #enter your root

def sort_per_len(caps):
    lens = [(idx, len(cap)) for idx, cap in enumerate(caps)]
    lens.sort(key=lambda x: x[1], reverse=True)
    lens = np.array([x[0] for x in lens])
    assert len(lens) == len(caps)
    return lens

def trim_caps(caps, minlen, maxlen):
    new_cap = [ [ cap for cap in cap_i if len(cap) <= maxlen and len(cap) >= minlen] for cap_i in caps]
    print("Before : {} captions / After : {} captions".format( sum_num_captions(caps), sum_num_captions(new_cap) ))
    return new_cap

def print_params_naka(names, sizes):
    comps = "decoder_trg encoder_src encoder_trg beholder".split()

    dd = OrderedDict()
    for cc in comps:
        dd[cc] = {}

    for name, size in zip(names, sizes):
        name_ = name.split(".")
        cc, rest = name_[0], ".".join(name_[1:])
        dd[cc][rest] = "{} ({})".format(rest, size[0]) if len(size) == 1 else "{} ({}, {})".format(rest, size[0], size[1])

    ss = ""
    for cc in comps:
        if len(dd[cc]) > 0:
            ss += "\t{}:\t".format(cc)
            for rr in sorted(dd[cc].keys()):
                ss += dd[cc][rr] + ", "
            ss += "\n"

    return ss

def print_params_nmt(names, sizes):
    comps = "encoder decoder".split()

    dd = OrderedDict()
    for cc in comps:
        dd[cc] = {}

    for name, size in zip(names, sizes):
        name_ = name.split(".")
        cc, rest = name_[0], ".".join(name_[1:])
        dd[cc][rest] = "{} ({})".format(rest, size[0]) if len(size) == 1 else "{} ({}, {})".format(rest, size[0], size[1])

    ss = ""
    for cc in comps:
        if len(dd[cc]) > 0:
            ss += "\t{}:\t".format(cc)
            for rr in sorted(dd[cc].keys()):
                ss += dd[cc][rr] + ", "
            ss += "\n"

    return ss

def print_params(names, sizes):
    agents = "l1_agent l2_agent".split()
    comps = "speaker listener beholder".split()

    dd = OrderedDict()
    for aa in agents:
        dd[aa] = {}
        for cc in comps:
            dd[aa][cc] = {}

    for name, size in zip(names, sizes):
        name_ = name.split(".")
        aa, cc, rest = name_[0], name_[1], ".".join(name_[2:])
        dd[aa][cc][rest] = "{} ({})".format(rest, size[0]) if len(size) == 1 else "{} ({}, {})".format(rest, size[0], size[1])

    ss = ""
    for aa in agents:
        ss += "\t{}\n".format(aa)
        for cc in comps:
            if len(dd[aa][cc]) > 0:
                ss += "\t\t{}:\t".format(cc)
                for rr in sorted(dd[aa][cc].keys()):
                    ss += dd[aa][cc][rr] + ", "
                ss += "\n"

    return ss

def print_captions(gen_indices, i2w, joiner,flores):
    #print("i2w :", i2w.type(), i2w)
    if flores:
        #print("seqs[0] :", seqs[0])
        seqs = [ " ".join( [i2w[ii] for ii in gen_idx] ).replace("@@ ", "") for gen_idx in gen_indices]
        #print("seqs[0] :", seqs[0])
        #print("return[0] :", [" ".join(["".join(w.split()) for w in seq.split("▁")]).strip() for seq in seqs][0])
        return [" ".join(["".join(w.split()) for w in seq.split("▁")]).strip() for seq in seqs]
    else:
        return [ joiner.join( [i2w[ii] for ii in gen_idx] ).replace("@@ ", "") for gen_idx in gen_indices]

def decode(gen_indices, i2w, flores):
    #print("i2w :", type(i2w), i2w)
    

    #print("gen_indices :", gen_indices[0],gen_indices[0][0].type())
    #gen_indices = [torch.tensor(gen_idx).tolist() for gen_idx in gen_indices]
    if flores:
        #print("seqs[0] :", seqs[0])
        seqs = [ " ".join( [i2w[ii] for ii in gen_idx] ).replace("@@ ", "") for gen_idx in gen_indices]
        #print("seqs[0] :", seqs[0])
        #print("return[0] :", [" ".join(["".join(w.split()) for w in seq.split("▁")]).strip() for seq in seqs][0])
        return [" ".join(["".join(w.split()) for w in seq.split("▁")]).strip() for seq in seqs]   
    else:
        return [ " ".join( [i2w[ii] for ii in gen_idx] ).replace("@@ ", "") for gen_idx in gen_indices]

def pick(i1, i2, whichs):
    res = []
    img = [i1, i2]
    for idx, which in enumerate(whichs):
        res.append(img[which][idx])
    return res

def idx_to_onehot(indices, nb_digits): # input numpy array
    y = torch.LongTensor(indices).view(-1, 1)
    y_onehot = torch.FloatTensor(indices.shape[0], nb_digits)

    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)

    return y_onehot

def max_logit_to_onehot(logits):
    max_element, max_idx = torch.max(logits.cuda(), 1)
    onehot = torch.FloatTensor(logits.size())
    onehot.zero_()
    onehot.scatter_(1, max_idx.data.long().cpu(), 1)
    onehot = Variable(torch.FloatTensor(onehot), requires_grad=False).cuda()
    return onehot, max_idx.data

def sample_logit_to_onehot(logits):
    #idx = torch.multinomial(logits, 1, replacement=True)
    indices = torch.multinomial(logits, 1)
    onehot = torch.FloatTensor(logits.size())
    onehot.zero_()
    for ii, jj in enumerate(indices.data.cpu().numpy().flatten().tolist()):
        onehot[ii][jj] = 1
    onehot = Variable(onehot, requires_grad=False).cuda()
    return onehot, indices.data

def logit_to_acc(logits, y): # logits: [batch_size, num_of_classes]
    y_max, y_max_idx = torch.max(logits, 1) # [batch_size]
    eq = torch.eq(y_max_idx, y)
    acc = float(eq.sum().data) / float(eq.nelement())
    return acc

def logit_to_top_k(logits, y, k): # logits: [batch_size, num_of_classes]
    logits_sorted, indices = torch.sort(logits, 1, descending=True)
    y = y.view(-1, 1)
    indices = indices[:,:k]
    y_big = y.expand(indices.size())
    eq = torch.eq(indices, y_big)
    eq2 = torch.sum(eq, 1)
    #acc = float(eq2.sum().data[0]) / float(eq2.nelement())
    #return acc
    return eq2.sum().data[0], eq2.nelement()

def loss_and_acc(logits, labels, loss_fn):
    loss = loss_fn(logits, labels)
    acc = logit_to_acc(logits, labels)
    return (loss, acc)

def loss_acc_dict():
    return {
        "spk":{\
               "loss":0},\
        "lsn":{\
               "loss":0,\
               "acc":0 }\
        }

def loss_acc_meter():
    return {
        "spk":{\
               "loss":AverageMeter()},\
        "lsn":{\
               "loss":AverageMeter(),\
               "acc":AverageMeter() }\
        }

def get_loss_dict():
    return { "l1":loss_acc_dict(), "l2":loss_acc_dict() }

def get_log_loss_dict():
    return {"l1":loss_acc_meter(), "l2":loss_acc_meter() }

def get_avg_from_loss_dict(log_loss_dict):
    res = get_loss_dict()
    for k1, v1 in log_loss_dict.items(): # en_agent / fr_agent
        for k2, v2 in v1.items(): # spk / lsn
            for k3, v3 in v2.items(): # loss / acc
                res[k1][k2][k3] = v3.avg
    return res

def l3_print_loss(epoch, alpha, avg_loss_dict, mode="train"):
    prt_msg = ""
    for pair in "en_de en_fr de_fr".split():
        prt_msg += "epoch {:5d} {} || {} |".format(epoch, mode, pair.upper())
        for agent in pair.split("_"):
            prt_msg += "| " # en_agent / fr_agent

            for person in "spk lsn".split():
                prt_msg += " {}_{}".format(agent, person) # spk / lsn

                if person == "spk":
                    prt_msg += " {:.3f}".format(avg_loss_dict[pair][agent][person]["loss"])
                elif person == "lsn":
                    prt_msg += " {:.3f} * {} = {:.3f}".format(avg_loss_dict[pair][agent][person]["loss"], alpha, avg_loss_dict[pair][agent][person]["loss"] * alpha)
                    prt_msg += " {:.2f}%".format(avg_loss_dict[pair][agent][person]["acc"])
                prt_msg += " |"
        prt_msg += "\n"
    return prt_msg

def print_loss(epoch, alpha, avg_loss_dict, mode="train"):
    prt_msg = "epoch {:5d} {} ".format(epoch, mode)
    #avg_loss_dict = get_avg_from_loss_dict(loss_dict)
    #for k1, v1 in avg_loss_dict.iteritems():
    for agent in "l1 l2".split():
        prt_msg += "| " # en_agent / fr_agent
        #for k2, v2 in v1.iteritems():
        for person in "spk lsn".split():
            prt_msg += " {}_{}".format(agent, person) # spk / lsn
            #for k3, v3 in v2.iteritems(): # loss / acc
            if person == "spk":
                prt_msg += " {:.3f}".format(avg_loss_dict[agent][person]["loss"])
            elif person == "lsn":
                prt_msg += " {:.3f} * {} = {:.3f}".format(avg_loss_dict[agent][person]["loss"], alpha, avg_loss_dict[agent][person]["loss"] * alpha)
                prt_msg += " {:.2f}%".format(avg_loss_dict[agent][person]["acc"])
            prt_msg += " |"
    return prt_msg

def clip_grad(v, min, max):
    v.register_hook(lambda g: g.clamp(min, max))
    return v

def check_mkdir(dir):
    try:
        os.stat(dir)
    except:
        os.mkdir(dir)

def idx_to_emb(idx, maxmax, tt):
    ans = tt.ByteTensor( len(idx), maxmax ).fill_(0)
    for aaa, iii in enumerate(idx):
        ans[aaa][iii] = 1
    return Variable(ans, requires_grad=False)

