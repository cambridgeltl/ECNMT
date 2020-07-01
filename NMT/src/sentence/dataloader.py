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

def next_batch_joint(images, labels, batch_size, num_dist, tt):
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = [], [], [], [], []
    total_indices = []
    keys = range(len(images))
    assert len(keys) >= num_dist
    #a = time.time()
    for batch_idx in range(batch_size):

        #s1 = time.time()
        #img_indices = random.choice( keys, num_dist, replace=False ) # (num_dist)
        img_indices = random.permutation(len(images))[:num_dist]
        #s2 = time.time()
                
        num_caps = [len(labels[label]) for label in img_indices] # (num_dist) number of captions for each image
        #s3 = time.time()
        cap_indices = [random.randint(0, num_cap) for num_cap in num_caps] # (num_dist)
        #print("img_indices:",img_indices)
        #print("images :",len(images),images[0].size(),images.size())
        #s4 = time.time()
        #lsn_img = torch.index_select(images,0,torch.tensor(img_indices))
        
        #lsn_img = [ images[img_idx] for img_idx in img_indices ]
        #s5 = time.time()
        lsn_cap = [ labels[img_idx][cap_idx] for img_idx, cap_idx in zip(img_indices, cap_indices) ] # (num_dist, 2048)
        #s6 = time.time()
        which = random.randint(0, num_dist) # (1)
        #s7 = time.time()
        spk_img = img_indices[which]
        ###spk_img = lsn_img[which]
        spk_cap = lsn_cap[which]

        spk_imgs.append(spk_img) # (batch_size, 2048)
        
        spk_caps.append(spk_cap) # (batch_size, seq_len)
        ###lsn_imgs.append(lsn_img)  # (batch_size, num_dist, 2048)
        lsn_imgs += list(img_indices)  # batch_size * num_dist
        lsn_caps.append(lsn_cap)  # (batch_size, num_dist, 2048)
        whichs.append(which) # (batch_size)
        total_indices.append( img_indices )
        #s8 = time.time()
        #print("random :",s2-s1, s7-s6)
    #b = time.time() 
    #print("for loop time :", b-a)    
#    print("Print Data TYPE :", type(spk_imgs), type(spk_caps), type(lsn_imgs), type(lsn_caps), type(whichs))
#    print("Print Data SHAPE :", len(spk_imgs), len(spk_caps), len(lsn_imgs), len(lsn_caps), len(whichs))
#    print("Print Data TYPE :", type(spk_imgs[0]), type(spk_caps[0]), type(lsn_imgs[0]), type(lsn_caps[0]), type(whichs[0]))
#    print("Print Data TYPE :", type(spk_imgs[0]), spk_caps[0], lsn_imgs[0], lsn_caps[0], whichs[0])
    #print('lsn_imags:',type(lsn_imgs),len(lsn_imgs),type(lsn_imgs[0]),len(lsn_imgs[0]),lsn_imgs[0][0].shape)

    #spk_imgs = np.array([t.numpy() for t in spk_imgs])
    #a = time.time()
    spk_imgs = torch.index_select(images,0,torch.tensor(spk_imgs)).numpy()
#    a = time.time()
    #spk_imgs = torch.stack(spk_imgs,dim=0).numpy()
    #print(spk_imgs.shape)
    spk_caps = np.array(spk_caps)
    #lsn_imgs = np.array([[v.numpy() for v in t] for t in lsn_imgs])
    lsn_imgs = torch.index_select(images,0,torch.tensor(lsn_imgs)).view(batch_size, num_dist,-1).numpy()
    ###lsn_imgs = torch.stack([torch.stack(t,dim=0) for t in lsn_imgs],dim=0).numpy()
    #print('##########',lsn_imgs.shape)
    lsn_caps = np.array(lsn_caps)
    whichs = np.array(whichs)
    #b = time.time()
    #print("batch step 1 :", b-a)



#    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = np.array(spk_imgs), np.array(spk_caps), np.array(lsn_imgs), np.array(lsn_caps), np.array(whichs)
#    print("end Print Data TYPE :", type(spk_imgs), type(spk_caps), type(lsn_imgs), type(lsn_caps), type(whichs))
#    print("end Print Data SHAPE :", spk_imgs.shape, spk_caps.shape, lsn_imgs.shape, lsn_caps.shape, whichs.shape)

    #a = time.time()
    sorted_order = sort_per_len(spk_caps)
    #print(len(spk_imgs))
    #print("sorted_order :", sorted_order)
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = spk_imgs[sorted_order], spk_caps[sorted_order], lsn_imgs[sorted_order], lsn_caps[sorted_order], whichs[sorted_order]
    spk_cap_lens = np.array([len(x)-1 for x in spk_caps])
    seq_len = max(spk_cap_lens)

##    print("spk_caps 1:",spk_caps)
##    print("lsn_caps 1 2:",lsn_caps)
##    print("whichs 1:",whichs)

    spk_caps_in = [x[:-1] for x in spk_caps]
    spk_caps_out = [x[1:] for x in spk_caps]
    #print(spk_caps_in)
    spk_caps_in = [ np.lib.pad( cap, (0, seq_len - len(cap) ), 'constant', constant_values=(0,0) ) for cap in spk_caps_in ]
    spk_caps_in = np.array(spk_caps_in)
#    print("spk_caps_in :",spk_caps_in.shape)
#    print(spk_caps_in)
##    print("spk_caps_in 2:",spk_caps_in )
##    print("spk_cap_lens 2:", spk_cap_lens)

    spk_caps_in = Variable(torch.LongTensor(spk_caps_in), requires_grad=False)
    #print(spk_caps_in)
    spk_caps_out = weave_out(spk_caps_out)
    spk_caps_out = np.array(spk_caps_out)


##    print("spk_caps_out 2:",spk_caps_out)





    spk_caps_out = Variable(torch.LongTensor(spk_caps_out), requires_grad=False)

##    spk_imgs = Variable(torch.stack(spk_imgs), requires_grad=False).view(batch_size, -1)
    spk_imgs = Variable(torch.from_numpy(spk_imgs), requires_grad=False).view(batch_size, -1)
##    lsn_imgs = [torch.stack(x) for x in lsn_imgs]
    lsn_imgs = torch.from_numpy(lsn_imgs)
    #spk_imgs = torch.from_numpy(spk_imgs)
    
    
    lsn_imgs = Variable(lsn_imgs, requires_grad=False).view(batch_size, num_dist, -1)

    whichs = Variable(torch.LongTensor(whichs), requires_grad=False).view(batch_size)

#    print("spk_caps_in :",spk_caps_in.size())
#    print(spk_caps_in)
#    print("spk_caps_out :",spk_caps_out.size())
#    print(spk_caps_out)
#    print("spk_imgs :",spk_imgs.size())
#    print("ksn_imgs :",spk_imgs.size())
#    print("spk_caps_lens :",spk_cap_lens)
#    print("lsn_caps :",type(lsn_caps))
#    print(lsn_caps)
#    print("total_indices :",len(total_indices))
#    print(total_indices)
#    print("whichs :",whichs.size())
#    print(whichs)

    #b = time.time()
    #print("batch step 2 :", b-a)

    #a = time.time()
    if tt == torch.cuda:
        spk_imgs = spk_imgs.cuda()
        lsn_imgs = lsn_imgs.cuda()
        whichs = whichs.cuda()
        spk_caps_in = spk_caps_in.cuda()
        spk_caps_out = spk_caps_out.cuda()
    #b = time.time()
    #print("batch step 3 :", b-a)
    #print("end Print Data SHAPE :", spk_imgs.shape, spk_caps_in.shape, type(total_indices), spk_caps_out.shape,spk_cap_lens.shape,lsn_imgs.shape, lsn_caps.shape, whichs.shape)
    return (spk_imgs, lsn_imgs, spk_caps_in, spk_cap_lens, lsn_caps, total_indices, spk_caps_out, whichs)

def weave_out(caps_out):
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans

def next_batch_naka_enc(images, lab_org, batch_size, tt):
    image_ids = random.choice( range(len(lab_org)), batch_size, replace=False ) # (num_dist)
    caption_ids = [random.randint(0, len(lab_org[ image_ids[idx] ])) for idx in range(batch_size)  ]  # choose an object
    captions = np.array([lab_org[image_id][caption_id] for (image_id, caption_id) in zip(image_ids, caption_ids)])

    caption_lens = [(idx,len(x),image_id,caption_id) for idx, (x, image_id, caption_id) in enumerate(zip(captions, image_ids, caption_ids))]
    caption_lens = sorted(caption_lens, key=lambda x:x[1], reverse=True)

    captions = captions[ np.array( [x[0] for x in caption_lens] ) ]
    image_ids = [x[2] for x in caption_lens]

    caps_in = [x[:-1] for x in captions]
    caps_in_lens = np.array([len(x) for x in caps_in])

    caps_mid = [x[1:-1] for x in captions]
    caps_mid_lens = np.array([len(x) for x in caps_mid])

    caps_out = [x[1:] for x in captions]

    caps_in = [ np.lib.pad( cap, (0, max(caps_in_lens) - ln), 'constant', constant_values=(0,0) ) for (cap, ln) in zip(caps_in, caps_in_lens) ]
    caps_in = np.array(caps_in)
    caps_in = Variable(tt.LongTensor(caps_in), requires_grad=False)

    caps_mid = [ np.lib.pad( cap, (0, max(caps_mid_lens) - ln), 'constant', constant_values=(0,0) ) for (cap, ln) in zip(caps_mid, caps_mid_lens) ]
    caps_mid = np.array(caps_mid)
    caps_mid = Variable(tt.LongTensor(caps_mid), requires_grad=False)

    caps_out = weave_out(caps_out)
    caps_out = np.array(caps_out)
    caps_out = Variable(tt.LongTensor(caps_out), requires_grad=False)

    imgs = Variable(torch.index_select(images, 0, torch.LongTensor(image_ids)), requires_grad=False)

    # imgs : (batch_size, D_img)
    # caps_in : (batch_size, seq_len)
    # caps_in_lens : (batch_size)

    if tt == torch.cuda:
        imgs = imgs.cuda()

    return imgs, caps_in, caps_in_lens, caps_mid, caps_mid_lens, caps_out

def next_batch_nmt(src_lab_org, trg_lab_org, batch_size, tt):
    image_ids = random.choice( range(len(src_lab_org)), batch_size, replace=False ) # (num_dist)
    #print([len(src_lab_org[ image_ids[idx] ]) for idx in range(batch_size)  ])
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
