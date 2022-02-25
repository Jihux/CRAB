import torch as ch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import os
import itertools
import math


def ablate(x, pos, k, total_pos, dim):
    # x : input
    # pos : starting position
    # k : size of ablation
    # total_pos : maximum position
    # dim : height or width (2 or 3)
    inp = ch.zeros_like(x)
    mask = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
    if pos + k > total_pos:
        idx1 = [slice(None, None, None) if _ != dim else slice(
            pos, total_pos, None) for _ in range(4)]
        idx2 = [slice(None, None, None) if _ != dim else slice(
            0, pos+k-total_pos, None) for _ in range(4)]
        inp[idx1] = x[idx1]
        inp[idx2] = x[idx2]
        mask[idx1] = 1
        mask[idx2] = 1
    else:
        idx = [slice(None, None, None) if _ != dim else slice(
            pos, pos+k, None) for _ in range(4)]
        inp[idx] = x[idx]
        mask[idx] = 1
    return ch.cat([inp, mask], dim=1)


def ablate2(x, block_pos, block_k, shape):
    inp = ch.zeros_like(x)
    mask = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))

    slices = []
    for pos, k, total_pos in zip(block_pos, block_k, shape):
        if pos + k > total_pos:
            slices.append([slice(0, pos+k-total_pos, None),
                          slice(pos, total_pos, None)])
        else:
            slices.append([slice(pos, pos+k, None)])

    for si, sj in itertools.product(*slices):
        idx = [slice(None, None, None), slice(None, None, None), si, sj]
        inp[idx] = x[idx]
        mask[idx] = 1

    return ch.cat([inp, mask], dim=1)


def certify_models(args, models, validation_loader, store=None):
    # print("Certification is replacing transform with ToTensor")
    ablation_size = args.certify_ablation_size
    stride = args.certify_stride
    if not os.path.exists(args.certify_out_dir):
        os.makedirs(args.certify_out_dir)
    # if os.path.exists(os.path.join(args.certify_out_dir, 'certified_acc.pt')):
    #     print('============================================')
    #     return

    if args.dataset == 'cifar10':
        nclasses = 10
        img_size = 32
    elif args.dataset == 'mnist':
        nclasses = 10
        img_size = 28
    else:
        raise ValueError("Unknown number of classes")
    pred = ch.zeros((10000, img_size))
    labels = ch.zeros((10000,))
    max_patch_size = img_size//2-ablation_size

    if isinstance(models, list):
        # model smoothing
        for model, pos in tqdm(zip(models, range(0, img_size, stride)), ncols=0):
            model.cuda()
            model.eval()

            for i, (X, y) in enumerate(validation_loader):
                if args.batch_id != None and args.batch_id < i:
                    break
                if args.batch_id != None and args.batch_id != i:
                    continue
                X, y = X.cuda(), y.cuda()

                out = model(ablate(X, pos, ablation_size, img_size, 3))
                if isinstance(out, tuple):
                    out = out[0]
                out = out.argmax(1)
                pred[i*args.batch_size:i*args.batch_size+len(y), pos] = out
                if pos == 0:
                    labels[i*args.batch_size:i*args.batch_size+len(y)] = y

            model.cpu()
            
    else:
        # no model smoothing
        for pos in tqdm(range(0, img_size, stride), ncols=0):
            for i, (X, y) in enumerate(validation_loader):
                X, y = X.cuda(), y.cuda()

                out = models(ablate(X, pos, ablation_size, img_size, 3))
                if isinstance(out, tuple):
                    out = out[0]
                out = out.argmax(1)
                pred[i*args.batch_size:i*args.batch_size+len(y), pos] = out
                if pos == 0:
                    labels[i*args.batch_size:i*args.batch_size+len(y)] = y
            
    
    ch.save({'pred':pred,'label':labels}, os.path.join(args.certify_out_dir, 'pred_lab.pt'))
    acc=[]
    if not args.random_patch:
        acc_col = []
        pred_count_full = ch.stack([(pred == i).sum(1) for i in range(nclasses)], dim=1)
        c1_count_full, c1_label_full = pred_count_full.kthvalue(nclasses, dim=1)
        c2_count_full, c2_label_full = pred_count_full.kthvalue(nclasses-1, dim=1)
        acc0 = (c1_count_full > c2_count_full) & (c1_label_full == labels)
        acc_col.append([acc0.sum().item()/10000]*img_size)
        acc.append(acc0.sum().item()/10000)
        
        
        for patch_size in range(1, max_patch_size+1):
            acc_ = []
            affected_column_count = patch_size+ablation_size-1
            cer_and_acc_full = (c1_count_full-c2_count_full > affected_column_count*2) & (c1_label_full == labels)
            acc.append(cer_and_acc_full.sum().item()/10000)
            for patch_pos in range(img_size):
                first_affected_ablation_pos = ablation_size-patch_pos-1
                shifted_pred = pred.roll(first_affected_ablation_pos, 1)
                # shift all affected preditcions to the begining of row
                unaffected_pred = shifted_pred[:, affected_column_count:]
                pred_count = ch.stack([(unaffected_pred == i).sum(1) for i in range(nclasses)], dim=1)
                c1_count, c1_label = pred_count.kthvalue(nclasses, dim=1)
                c2_count, c2_label = pred_count.kthvalue(nclasses-1, dim=1)
                cer_and_acc = (c1_count-c2_count > affected_column_count) & (c1_label == labels)
                acc_.append(cer_and_acc.sum().item()/10000)
            acc_col.append(acc_)
            
        print("==========certified accuracy==========")
        for i in range(max_patch_size+1):
            print(f"Patch size: {i} Certified accuracy: {acc[i]} Column: {acc_col[i]}")
        ch.save(acc, os.path.join(args.certify_out_dir, 'certified_acc.pt'))
        ch.save(acc_col, os.path.join(args.certify_out_dir, 'certified_acc_col.pt'))
            
    else:
        pred_count = ch.stack([(pred == i).sum(1) for i in range(nclasses)], dim=1)
        c1_count, c1_label = pred_count.kthvalue(nclasses, dim=1)
        c2_count, c2_label = pred_count.kthvalue(nclasses-1, dim=1)
        affected_column_count = args.certify_patch_size+ablation_size-1
        if args.certify_patch_size == 0:
            affected_column_count = 0
        cer_and_acc = (c1_count-c2_count > affected_column_count*2) & (c1_label == labels)
        acc.append(cer_and_acc.sum().item()/10000)
        
        print("==========certified accuracy==========")
        print(f"Patch size: {args.certify_patch_size} Certified accuracy: {acc[0]}")
                
        ch.save(acc, os.path.join(args.certify_out_dir, 'certified_acc.pt'))
    return acc

