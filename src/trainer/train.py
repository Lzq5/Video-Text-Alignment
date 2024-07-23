import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from sklearn import metrics 
from tqdm import tqdm
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import random
import json
import ipdb
from ..dataset.dataset import pad_sequence_by_last, MRDataset
from ..utils.utils import AverageMeter
from ..utils.visualize import *
from torch.utils.data import DataLoader
import sys
sys.path.append('path_to_Video-Text-Alignment')
from src.InternVideo import load_model, tokenize


@torch.no_grad()
def evaluate_htmalign(cfg, loader, model, criterion, device, epoch):
    losses = AverageMeter('Loss', ':.4f')
    pbar = tqdm(loader, desc=f'Evaling HTM-Align epoch {epoch}')

    input_correct_num, pred_correct_num, total_num = 0, 0, 0

    model.eval()
    for batch_idx, input_batch in enumerate(pbar, start=1):
        model.eval()

        visual_input = input_batch['visual_input'].to(
            device, non_blocking=True)
        visual_padding_mask = input_batch['visual_padding_mask'].to(
            device, non_blocking=True)

        raw_text = input_batch['raw_text']
        text_input = input_batch['text_input'].to(
            device, non_blocking=True)
        text_padding_mask = input_batch['text_padding_mask'].to(
            device, non_blocking=True)
        target_sign = input_batch['target_sign'].to(device, non_blocking=True)
        time_mask = input_batch['time_mask'].to(device, non_blocking=True)

        shuffle = input_batch['shuffle']

        # forward
        with torch.no_grad():
            pred_similarity_list, pred_time = model.compute_all(
                visual_input, visual_padding_mask, text_input, text_padding_mask, shuffle)
            
            input_time = model.compute_input_similarity(visual_input, visual_padding_mask, text_input, text_padding_mask)
            loss = criterion(pred_similarity_list[-1], text_padding_mask, visual_padding_mask, time_mask)

        B, T, _ = visual_input.shape[:]
        S = text_input.shape[1]
        losses.update(loss.item(), B)

        time_mask = time_mask.reshape(B*S, T)
        text_sign = (target_sign == 1).reshape(B*S,)
        input_time = input_time.reshape(B*S, -1)
        pred_time = pred_time.reshape(B*S, -1)
        input_correct_num += (torch.gather(time_mask, 1, input_time)[text_sign] == True).float().sum().data
        pred_correct_num += (torch.gather(time_mask, 1, pred_time)[text_sign] == True).float().sum().data
        total_num += text_sign.float().sum()

    print("Test Recall (epoch={}): {:.4f}".format(epoch, pred_correct_num/total_num))
    if cfg.use_wandb:
        wandb.log({
            "test/recall_input": input_correct_num/total_num,
            "test/recall_pred": pred_correct_num/total_num,
            "test/avg total loss": losses.avg,
        }, step=epoch)


@torch.no_grad()
def evaluate_htmstep(cfg, loader, model, criterion, device, epoch):
    
    pbar = tqdm(loader, desc=f'Evaling HTM-Step epoch {epoch}')
    result = {}

    model.eval()
    for batch_idx, input_batch in enumerate(pbar, start=1):
        model.eval()

        visual_input = input_batch['visual_input'].to(
            device, non_blocking=True)
        visual_padding_mask = input_batch['visual_padding_mask'].to(
            device, non_blocking=True)

        vid = input_batch['vid']
        raw_text = input_batch['raw_text']
        text_input = input_batch['text_input'].to(
            device, non_blocking=True)
        text_padding_mask = input_batch['text_padding_mask'].to(
            device, non_blocking=True)
        
        shuffle = input_batch['shuffle']

        # forward
        with torch.no_grad():
            _, pred_time = model.compute_all(
                visual_input, visual_padding_mask, text_input, text_padding_mask, shuffle)
            
        for idx, time in enumerate(pred_time):
            result[vid[idx]] = time[~text_padding_mask[idx]].cpu().numpy().tolist()

    with open(osp.join(cfg.output_dir, "result.json"), "w") as fp:
        json.dump(result, fp, indent=2)

def read_assignment(T, K, path):
    Y = np.zeros([T, K], dtype=np.uint8)
    with open(path,'r') as f:
        for line in f:
            step,start,end = line.strip().split(',')
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1
            Y[start:end,step] = 1
    return Y

def get_recalls(Y_true, Y_pred):
    step_match = {task: 0 for task in Y_true.keys()}
    step_total = {task: 0 for task in Y_true.keys()}
    for task,ys_true in Y_true.items():
        ys_pred = Y_pred[task]
        for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
            y_true = ys_true[vid]
            y_pred = ys_pred[vid]
            step_total[task] += (y_true.sum(axis=0)>0).sum()
            step_match[task] += (y_true*y_pred).sum()
    recalls = {task: step_match[task] / n for task,n in step_total.items()}
    return recalls

def train_similarity(cfg, loader, model, criterion, optimizer, lr_scheduler, device, epoch):
    losses = AverageMeter('Loss', ':.4f')
    pbar = tqdm(loader, desc=f'Training epoch {epoch}')

    pred_correct_num, total_num = 0, 0
    
    for batch_idx, input_batch in enumerate(pbar, start=1):
        model.train()
        
        # ipdb.set_trace()
        visual_input = input_batch['visual_input'].to(
            device, non_blocking=True)
        visual_padding_mask = input_batch['visual_padding_mask'].to(
            device, non_blocking=True)
        
        raw_text = input_batch['raw_text']
        text_input = input_batch['text_input'].to(
            device, non_blocking=True) 
        text_padding_mask = input_batch['text_padding_mask'].to(
            device, non_blocking=True)
        time_mask = input_batch['time_mask'].to(device, non_blocking=True)

        shuffle = input_batch['shuffle']

        B, T, _ = visual_input.shape[:]
        S = text_input.shape[1]
        # forward
        pred_similarity_list, pred_time = model.compute_all(
            visual_input, visual_padding_mask, text_input, text_padding_mask, shuffle)

        loss, num_layers = .0, pred_similarity_list.shape[0]
        for i in range(num_layers):
            loss += criterion(pred_similarity_list[i], text_padding_mask, visual_padding_mask, time_mask)
        loss /= num_layers
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        if 'grad_clip' in cfg.optimizer:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        
        pred_correct_num += (torch.gather(time_mask, 2, pred_time[:, :, None]) == True).float().sum().data
        total_num += sum([len(text_list) for text_list in raw_text])

    if cfg.use_wandb:
        wandb.log({
            "train/avg total loss": losses.avg, 
            "train/lr": lr_scheduler.get_last_lr()[0],
            "train/recall_pred": pred_correct_num/total_num,
            "epoch": epoch,
        }, step=epoch)

def do_train(cfg, loaders, model, criterion, optimizer, lr_scheduler, device, checkpointer):

    num_params = sum(p.numel() for p in model.parameters())*1.0/1e6
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)*1.0/1e6
    stat1 = "\nTotal number of model parameters: {:.2f}M".format(num_params)
    stat2 = "\nNumber of learnable parameters: {:.2f}M\n".format(trainable_params)
    print(stat1, stat2)

    start, total = cfg.scheduler.start_epoch, cfg.scheduler.num_epochs
    if checkpointer.ckpt is not None:
        evaluate_htmstep(cfg, loaders['ht-step'], model, criterion, device, 0)
        evaluate_htmalign(cfg, loaders['htm-align'], model, criterion, device, 0)
        return

    for epoch in range(start, start+total):
        train_similarity(
            cfg, 
            loaders['train'], 
            model, 
            criterion, 
            optimizer, 
            lr_scheduler, 
            device, 
            epoch
        )
        if epoch % cfg.eval_epochs == 0 or epoch == 1:
            evaluate_htmalign(cfg, loaders['htm-align'], model, criterion, device, epoch)
        if epoch == total:
            if cfg.model.save_model:
                checkpointer.save(epoch, model, optimizer)
            evaluate_htmstep(cfg, loaders['ht-step'], model, criterion, device, 0)

def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))