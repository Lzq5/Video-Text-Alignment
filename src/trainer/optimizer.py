import torch.optim as optim
import math
import numpy as np

import transformers
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def optim_policy(model, cfg, policy='default'):
    params = []
    no_decay = ['.ln_', '.bias']
    param_group_no_decay = []
    param_group_with_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if policy == 'default':
            if any([i in name for i in no_decay]):
                param_group_no_decay.append(param)
            else:
                param_group_with_decay.append(param)

    params.append({'params': param_group_no_decay, 'lr': cfg.optimizer.init_lr, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'lr': cfg.optimizer.init_lr, 'weight_decay': cfg.optimizer.weight_decay})
    return params


def build_optimizer(cfg, model):
    # params = model.parameters()
    params = optim_policy(model, cfg)

    if cfg.optimizer.name == 'sgd':
        optimizer = optim.SGD(params,
                              lr=cfg.optimizer.init_lr,
                              weight_decay=cfg.optimizer.weight_decay,
                              momentum=cfg.optimizer.momentum)
    elif cfg.optimizer.name == 'adam':
        optimizer = optim.Adam(params,
                               lr=cfg.optimizer.init_lr,
                               weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == 'adamw':
        optimizer = optim.AdamW(params,
                                lr=cfg.optimizer.init_lr,
                                weight_decay=cfg.optimizer.weight_decay)
    else:
        raise RuntimeError('Unknown optimizer: {}'.format(cfg.optimizer.name))
    return optimizer


def build_scheduler(cfg, optimizer, num_iters_per_epoch):

    # actually last iteration num
    last_epoch = max(0, (cfg.scheduler.start_epoch - 1)) * \
        num_iters_per_epoch - 1
    max_iters = cfg.scheduler.num_epochs * num_iters_per_epoch

    if cfg.scheduler.name == 'warmup_cosine':
        num_warmup_steps = int(max_iters * cfg.scheduler.warmup_ratio)
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=num_warmup_steps,
                                                                 num_training_steps=max_iters,
                                                                 last_epoch=last_epoch
                                                                 )
    elif cfg.scheduler.name == 'constant':
        scheduler = transformers.get_constant_schedule(optimizer, last_epoch)
    elif cfg.scheduler.name == 'cos_decay':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max_iters, last_epoch=last_epoch)
    elif cfg.scheduler.name == 'step':
        milestones = [epoch * num_iters_per_epoch for epoch in cfg.scheduler.milestones]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=cfg.scheduler.gamma)

    return scheduler
