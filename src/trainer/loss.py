import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from einops import rearrange
from sklearn import metrics 

from ..utils.registry import registry

@registry.register_criterion('similarity')
class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, pred_similarity, text_padding_mask, visual_padding_mask, time_mask):
        """
        target_time: [[], []]
        beta: 
        """
        T, S = visual_padding_mask.shape[1], text_padding_mask.shape[1]
        N, M = pred_similarity.shape[0], pred_similarity.shape[1] #N=BS, M=BT

        # generate mask for rows & columns of similarity matrix
        time_pos_mask = time_mask # True for related frames (B, S, T)
        time_neg_mask = ~(torch.block_diag(*time_pos_mask).bool()) # True for unrelated frames (BS, BT)
        time_pad_mask = visual_padding_mask.reshape(-1,)[None, :].repeat(N, 1) # True for visual padding (BS, BT)

        # compute InfoNCE according to each text
        v_numerator = torch.logsumexp(pred_similarity.masked_fill(time_neg_mask|time_pad_mask, -6e4), dim=-1) #(BS,) value may be very small
        v_denomenator = torch.logsumexp(pred_similarity.masked_fill(time_pad_mask, -6e4), dim=-1) #(BS,)

        null_text_mask = ~(((~time_neg_mask) == True).sum(-1).bool())
        v_nce_loss = -((v_numerator - v_denomenator).masked_fill(null_text_mask, 0)).mean()

        return v_nce_loss


def build_criterion(cfg, device=None):
    name = cfg.criterions.name
    params = cfg.criterions.params
    criterion = registry.get_criterion_class(name)(**params).to(device)
    return criterion