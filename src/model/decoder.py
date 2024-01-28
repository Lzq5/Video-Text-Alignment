import math
import sys
import os
import numpy as np
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from ..utils.registry import registry
from .transformer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # (B, L, d_model)
        return self.dropout(x)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@registry.register_model('detr')
class DETR(nn.Module):

    def __init__(self, cfg):
        super(DETR, self).__init__()
        if cfg.dataset.visual_backbone == 'internvideo':
            self.d_input = 768
        elif cfg.dataset.visual_backbone == 'S3D':
            self.d_input = 512
        elif cfg.dataset.visual_backbone == 'CLIP':
            self.d_input = 768
        self.d_model = cfg.model.d_model
        self.visual_proj = nn.Sequential(
            nn.Linear(self.d_input, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.visual_pos_enc = PositionalEncoding(self.d_model)

        self.text_proj = nn.Sequential(
            nn.Linear(self.d_input, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        self.d_proj = cfg.model.d_proj
        self.v_sim_proj = nn.Linear(self.d_model, self.d_proj)
        self.t_sim_proj = nn.Linear(self.d_model, self.d_proj)

        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.model.nhead,
            dropout=cfg.model.dropout,
        )
        self.visual_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=cfg.model.enc_layers,
        )

        self.use_text_pos = cfg.dataset.text_pe
        self.text_pos_enc = nn.Embedding(1024, self.d_model)
        
        decoder_layer = TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=cfg.model.nhead,
            dropout=cfg.model.dropout,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_layers=cfg.model.dec_layers,
            norm=nn.LayerNorm(self.d_model),
            return_intermediate=True
        )

        self.alignable_head = nn.Linear(self.d_model, 2)
        self.timestamp_head = nn.Sequential(
            MLP(self.d_model, 128, output_dim=1, num_layers=3),
            nn.Sigmoid(),
        )

        self.apply(self.weights_init)
        self.t = torch.tensor(100.0)


    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, val=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=.02)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, visual_input, visual_padding_mask, text_input, text_padding_mask):
        T, S = visual_input.shape[1], text_input.shape[1]

        visual_input = visual_input.transpose(0, 1)
        text_input = text_input.transpose(0, 1)
        
        visual_embed = self.visual_proj(visual_input)
        visual_pos_enc = self.visual_pos_enc.pe[:, :T, :].transpose(0, 1)

        text_embed = self.text_proj(text_input)
        text_pos_enc = self.text_pos_enc.weight[:S, None, :]

        visual_embed = self.visual_encoder(
            visual_embed, src_key_padding_mask=visual_padding_mask, pos=visual_pos_enc)

        if self.use_text_pos:
            out = self.decoder(text_embed, visual_embed, tgt_key_padding_mask=text_padding_mask,
                            memory_key_padding_mask=visual_padding_mask, pos=visual_pos_enc,
                            query_pos=text_pos_enc).transpose(1, 2)
        else:
            out = self.decoder(text_embed, visual_embed, tgt_key_padding_mask=text_padding_mask,
                           memory_key_padding_mask=visual_padding_mask, pos=visual_pos_enc,
                           query_pos=None).transpose(1, 2)  
        
        normalized_text = F.normalize(self.t_sim_proj(out[-1]), dim=-1)
        normalized_visual = F.normalize(self.v_sim_proj(visual_embed).transpose(0, 1), dim=-1)
        mask = torch.logical_or(visual_padding_mask[:, None, :].repeat(1, S, 1), text_padding_mask[:, :, None].repeat(1, 1, T))
        last_layer_similarity = torch.einsum('BSD,BTD->BST', [normalized_text, normalized_visual])
        return last_layer_similarity

    def compute_all(self, visual_input, visual_padding_mask, text_input, text_padding_mask):
        T, S = visual_input.shape[1], text_input.shape[1]

        visual_input = visual_input.transpose(0, 1)
        text_input = text_input.transpose(0, 1)
        
        visual_embed = self.visual_proj(visual_input)
        visual_pos_enc = self.visual_pos_enc.pe[:, :T, :].transpose(0, 1)

        text_embed = self.text_proj(text_input)
        text_pos_enc = self.text_pos_enc.weight[:S, None, :]
        visual_embed = self.visual_encoder(
            visual_embed, src_key_padding_mask=visual_padding_mask, pos=visual_pos_enc)

        if self.use_text_pos:
            out = self.decoder(text_embed, visual_embed, tgt_key_padding_mask=text_padding_mask,
                            memory_key_padding_mask=visual_padding_mask, pos=visual_pos_enc,
                            query_pos=text_pos_enc).transpose(1, 2)
        else:
            out = self.decoder(text_embed, visual_embed, tgt_key_padding_mask=text_padding_mask,
                           memory_key_padding_mask=visual_padding_mask, pos=visual_pos_enc,
                           query_pos=None).transpose(1, 2)

        normalized_text = F.normalize(self.t_sim_proj(out[-1]), dim=-1)
        normalized_visual = F.normalize(self.v_sim_proj(visual_embed).transpose(0, 1), dim=-1)
        mask = torch.logical_or(visual_padding_mask[:, None, :].repeat(1, S, 1), text_padding_mask[:, :, None].repeat(1, 1, T))
        last_layer_similarity = (torch.einsum('BSD,BTD->BST', [normalized_text, normalized_visual]) / 0.07).masked_fill(mask, -6e4)
        similarity = F.softmax(last_layer_similarity, dim=-1)
        pred_time = similarity.argmax(-1)
        
        pred_similarity_list = []
        for i in range(out.shape[0]):
            normalized_text = F.normalize(self.t_sim_proj(out[i]), dim=-1)
            pred_similarity = (torch.einsum('ND,MD->NM', [normalized_text.reshape(-1, self.d_proj), normalized_visual.reshape(-1, self.d_proj)]) / 0.07)
            pred_similarity_list.append(pred_similarity)
        pred_similarity_list = torch.stack(pred_similarity_list, dim=0)

        return pred_similarity_list, pred_time
    
    def compute_crosstask(self, visual_input, visual_padding_mask, text_input, text_padding_mask):
        T, S = visual_input.shape[1], text_input.shape[1]

        visual_input = visual_input.transpose(0, 1)
        text_input = text_input.transpose(0, 1)

        visual_embed = self.visual_proj(visual_input)
        visual_pos_enc = self.visual_pos_enc.pe[:, :T, :].transpose(0, 1)

        text_embed = self.text_proj(text_input)
        text_pos_enc = self.text_pos_enc.weight[:S, None, :]

        visual_embed = self.visual_encoder(
            visual_embed, src_key_padding_mask=visual_padding_mask, pos=visual_pos_enc)

        if self.use_text_pos:
            out = self.decoder(text_embed, visual_embed, tgt_key_padding_mask=text_padding_mask,
                            memory_key_padding_mask=visual_padding_mask, pos=visual_pos_enc,
                            query_pos=text_pos_enc).transpose(1, 2)
        else:
            out = self.decoder(text_embed, visual_embed, tgt_key_padding_mask=text_padding_mask,
                           memory_key_padding_mask=visual_padding_mask, pos=visual_pos_enc,
                           query_pos=None).transpose(1, 2)

        normalized_text = F.normalize(self.t_sim_proj(out[-1]), dim=-1)
        normalized_visual = F.normalize(self.v_sim_proj(visual_embed).transpose(0, 1), dim=-1)
        mask = torch.logical_or(visual_padding_mask[:, None, :].repeat(1, S, 1), text_padding_mask[:, :, None].repeat(1, 1, T))
        last_layer_similarity = (torch.einsum('BSD,BTD->BST', [normalized_text, normalized_visual]) / 0.07).masked_fill(mask, -6e4)
        similarity = F.softmax(last_layer_similarity, dim=-1)

        return similarity, last_layer_similarity

    @torch.no_grad()
    def compute_input_similarity(self, visual_input, visual_padding_mask, text_input, text_padding_mask):

        T, S = visual_input.shape[1], text_input.shape[1]

        normalized_text_input = F.normalize(text_input, dim=-1).squeeze(1)
        normalized_visual_input = F.normalize(visual_input, dim=-1).squeeze(1)
        
        mask = torch.logical_or(visual_padding_mask[:, None, :].repeat(1, S, 1), text_padding_mask[:, :, None].repeat(1, 1, T))
        input_similarity = (torch.einsum('BSD,BTD->BST', [normalized_text_input, normalized_visual_input]) * self.t).masked_fill(mask, -6e4).softmax(-1)
        input_time = input_similarity.argmax(-1)

        return input_time
