import argparse
import json
import math
import os
import os.path as osp
import random
import numpy as np

import pandas as pd
from tqdm import tqdm
import sys
import torch
import torch.nn.functional as F

sys.path.append('src')
from InternVideo import load_model, tokenize


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    random.seed(300)

    device = torch.device(f"cuda:{args.gpu}")

    lm = load_model("./InternVideo/Pretrain/Multi-Modalities-Pretraining/models/InternVideo-MM-L-14.ckpt").to(device)
    lm.requires_grad_(False)

    #############################################
    # pre-process htm text feature
    # text_root = "/remote-home/share/zeqianli/HowTo100M/S3_whisperX_large_v2/S3_whisperX_large_v2"
    # save_root = "/remote-home/share/zeqianli/HowTo100M/whisperX_InternVideo/"

    # sub_folders = sorted(os.listdir(text_root))
    # for sub_folder in tqdm(sub_folders[32:]): #32
    #     json_data = {}
    #     os.makedirs(osp.join(save_root, sub_folder), exist_ok=True)

    #     for asr_file in tqdm(os.listdir(osp.join(text_root, sub_folder))):
    #         vid = asr_file[:-len('.json')]
    #         asr = json.load(open(osp.join(text_root, sub_folder, asr_file)))
    #         segment_list = asr['segments']

    #         start = [segment['start'] for segment in segment_list]
    #         end = [segment['end'] for segment in segment_list]
    #         text = [segment['text'].strip() for segment in segment_list]

    #         with torch.no_grad():
    #             tokens = tokenize(text, truncate=True).to(device)
    #             features = lm.encode_text(tokens)

    #         if features.shape[0] == 0:
    #             continue

    #         save_dict = {
    #             "vid": vid,
    #             "start": start,
    #             "end": end,
    #             "text": text,
    #             "features": features,
    #         }
    #         torch.save(save_dict, osp.join(save_root, sub_folder, f'{vid}.pth'))

    #         json_dict = {
    #             "vid": vid,
    #             "pos_anno": list(zip(start, end, text)),
    #         }
    #         json_data[vid] = json_dict

    #     with open(osp.join(text_root, sub_folder, f"{sub_folder}.json"), "w") as fp:
    #         json.dump(json_data, fp)

    #############################################
    # pre-process htm-align text feature
    # htm_align_path = "/remote-home/share/zeqianli/HowTo100M/TAN/HTM-Align/htm_align.json"
    # htm_align_save_path = "/remote-home/share/zeqianli/HowTo100M/TAN/HTM-Align/text_internvideo"
    # os.makedirs(htm_align_save_path, exist_ok=True)
    # anno_data = json.load(open(htm_align_path))

    # for vid, anno in tqdm(anno_data.items()):
    #     sign = list(map(lambda x: x[0], anno))
    #     start = list(map(lambda x: x[1], anno))
    #     end = list(map(lambda x: x[2], anno))
    #     text = list(map(lambda x: x[3].strip(), anno))

    #     with torch.no_grad():
    #         tokens = tokenize(text, truncate=True).to(device)
    #         features = lm.encode_text(tokens)

    #     save_dict = {
    #         "vid": vid,
    #         "start": start,
    #         "end": end,
    #         "text": text,
    #         "features": features,
    #     }
    #     torch.save(save_dict, osp.join(htm_align_save_path, f'{vid}.pth'))

    #############################################
    # pre-process htm-step text feature
    htm_step_path = "/remote-home/share/zeqianli/HowTo100M/HT-Step/test_input_headline.json"
    htm_step_save_path = "/remote-home/share/zeqianli/HowTo100M/HT-Step/text_internvideo"
    os.makedirs(htm_step_save_path, exist_ok=True)
    anno_data = json.load(open(htm_step_path))

    for vid, anno in tqdm(anno_data.items()):

        text = anno["step_headline"]

        with torch.no_grad():
            tokens = tokenize(text, truncate=True).to(device)
            features = lm.encode_text(tokens)

        save_dict = {
            "vid": vid,
            "text": text,
            "features": features,
        }
        torch.save(save_dict, osp.join(htm_step_save_path, f'{vid}.pth'))