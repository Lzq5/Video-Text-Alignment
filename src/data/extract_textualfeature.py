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

    lm = load_model("path_to_InternVideo-MM-L-14.ckpt_from_https://github.com/OpenGVLab/InternVideo").to(device)
    lm.requires_grad_(False)
    lm.eval()

    htm_step_path = "path_text_files"
    htm_step_save_path = "path_to_textual_feature"
    os.makedirs(htm_step_save_path, exist_ok=True)
    anno_data = json.load(open(htm_step_path))

    for vid, anno in tqdm(anno_data.items()):

        text = anno["step_headline"] # read data from text files

        with torch.no_grad():
            tokens = tokenize(text, truncate=True).to(device)
            features = lm.encode_text(tokens)

        save_dict = {
            "vid": vid,
            "text": text,
            "features": features,
        }
        torch.save(save_dict, osp.join(htm_step_save_path, f'{vid}.pth'))