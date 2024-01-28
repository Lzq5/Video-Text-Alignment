import os
import glob
import os.path as osp
import json
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from ..utils.registry import registry


def pad_sequence_by_last(sequences):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].new_full(out_dims, 0.0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor
        out_tensor[i, length:, ...] = tensor[-1, ...]
    return out_tensor


def pad_mask_matrix(sequences):
    max_texts = max([s.size(0) for s in sequences])
    max_frames = max([s.size(1) for s in sequences])

    out_tensor = []
    for mask_matrix in sequences:
        num_texts, num_frames = mask_matrix.shape[:]
        padded_mask_matrix = F.pad(mask_matrix, 
        (0, max_frames-num_frames, 0, max_texts-num_texts),
        "constant",
        False
        )
        out_tensor.append(padded_mask_matrix)

    return torch.stack(out_tensor, dim=0)

@registry.register_dataset('htm-align')
class HTMAlignDataset(data.Dataset):
    def __init__(self, cfg=None):
        if cfg.dataset.visual_backbone == 'internvideo':
            self.clip_root = '/DB/data/zeqianli/datasets/HowTo100M/TAN/HTM-Align/InternVideoFeature'
            self.text_root = '/DB/data/zeqianli/datasets/HowTo100M/TAN/HTM-Align/text_internvideo'
        anno_data = json.load(open('/DB/data/zeqianli/datasets/HowTo100M/TAN/HTM-Align/htm_align.json'))
        
        self.anno_data = []
        for vid, anno in anno_data.items():
            visual_path = glob.glob(self.clip_root + f'/{vid}.pth.tar')
            text_path = glob.glob(self.text_root + f'/{vid}.pth')
            if len(visual_path) != 0 and len(text_path) != 0 and len(list(filter(lambda x: x[0] == 1, anno))): # 20th video only has unalignable texts
                self.anno_data.append([vid, visual_path[0], text_path[0], anno])

    def create_target_similarity(self, start, end, T):
        
        start = torch.tensor(start)[:, None] 
        end = torch.tensor(end)[:, None]
        time_axis = torch.arange(T)[None, :] 
        time_mask = ((time_axis >= start) * (time_axis <= end)).bool()

        return time_mask

    def __getitem__(self, index):
        """
        visual: (T, 768)
        text: (S=#pos+#neg, max_len=32)
        target_sign: (S,)
        target_time: (S,)
        """

        [vid, visual_path, text_path, anno] = self.anno_data[index]

        visual_input = torch.load(visual_path, map_location='cpu').float()  
        visual_padding_mask = torch.zeros(visual_input.size(0)).bool()

        idxs = list(range(len(anno)))

        target_sign = list(map(lambda x: x[0], anno))
        start = list(map(lambda x: x[1], anno))
        end = list(map(lambda x: x[2], anno))
        raw_texts = list(map(lambda x: x[3], anno))

        text_info = torch.load(text_path, map_location='cpu')
        text_input = text_info["features"][idxs].float()
        text_padding_mask = torch.zeros(text_input.size(0)).bool()

        time_mask = self.create_target_similarity(start, end, visual_input.shape[0])

        input_dict = {
            "visual_input": visual_input,
            "visual_padding_mask": visual_padding_mask,
            "raw_text": raw_texts,
            "text_input": text_input,
            "text_padding_mask": text_padding_mask,
            "target_sign": torch.tensor(target_sign, dtype=torch.long),
            "time_mask": time_mask,
        }

        return input_dict

    def __len__(self):
        return len(self.anno_data)

    @staticmethod
    def collate_fn(batch):
        """
        pad visual_input, target
        """
        input_batch = {}
        input_batch['visual_input'] = pad_sequence_by_last(
            [sample['visual_input'] for sample in batch])
        input_batch['visual_padding_mask'] = pad_sequence(
            [sample['visual_padding_mask'] for sample in batch], batch_first=True, padding_value=1)

        input_batch['raw_text'] = [sample['raw_text'] for sample in batch]
        input_batch['text_input'] = pad_sequence_by_last([sample['text_input']
                                      for sample in batch])
        input_batch['text_padding_mask'] = pad_sequence(
            [sample['text_padding_mask'] for sample in batch], batch_first=True, padding_value=1)

        input_batch['target_sign'] = pad_sequence(
            [sample['target_sign'] for sample in batch], batch_first=True, padding_value=-100)
        input_batch['time_mask'] = pad_mask_matrix([sample['time_mask'] for sample in batch])

        return input_batch


@registry.register_dataset('htm')
class HTMDataset(data.Dataset):
    def __init__(self, cfg=None):

        self.nar_rand_exp_factor = 0
        self.max_vlen = 1200
        self.shuffle_text = cfg.dataset.text_shuffle

        HTSTEP_data = json.load(open('/DB/data/zeqianli/datasets/HowTo100M/HT-Step/test_input_headline.json'))
        HTSTEP_vids = list(HTSTEP_data.keys())
        htm370k_vid_path = '/DB/data/zeqianli/datasets/HowTo100M/TAN/Sentencified-HTM/HTM-370K/htm370k_vids.txt'
        vid_to_folder_path = '/DB/data/zeqianli/datasets/HowTo100M/vid_to_folder.json'
        if cfg.dataset.visual_backbone == 'internvideo':
            self.clip_root = '/DB/data/zeqianli/datasets/HowTo100M/InternVideo_feature/internvideo_MM_L14/'
            if cfg.dataset.text_type == 'whisperx':
                self.text_root = '/DB/data/zeqianli/datasets/HowTo100M/whisperX_InternVideo/'
                path_record = 'data/htm370k.pth'
            elif cfg.dataset.text_type == 'step':
                self.text_root = '/DB/data/zeqianli/datasets/HowToStep/Steps370k_round2/internvideo_round1_0.15_round2_duration_0.8_8_start'
                step_param = self.text_root.split('/')[-2] + '_' + self.text_root.split('/')[-1]
                path_record = 'data/{}.pth'.format(step_param)

        # text timestamp path
        with open(htm370k_vid_path, 'r') as f:
            htm370k_vid = f.readlines()

        skip_list = ['7DF2yD9-tkg']
        vid_to_folder = json.load(open(vid_to_folder_path))

        if osp.exists(path_record):
            
            if cfg.dataset.mix_dataset == False:
                ## only whisperx or howtostep
                self.anno_data = torch.load(path_record, map_location='cpu')
            else:
                ## use whisperx and howtostep
                anno_data1 = torch.load(path_record, map_location='cpu')
                if cfg.dataset.visual_backbone == 'internvideo':
                    anno_data2 = torch.load("data/htm370k.pth", map_location='cpu')
                self.anno_data = anno_data1 + anno_data2

        else:
            anno_data = []
            for vid in tqdm(htm370k_vid, desc="Reading Video Path"):
                vid = vid.strip()
                if (vid not in vid_to_folder) or (vid in skip_list) or (vid in HTSTEP_vids):
                    continue
                else:
                    sub_folder = vid_to_folder[vid]
                    visual_path = osp.join(self.clip_root, sub_folder, f"{vid}.pth.tar")
                    text_path = osp.join(self.text_root, sub_folder, f"{vid}.pth")
                    if osp.exists(visual_path) and osp.exists(text_path):
                        anno_data.append([vid, visual_path, text_path])
            torch.save(anno_data, path_record)
            
            if cfg.dataset.mix_dataset == False:
                ## only whisperx or howtostep
                self.anno_data = anno_data
            else:
                ## use whisperx and howtostep
                anno_data1 = anno_data
                if cfg.dataset.visual_backbone == 'internvideo':
                    anno_data2 = torch.load("data/htm370k.pth", map_location='cpu')
                self.anno_data = anno_data1 + anno_data2
        print("==== {} Samples ====".format(len(self.anno_data)))


    def create_target_similarity(self, start, end, T, caption=False):
        
        start = torch.tensor(start)[:, None]
        end = torch.tensor(end)[:, None]

        time_axis = torch.arange(T)[None, :]
        time_mask = ((time_axis >= start) * (time_axis <= end)).bool()

        return time_mask

    def __getitem__(self, index):

        [vid, visual_path, text_path] = self.anno_data[index]

        visual_input = torch.load(visual_path, map_location='cpu').float()

        text_info = torch.load(text_path, map_location='cpu')
        start = text_info["start"]
        end = text_info["end"]
        raw_texts = text_info["text"]
        idxs = list(range(len(start)))
        anno_tuple = list(zip(idxs, start, end, raw_texts))

        anno_tuple = list(filter(lambda x: x[2]<self.max_vlen, anno_tuple))
        visual_input = visual_input[:self.max_vlen]

        if len(anno_tuple) == 0:
            idx = np.random.randint(0, len(self)-1)
            return self.__getitem__(idx)
        
        if self.shuffle_text:
            random.shuffle(anno_tuple)

        idxs, start, end, raw_texts = map(list, zip(*anno_tuple))

        text_input = text_info["features"][idxs].float()
        text_padding_mask = torch.zeros(text_input.size(0)).bool()

        if_caption = True if "HowToCaption" in text_path else False
        time_mask = self.create_target_similarity(start, end, visual_input.shape[0], if_caption)

        visual_padding_mask = torch.zeros(visual_input.size(0)).bool()
        input_dict = {
            "visual_input": visual_input,
            "visual_padding_mask": visual_padding_mask,
            "raw_text": raw_texts,
            "text_input": text_input,
            "text_padding_mask": text_padding_mask,
            "time_mask": time_mask,
        }

        while input_dict["text_input"].shape[0] == 0:
            idx = np.random.randint(0, len(self)-1)
            input_dict = self.__getitem__(idx)

        return input_dict

    def __len__(self):
        return len(self.anno_data)

    @staticmethod
    def collate_fn(batch):

        input_batch = {}
        input_batch['visual_input'] = pad_sequence_by_last(
            [sample['visual_input'] for sample in batch])
        input_batch['visual_padding_mask'] = pad_sequence(
            [sample['visual_padding_mask'] for sample in batch], batch_first=True, padding_value=1)
            
        input_batch['raw_text'] = [sample['raw_text'] for sample in batch]
        input_batch['text_input'] = pad_sequence_by_last([sample['text_input']
                                      for sample in batch])
        input_batch['text_padding_mask'] = pad_sequence(
            [sample['text_padding_mask'] for sample in batch], batch_first=True, padding_value=1)

        input_batch['time_mask'] = pad_mask_matrix([sample['time_mask'] for sample in batch])

        return input_batch


@registry.register_dataset('htm-step')
class HTMStepDataset(data.Dataset):
    def __init__(self, cfg=None):

        if cfg.dataset.visual_backbone == 'internvideo':
            self.clip_root = '/DB/data/zeqianli/datasets/HowTo100M/InternVideo_feature/internvideo_MM_L14'
            self.text_root = '/DB/data/zeqianli/datasets/HowTo100M/HT-Step/text_internvideo'
        anno_data = json.load(open('/DB/data/zeqianli/datasets/HowTo100M/HT-Step/test_input_headline.json'))
        
        self.anno_data = []
        for vid, anno in anno_data.items():
            visual_path = glob.glob(osp.join(self.clip_root + f'/**/{vid}.pth.tar'))
            text_path = glob.glob(osp.join(self.text_root + f'/{vid}.pth'))
            if len(visual_path) != 0 and len(text_path) != 0:
                self.anno_data.append([vid, visual_path[0], text_path[0], anno])

    def create_target_similarity(self, start, end, T):
        
        start = torch.tensor(start)[:, None]
        end = torch.tensor(end)[:, None]
        time_axis = torch.arange(T)[None, :]
        time_mask = ((time_axis >= start) * (time_axis <= end)).bool()

        return time_mask

    def __getitem__(self, index):

        [vid, visual_path, text_path, anno] = self.anno_data[index]

        visual_input = torch.load(visual_path, map_location='cpu').float()
        visual_padding_mask = torch.zeros(visual_input.size(0)).bool()
        raw_texts = anno["step_headline"]

        text_info = torch.load(text_path, map_location='cpu')
        text_input = text_info["features"].float()
        text_padding_mask = torch.zeros(text_input.size(0)).bool()

        input_dict = {
            "vid": vid,
            "visual_input": visual_input,
            "visual_padding_mask": visual_padding_mask,
            "raw_text": raw_texts,
            "text_input": text_input,
            "text_padding_mask": text_padding_mask,
        }

        return input_dict

    def __len__(self):
        return len(self.anno_data)

    @staticmethod
    def collate_fn(batch):
        """
        pad visual_input, target
        """
        input_batch = {}
        input_batch['vid'] = [sample['vid'] for sample in batch]
        input_batch['visual_input'] = pad_sequence_by_last(
            [sample['visual_input'] for sample in batch])
        input_batch['visual_padding_mask'] = pad_sequence(
            [sample['visual_padding_mask'] for sample in batch], batch_first=True, padding_value=1)
        
        input_batch['raw_text'] = [sample['raw_text'] for sample in batch]
        input_batch['text_input'] = pad_sequence_by_last([sample['text_input']
                                      for sample in batch])
        input_batch['text_padding_mask'] = pad_sequence(
            [sample['text_padding_mask'] for sample in batch], batch_first=True, padding_value=1)

        return input_batch