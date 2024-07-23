from .dataset import *

from pprint import pprint
import torch.utils.data as data


def build_dataloader(cfg, phase):

    if cfg.dataset.name == 'htm':
        if phase == 'htm-align':
            dataset = HTMAlignDataset(cfg)
        elif phase == 'ht-step':
            dataset = HTMStepDataset(cfg)
        else:
            dataset = HTMDataset(cfg)
    else:
        dataset = registry.get_dataset_class(cfg.dataset.name)(cfg, phase)

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True if phase == "train" else False, # according to video length
        drop_last=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return loader
