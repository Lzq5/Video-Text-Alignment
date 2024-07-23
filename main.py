import sys
# sys.path.insert(0, 'src')
import os
import wandb
import torch
from pprint import pprint
import ipdb
from src.dataset import build_dataloader
from src.model import build_model
from src.trainer import build_optimizer, build_scheduler, build_criterion
from src.trainer.train import do_train
from src.utils.config import load_configs
from src.utils.utils import set_random_seed, set_device
from src.utils.checkpointer import Checkpointer
from src.utils.logger import Logger


def main():

    # load configs
    cfg = load_configs()
    # ipdb.set_trace()
    
    sys.stdout = Logger(os.path.join(cfg.output_dir,'log.txt'))

    # save the current config
    with open(os.path.join(cfg.output_dir, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    # set random seed
    set_random_seed(cfg.seed)

    # set device & logger & checkpointer
    device = set_device(cfg.gpu)
    checkpointer = Checkpointer(cfg, device)

    # print & log configs
    if cfg.use_wandb:
        wandb.init(project=cfg.project_name,
                name=cfg.run_name,
                config=cfg,
                mode="online"
                )
    pprint(cfg)

    # build data loaders
    data_loaders = {
        phase: build_dataloader(cfg, phase)
        for phase in cfg.dataloader.phases
    }
    # ipdb.set_trace()
    # build model
    model = build_model(cfg, device)

    # ckpt = torch.load("/remote-home/share/qiruichen/projects/Alignment/output/htm-align/task-concat-debug/epoch-12.pth", map_location="cpu")['model_state_dict']
    # model.load_state_dict(ckpt)
    # model = model.to(device)

    # build criterion
    criterion = build_criterion(cfg, device)

    # set optimizer & lr scheduler
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(data_loaders['train']))

    # load pre-trained model weights
    checkpointer.load(model, optimizer=None)

    do_train(
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpointer,
    )


if __name__ == '__main__':

    main()
