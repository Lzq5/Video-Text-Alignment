import torch
import os.path as osp
import pprint


class Checkpointer(object):

    def __init__(self, cfg, device):

        # Load pretrained checkpoint
        self.ckpt = self._load_ckpt(cfg.model.checkpoint, device)
        self.output_dir = cfg.output_dir

    def load(self, model, optimizer=None):
        if self.ckpt is not None:
            model_weights = self.ckpt['model_state_dict']

            [missing, unexpected] = model.load_state_dict(
                model_weights, strict=False)
            missing_params = set([v.split('.', 1)[0] for v in missing])
            unexpected_params = set([v.split('.', 1)[0] for v in unexpected])
            loaded_params = set([v.split('.', 1)[0]
                                for v in model_weights.keys()]) - unexpected_params

            print("\n====== Loading Model Parameters ======")
            loading_info = "Loaded: " +  str(list(loaded_params))[1:-1] + \
                           "\nMissed: " +  str(list(missing_params))[1:-1] + \
                           "\nUnexpected: " + str(list(unexpected_params))[1:-1]
            print(loading_info)

            if optimizer is not None:
                if optimizer.__class__.__name__ != self.ckpt['optimizer_name']:
                    print("====== Not Loading Optimizer Parameters ======")
                else:
                    print("====== Loading Optimizer Parameters ======")
                    optimizer.load_state_dict(
                        self.ckpt['optimizer_state_dict'])

    def save(self, epoch, model, optimizer):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(self.output_dir, 'epoch-{}.pth'.format(epoch)))

    def _load_ckpt(self, checkpoint, device):
        if checkpoint is not None and osp.isfile(checkpoint):
            return torch.load(checkpoint, map_location=device)
        return None