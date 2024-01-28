from .decoder import *

from ..utils.registry import registry


def build_model(cfg, device):

    model = registry.get_model_class(cfg.model.name)(cfg).to(device)

    return model
