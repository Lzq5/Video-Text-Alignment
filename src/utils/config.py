import argparse
import os
import ast

import yaml
from easydict import EasyDict


BASE_CONFIG = {
    'seed': 2023,
    'output_dir': './output',
}


def load_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        default='configs/similarity.yaml',
        type=str,
        help='path to yaml config file',
    )
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='gpu device id',
    )
    parser.add_argument(
        '--run_name',
        default='debug',
        type=str,
        help='special tag for each execution',
    )
    parser.add_argument(
        '--verbose',
        default='False',
        action='store_true',
    )
    args = parser.parse_args()

    BASE_CONFIG.update(args.__dict__)
    cfg = EasyDict(BASE_CONFIG)

    # merge config from yaml file
    with open(cfg.config_file, 'r') as f:
        yaml_config = yaml.full_load(f)
    cfg.update(yaml_config)

    if cfg.model.checkpoint != '':
        cfg.use_wandb = False

    # set the output path
    cfg.output_dir = os.path.join(
        cfg.output_dir, cfg.project_name, cfg.run_name)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # eval dict leaf
    cfg = eval_dict_leaf(cfg)
    return cfg


def eval_dict_leaf(d):
    """Eval values of dict leaf.
    Args:
        d (dict): The dict to eval.
    Returns: dict.
    """
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = eval_string(v)
        else:
            eval_dict_leaf(v)
    return d


def eval_string(string):
    """Automatically evaluate string to corresponding types.
    For example:
        not a string -> return the original input
        '0' -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0, 1, 2]
        'eval(1 + 2)' -> 3
        'eval(range(5))' -> [0, 1, 2, 3, 4]
    Args:
        string (string): the string.
    Returns: the corresponding type
    """
    if not isinstance(string, str):
        return string
    if len(string) > 1 and string[0] == '[' and string[-1] == ']':
        return eval(string)
    if string[0:5] == 'eval(':
        return eval(string[5:-1])
    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v