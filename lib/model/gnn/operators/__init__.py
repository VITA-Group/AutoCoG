from .attention_functions import OPS as att_ops
from .activation_functions import OPS as act_ops
from .combine_functions import OPS as comb_ops
from .data_functions import OPS as data_ops
#
import torch
from typing import Union
from easydict import EasyDict as edict
from yacs.config import CfgNode as CN
from collections import OrderedDict as odict
from collections import defaultdict as ddict
from typing import Dict


def check_decode(arch):
    for layer, settings in arch.items():
        for op_type, ops in settings.items():
            total = sum([v for k, v in ops.items()])
            assert total == 1.0, f"{op_type}, {total} "


def dart_decode(arch: Dict[str, torch.tensor], cfg: Union[edict, CN]):
    # torch.tensor shape: [layer x options]
    # except residual shape [layer+1, layer+1, options]
    settings = odict(cfg.NAS)
    settings = odict(sorted(settings.items(), key=lambda x: len(x[-1]), reverse=True))
    ret = ddict(dict)
    for name, options in settings.items():
        if name.lower() == 'residual':
            continue
        weights = arch[f"arch_{name.lower()}"]
        if name.lower() != 'residual':
            for l in range(weights.shape[0]):
                ret[l][name.lower()] = {}
                for o, opt in enumerate(options):
                    ret[l][name.lower()][opt] = weights[l][o]
        else:
            row, col= weights.shape

            for r in range(row):
                ret[r][name.lower()] = {}
                for p in range(r):
                    w = weights[r][p]
                    # for o, opt in enumerate(options):
                    ret[r][name.lower()][p] = w

    # check_decode(ret)
    return ret

def gdas_decode(alpha: torch.tensor , cfg: Union[edict, CN]):
    settings = odict(cfg.NAS)
    settings = odict(sorted(settings.items(), key=lambda x: len(x[-1]), reverse=True))

    ret = {}
    al_idx = idx = alpha.argmax()
    al_w = alpha[al_idx]
    for name, values in settings.items():
        n = len(values)
        sel =  values[al_idx % n]
        idx = idx // n
        ret[name.lower()] = sel

    ret['arch'] = {}
    ret['arch']['index'] = al_idx
    ret['arch']['weight'] = al_w

    return ret
