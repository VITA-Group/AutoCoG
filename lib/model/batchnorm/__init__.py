from .CBN import GnnCBN
from .others import pair_norm, mean_norm, group_norm
import torch.nn as nn

BN_TYPE = ['batch', 'group', 'pair', 'mean', 'condition']


class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x


class BatchNorm1D(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super(BatchNorm1D, self).__init__(*args, **kwargs)

    def forward(self, x):
        return super(BatchNorm1D, self).forward(x)

def construct_bn_from_cfg(cfg, layers):
    bn = nn.ModuleList()
    for l in layers:
        bn_type = cfg.MODEL.BATCHNORM.TYPE
        assert bn_type in BN_TYPE
        if l.concat:
            c_out = l.c_out * l.num_head
        else:
            c_out = l.c_out
        bn.append(construct_bn(cfg, c_out))
    return bn

def construct_bn(cfg, c, num_head=None):
    bn = None
    if cfg.MODEL.BATCHNORM.ENABLED:
        bn_type = cfg.MODEL.BATCHNORM.TYPE
        if bn_type in BN_TYPE:
            if bn_type == "batch":
                bn = BatchNorm1D(c)
            elif bn_type == 'mean':
                bn = mean_norm()
            elif bn_type == 'pair':
                bn = pair_norm()
            elif bn_type == 'group':
                bn = group_norm(dim_hidden=c,
                                num_groups=cfg.MODEL.BATCHNORM.NUM_GROUP,
                                skip_weight=cfg.MODEL.BATCHNORM.RATIO)
            else:
                raise NotImplementedError
    else:
        return nn.Identity()
    return bn
