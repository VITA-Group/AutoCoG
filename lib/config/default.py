from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN

from .state_space import CONFIG

_C = CN()
_C.RESUME = False
_C.IS_SEARCH = False
_C.SAVE_DIR = './output/'
_C.CHECKPOINT_FREQ = 100
_C.PRINT_FREQ = 50
_C.LOG_WANDB = False
_C.ARGS = CN()
#
_C.DATA = CN()
D = _C.DATA
D.NAME = 'Cora'  #
D.MODE = 'policy'
D.PATH = ''
#
D.NAS = CONFIG['data_nas']
D.IS_SEARCH = False
#
_C.MODEL = CN()
M = _C.MODEL
M.ONEHOT = False
M.WEIGHT_TRANSFER = ''
M.BATCHNORM = CN()
BN = M.BATCHNORM
BN.ENABLED = False
BN.TYPE = 'batch'  # [batch, bcn, group, pair]
BN.HIDDEN_DIM = 32
BN.NUM_GROUP=10
BN.RATIO=0.001
M.RESIDUAL = False
M.NUM_LAYERS = 2
M.DROPOUT = 0.6
M.N_HEADS = 3  #
M.HIDDEN_DIM = 32
M.CLIP_NORM = 0.0
M.NEGATIVE_SLOPE = 0.2
M.ARCHITECTURE = ''
M.EDGE_WEIGHT = ''
M.BIAS = True
M.STABILIZER = False
M.ALLOW_AUG = False
M.AUG_FIRST=False
M.STD = 0.3
M.RANDOM_SEARCH = False
M.SIMPLE = False
M.SIMPLE_GROUP = 2
M.CONCAT = False
# M.CLASS_EMB = 8
M.FORWARD_WEIGHT = False
M.LAMBDA = 0.5
M.ALPHA = 0.1
M.NAS_CFG_FILE = ""
M.MIX_DATA_POLICY = False
M.ID_MAP = True
M.ALLOW_MASK = False
M.MASK_P = 5
#
M.NAS = CONFIG['model_nas']
_C.GRAPH = CN()
G = _C.GRAPH
# G.NUM_VIEW = 4
# G.EPS = 0.5
# G.LAMBDA = 0.5  #0.4
# G.MU = 0.5  #0.8
# G.ANCHOR = True
# G.ANCHOR_SIZE = 1000
# G.BERN = False
# G.ONEHOT = False
# G.LR = 0.0001
G.WITH_SEARCH = False
G.WITH_TRAIN = False
# G.SMOOTHNESS_RATIO = 0.1
# G.DEGREE_RATIO = 0.0
# G.SPARSITY_RATIO = 0.1
# G.HIDDEN = 16
# G.INCLUDE_ORG = True
# G.FORWARD_GRAPH = True
# G.EDGE_WEIGHT = ''
# G.EDGE_INDEX = ''

_C.TRAIN = CN()
S = _C.TRAIN
S.TAU = [10.0, 0.1]
S.TAU_DECAY_FACTOR = 0.99
S.END_EPOCH = 100
S.SEARCH_EPOCH = 600
S.TRAIN_EPOCH = 10000
S.BEGIN_EPOCH = 0
S.GRAPH_BEGIN_EPOCH = 0
S.LR_MASK = 1e-2
S.LR = 1e-3
S.LR_ARCH = 5e-4
S.WEIGHT_DECAY = 5e-4
S.WEIGHT_DECAY_1 = 0.01
S.WEIGHT_DECAY_2 = 5e-4
S.PATIENT = 1000
S.TRAIN_PATIENCE=100
S.SEARCH_PATIENCE=1000
#


def update_config(cfg, arg):
    cfg.defrost()
    cfg.merge_from_file(arg.cfg)
    cfg.merge_from_list(arg.opts)

    if 'search' in arg:
        cfg.MODEL.IS_SEARCH = arg.search
        # cfg.MODEL.RANDOM_SEARCH = arg.random_arch
    cfg.freeze()


def get_model_name(cfg):
    return 'default', 'default'

def convert_to_dict(cfg_node, key_list=[]):
    """ Convert a config node to dictionary """
    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
        cfg_dict[k] = v
    return cfg_dict