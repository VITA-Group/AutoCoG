from yacs.config import CfgNode as CN

MODEL = CN()
MODEL.ATT_TYPE = [
    "gcn", "cos", "const", 'linear', 'generalized_linear', "gat", "gat_sym"]
# MODEL.FINAL_COMB = ['add', 'max', 'concat', 'mean']
MODEL.UPDATE_TYPE = ['identity']#, 'mlp']
MODEL.ACT_TYPE = [
    'relu', "sigmoid", "tanh", "leaky_relu", "elu", 'swish', 'prelu'
]
# MODEL.SKIP_TO_LAST = [False, True]
MODEL.FIRST_TO_SKIP = [False, True]
# MODEL.RESIDUAL = [False, True]
MODEL.DENSE_COMB = ['mean', 'max', 'sum', 'last']
# MODEL.DATA_POLICY = [
#     'edge_perturbation',
#     'node_drop',
#     'edge_add',
#     'none'
# ]
# MODEL.EDGE_PERTURBATION = [0.2, 0.3,0.4, 0.5,  0.6]
# MODEL.EDGE_ADD = [0.2, 0.3, 0.4, 0.5, 0.6]
# MODEL.NODE_DROP = [0.2, 0.3, 0.4, 0.5, 0.6]
# MODEL.ATTR_MASKING = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

MODEL.HIDDEN_DIM = [64]  #[128, 64, 32]#, 16 ,8]
MODEL.NUM_HEADS = [3, 2, 1]  #[6,9,3,12]#,3,1]

DATA = CN()
# DATA.DATA_POLICY = [
#     'edge_perturbation',
#     'node_drop',
#     'edge_add',
# ]
# DATA.EDGE_PERTURBATION = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# DATA.EDGE_ADD = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# DATA.NODE_DROP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CONFIG = {'model_nas': MODEL, 'data_nas': DATA}

# [gcn_n_layers, all combinations ] onehot
# -------------------------------------------
# [gcn_n_layers, 1, num_options] x 5 (continous) -> derive one-hot (option with highest probability)
#
#
