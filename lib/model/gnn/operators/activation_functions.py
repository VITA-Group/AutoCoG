import torch.nn.functional as F
import torch
from collections import OrderedDict as odict
OPS = odict({'linear': lambda x: x,
       'elu': F.elu, 'sigmoid': F.sigmoid, 'tanh': torch.tanh, 'relu': F.relu, 'relu6': F.relu6,
       'softplus': F.softplus, 'leaky_relu': F.leaky_relu, 'swish': F.hardswish})
