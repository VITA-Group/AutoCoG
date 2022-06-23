import collections
import logging
import math
import os
from typing import Callable
from typing import Union, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as tgu
from easydict import EasyDict as edict
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter
from yacs.config import CfgNode as CN

from .batchnorm import construct_bn_from_cfg, construct_bn
from .gnn.message_passing import DARTGnnConv
from .gnn.operators import act_ops
from .gnn.operators import dart_decode
from .gnn.operators import data_ops
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor

from torch_geometric.utils import add_remaining_self_loops

EXCEPTION = []

def coalesce(
    edge_index: Tensor,
    edge_attr: Optional[Union[Tensor, List[Tensor]]] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
    is_sorted: bool = False,
    sort_by_row: bool = True,
) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, List[Tensor]]]:
    """Row-wise sorts :obj:`edge_index` and removes its duplicated entries.
    Duplicate entries in :obj:`edge_attr` are merged by scattering them
    together according to the given :obj:`reduce` option.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor or List[Tensor], optional): Edge weights or multi-
            dimensional edge features.
            If given as a list, will re-shuffle and remove duplicates for all
            its entries. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        reduce (string, optional): The reduce operation to use for merging edge
            features (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"mul"`). (default: :obj:`"add"`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
        sort_by_row (bool, optional): If set to :obj:`False`, will sort
            :obj:`edge_index` column-wise.

    :rtype: :class:`LongTensor` if :attr:`edge_attr` is :obj:`None`, else
        (:class:`LongTensor`, :obj:`Tensor` or :obj:`List[Tensor]]`)
    """
    nnz = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[1 - int(sort_by_row)]
    idx[1:].mul_(num_nodes).add_(edge_index[int(sort_by_row)])

    if not is_sorted:
        idx[1:], perm = idx[1:].sort()
        edge_index = edge_index[:, perm]
        if edge_attr is not None and isinstance(edge_attr, Tensor):
            edge_attr = edge_attr[perm]
        elif edge_attr is not None:
            edge_attr = [e[perm] for e in edge_attr]

    mask = idx[1:] > idx[:-1]

    # Only perform expensive merging in case there exists duplicates:
    if mask.all():
        return edge_index if edge_attr is None else (edge_index, edge_attr)

    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index

    dim_size = edge_index.size(1)
    idx = torch.arange(0, nnz).sub_(mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, None, dim_size, reduce)
    else:
        edge_attr = [
            scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr
        ]

    return edge_index, edge_attr


class Architecture:
    def __init__(self, model: nn.Module, data: Planetoid, cfg: Union[CN,
                                                                     edict],
                 search_fn: Callable, loss_fn: Callable):
        self.cfg = cfg
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=cfg.TRAIN.LR_ARCH,
                                          weight_decay=3e-4)
        self.search_fn = search_fn
        self.loss_fn = loss_fn
        self.data = data
        self.logger = logging.getLogger()
        self.acc_valid = self.loss_valid = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self, *args, **kwargs):
        self.acc_valid, self.loss_valid = self.search_fn(self.cfg,
            self.model, self.data, self.loss_fn)
        self.optimizer.zero_grad()
        self.loss_valid.backward()
        self.optimizer.step()

    def decay(self):
        self.model.decay()

    def status(self):
        ret = {}
        ret['tau'] = self.model.tau
        ret.update(self.get_entropy())
        return ret

    def get_entropy(self):
        arch_state_dict = self.model.arch_state_dict()
        entropies = {}
        for k, v in arch_state_dict.items():
            prob = F.softmax(v, dim=-1)
            log_prob = F.log_softmax(v, dim=-1)
            entropies[k] = -(log_prob * prob).sum(-1,
                                                  keepdim=False).mean().item()
        return entropies

    def load_architecture(self, arch):
        if self.cfg.MODEL.RANDOM_SEARCH:
            with torch.no_grad():
                for v in self.model.arch_parameters():
                    v.copy_(torch.rand_like(v))
            return
        if isinstance(arch, str):
            assert os.path.isfile(arch)
            w = torch.load(arch)
        else:
            w = arch
        if self.cfg.MODEL.FORWARD_WEIGHT:
            self.model.load_state_dict(w, strict=False)
            glorot(self.model.final_fc.weight)
            zeros(self.model.final_fc.bias)
        else:
            _w = {}
            for k in w.keys():
                if 'arch' in k:
                    _w[k] = w[k]
            self.model.load_state_dict(_w, strict=False)
        # self.summary()

    def save_architecture(self, path: str):
        state_dict = self.model.arch_state_dict()
        torch.save(state_dict, path)

    def summary(self):
        architecture = self.model.arch_state_dict()

        model_architecture = collections.defaultdict(list)
        for k, v in architecture.items():
            if 'edges_alpha' in k:
                num_edges = self.model._softmax(True, v)[:, 0].sum()
                self.logger.info(f"gt edges dropped: {num_edges}")
            if 'edges_beta' in k:
                new_pair = self.model.edge_generator._get_new_pair(v)
                self.logger.info(f"new pair added: {len(new_pair)}")

            architecture[k] = self.model._softmax(True, v)
        model_struct = dart_decode(architecture, self.cfg.MODEL)


        for i, (layer, settings) in enumerate(model_struct.items()):
            for setting, v in settings.items():
                if setting != "residual":
                    for name, weight in v.items():
                        if weight > 0.0:
                            model_architecture[setting].append(name)
                else:
                    skip_connections = []
                    for j in range(layer):
                        for name, weight in v[j].items():
                            if weight > 0.0 and name is True:
                                skip_connections.append(j)
                    model_architecture[setting].append(skip_connections)
        for k, v in model_architecture.items():
            self.logger.info(f"{k}: {v}")





class Model(nn.Module):
    def __init__(self, c_in: int, c_out: int, cfg: Union[CN, edict],
                 num_edges: int, is_search: bool, **kwargs):
        super().__init__()
        self.logger = logging.getLogger()
        self._is_search = is_search
        self.num_edges = num_edges
        self.cfg = cfg
        self.c_in = c_in
        self.c_out = c_out
        self.data_name = cfg.DATA.NAME.lower()
        self.one_hot = cfg.MODEL.ONEHOT
        self.lambd = cfg.MODEL.LAMBDA
        settings = cfg.MODEL
        # non-nas settings
        self.residual = settings.RESIDUAL  # bool
        self.n_layers = settings.NUM_LAYERS  # int
        self.dropout = settings.DROPOUT  # float
        self.n_heads = settings.N_HEADS  # int
        self.n_states = self.total_states(cfg.MODEL.NAS)
        self.hidden_dim = settings.HIDDEN_DIM
        # nas settings
        self.allow_aug = cfg.MODEL.ALLOW_AUG
        self.tau_max, self.tau_min = self.cfg.TRAIN.TAU
        decay_factor = self.cfg.TRAIN.TAU_DECAY_FACTOR
        self.tau = self.tau_max
        self.tau_decay = (self.tau_max - self.tau_min) / (
            (cfg.TRAIN.SEARCH_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * decay_factor)

        self.std = self.cfg.MODEL.STD
        self.std_ratio = (10 * self.std - self.std) / (
            (cfg.TRAIN.SEARCH_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * 0.8)

        self.edge_weights_all = None
        self.final_embeddings = None

        if self._is_search:
            self.beta_residual = 1.0
            self.beta_decay = (1.0) / ((cfg.TRAIN.SEARCH_EPOCH - cfg.TRAIN.BEGIN_EPOCH) * 0.7)
        else:
            self.beta_residual = 0.0
            self.beta_decay = 0.0

        self.construct_model()
        self.init()


    def total_states(self, config: Union[CN, edict]) -> int:
        config = dict(config)
        ret = 1
        for name, states in config.items():
            ret *= len(states)
            setattr(self, name.lower(), states)
        return ret

    def init(self) -> None:
        for name, m in self.named_parameters():
            self.logger.debug(f"init {name} {m.shape}")
            if 'arch' in name:
                if name == 'arch_residual':
                    torch.nn.init.constant_(m, 0.1)
                else:
                    torch.nn.init.constant_(m, 1e-3)
            elif 'bias' in name:
                nn.init.zeros_(m)
            elif 'bn' in name:
                nn.init.uniform_(m, 0, 1)
            elif 'prelu' in name:
                pass
            else:
                glorot(m)
                # nn.init.xavier_normal_(m)

    def construct_model(self) -> None:
        # constructing architecture
        for name, options in self.cfg.MODEL.NAS.items():
            if name.lower() == 'residual':
                self.register_parameter(
                    f'arch_{name.lower()}',
                    torch.nn.Parameter(torch.empty(self.n_layers, self.n_layers)))
            else:
                # if not self.cfg.MODEL.SIMPLE:
                self.register_parameter(
                    f'arch_{name.lower()}',
                    torch.nn.Parameter(torch.empty(self.n_layers,
                                                   len(options))))
                setattr(self, f"_arch_{name.lower()}", None)
        #
        self.layers = nn.ModuleList()
        c_in = self.c_in
        hidden_dim = self.hidden_dim
        concat = True
        self.first_layer = nn.Linear(c_in, self.hidden_dim * self.n_heads)
        self.prelu = nn.ModuleList()
        c_in = hidden_dim * self.n_heads
        for i in range(self.n_layers):
            l = DARTGnnConv('add', c_in, hidden_dim, self.n_heads, 128,
                            concat, self.cfg)
            self.layers.append(l)
            if concat:
                c_in = hidden_dim * self.n_heads
            else:
                c_in = hidden_dim
            self.prelu.append(nn.PReLU(c_in))
        self.bn_layers = construct_bn_from_cfg(self.cfg, self.layers)
        self.final_fc = nn.Linear(c_in, self.c_out)




    def forward(self, x, edge, edge_weight=None, val=False):
        if self.is_search and self.training:
            if val:
                self.gumbel_softmax(self.one_hot)
            else:
                self.gumbel_softmax(self.one_hot,
                                    add_noise=self.cfg.MODEL.STABILIZER)
                self.increase()
                self.beta_residual = max(0 , self.beta_residual - self.beta_decay)
            output, final_embeddings = self._forward(x, edge, edge_weight)
            return output
        else:
            self.softmax(True)
            _beta = self.beta_residual
            self.beta_residual = 0
            output, final_embeddings = self._forward(x, edge, edge_weight)
            self.beta_residual = _beta
            return output

    def _forward(self,
                 x: torch.Tensor,
                 edge_index_all: torch.Tensor,
                 edge_weight=None):
        structures = dart_decode(self.get_current_arch(), self.cfg.MODEL)
        output = self.loop(
            x, edge_index_all, structures, edge_weight)

        self.final_embeddings= final_embeddings = output #layer_outputs[-1]
        final_embeddings = F.dropout(final_embeddings , p=self.dropout , training=self.training and not self._is_search)
        output = self.final_fc(final_embeddings)
        return output, final_embeddings

    def loop(self, x, edge_index_all, structures, edge_weights_all):
        x = x_init = F.relu(
            self.first_layer(
                F.dropout(x, p=self.dropout, training=self.training)))
        output = x
        layer_outputs = [x]
        sampled_edges = edge_index_all
        sampled_weights = edge_weights_all
        sampled_edges, sampled_weights = add_remaining_self_loops(
            sampled_edges, sampled_weights, 1.0, x.size(0))
        adj = SparseTensor(row=sampled_edges[0], col=sampled_edges[1], value=sampled_weights)
        adj_t = adj.t()
        for i, layer in enumerate(self.layers):
            output = F.dropout(output, p=self.dropout, training=self.training)
            # simple derivation means we use fewer arch settings that actually initialized for
            structure = structures[i]
            #########################################################
            if self.data_name == 'ppi':
                raise NotImplementedError
            else:
                beta = math.log(self.lambd / (i + 1) + 1)
                ######################################################
                l0 = None
                for k, v in structure['first_to_skip'].items():
                    if v > 0 and k:
                        l0 = x_init
                        if self.is_search:
                            l0 = l0 * v
                        break
                ######################################################
                output = layer(F.dropout(layer_outputs[-1], p=self.dropout, training=self.training), adj_t,
                               structure, l0, beta)
                if self.residual:
                    output = output + layer_outputs[-1]
                ######################################################
                output = self.bn_layers[i](output)
                for activation, weight in structure['act_type'].items():
                    if weight == 0: continue
                    output = act_ops.get(activation, self.prelu[i])(output)
                    if self.is_search:
                        output = output * weight
                    break
                #######################################################
            layer_outputs.append(output)

        for activation, weight in structures[len(self.layers)-1]['dense_comb'].items():
            if weight != 0:
                if activation == 'last':
                    pass
                elif activation == 'mean':
                    output = sum(layer_outputs)/len(layer_outputs)
                elif activation == 'max':
                    output = torch.stack(layer_outputs, dim=1)
                    output, _ = torch.max(output, dim=1)
                else:
                    output = sum(layer_outputs)
                ##############################
                if self.is_search:
                    output = output * weight
                break

        return output

    def gumbel_softmax(self, onehot=False, add_noise=False) -> None:
        for k, v in self.arch_state_dict().items():

            if add_noise:
                v = v + torch.empty(v.shape).normal_(mean=0.0,
                                                     std=self.std).to(v.device)

            k = k.split('.')
            name = k[-1][5::]

            k[-1] = f"_{k[-1]}"

            if 'residual' in name:
                pass
                # print(v)
                # setattr(self, k[-1], v)
            else:
                setattr(self, k[-1], self._gumbel_softmax(onehot, v))

    def _gumbel_softmax(self, onehot, arch):
        return F.gumbel_softmax(F.log_softmax(arch, dim=-1),
                                tau=self.tau,
                                hard=onehot,
                                dim=-1)

    def softmax(self, onehot=True):
        for k, v in self.arch_state_dict().items():
            # if k == 'residual':
            #     setattr(self , f"_{k}" , self._softmax(False , v).detach().clone())
            # else:
            k = k.split('.')
            k[-1] = f"_{k[-1]}"
            if 'residual' in k[-1]:
                pass
                # setattr(self , k[-1] , v)
            else:
                setattr(self, k[-1], self._softmax(onehot, v))

    def _softmax(self, onehot, arch):
        if onehot:
            max_idx = torch.argmax(arch, -1, keepdim=True)
            one_hot = torch.zeros_like(arch)
            one_hot.scatter_(-1, max_idx, 1)  # onehot but detach from graph
            return one_hot - arch.detach() + arch  # attach to gradient
        else:
            return F.softmax(arch, dim=-1)

    def decay(self):
        self.tau = max(self.tau_min, self.tau - self.tau_decay)

    def increase(self):
        self.std = min(10 * self.cfg.MODEL.STD, self.std + self.std_ratio)

    def get_current_arch(self):
        ret = {}
        for name, param in self.named_parameters():
            if 'arch' in name:
                if "residual" in name:
                    continue
                else:
                    ret[name] = getattr(self, f"_{name}")
        return ret

    def arch_state_dict(self):
        ret = {}
        for name, param in self.named_parameters():
            if 'arch' in name:
                ret[name] = param
        return ret  # {"arch_alpha": self.arch_alpha, "arch_beta": self.arch_beta}

    def arch_parameters(self):
        for name, param in self.named_parameters():
            if 'arch' in name:
                yield param

    def net_parameters(self):
        for name, m in self.named_parameters():
            if 'arch' in name: continue
            yield m

    def net_reg_parameters(self):
        return self.layers.parameters()

    def net_non_reg_parameters(self):
        return [*self.first_layer.parameters(), *self.final_fc.parameters()]

    @property
    def is_search(self):
        return self._is_search

    @is_search.setter
    def is_search(self, value):
        assert isinstance(value, bool)
        self._is_search = value



