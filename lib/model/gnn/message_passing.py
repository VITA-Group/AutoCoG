import logging
from typing import Dict
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import SparseTensor
from yacs.config import CfgNode as CN

from .operators import att_ops
from .operators import gdas_decode

from torch_geometric.nn.inits import glorot

class DARTGnnConv(MessagePassing):
    def __init__(self, aggr: str, c_in: int, c_out: int, num_head: int,
                 pooling_dim: int, concat: bool, cfg: Union[CN, edict]):
        self.logger = logging.getLogger()
        self.logger.debug(
            msg=
            f"aggr {aggr}, c_in {c_in}, c_out {c_out}, num_head {num_head}, pooling_dim {pooling_dim}, concat {concat}"
        )
        super().__init__(aggr=aggr, node_dim=0)
        self.c_in, self.c_out, self.num_head = c_in, c_out, num_head
        self.concat = concat
        self.pooling_dim = pooling_dim
        self.cfg = cfg
        self.dataset = cfg.DATA.NAME.lower()
        self._structure = None
        # nas settings
        self.search_space = ss = cfg.MODEL.NAS
        self.att_type = ss.ATT_TYPE  # list of string
        self.update_type = ss.UPDATE_TYPE  # list of str
        #
        self.negative_slope = cfg.MODEL.NEGATIVE_SLOPE
        self.dropout = cfg.MODEL.DROPOUT
        #
        #
        if self.concat:
            self.lin = nn.Linear(c_in, c_out * num_head, bias=False)
            self.weight = nn.Parameter(torch.ones(c_in, c_out*num_head), requires_grad=True)
        else:
            self.lin = nn.Linear(c_out, c_out , bias=False)
            self.weight = nn.Parameter(torch.ones(c_in, c_out), requires_grad=True)
        glorot(self.weight)
        # attention
        self.att = nn.ParameterDict()
        if cfg.DATA.NAME == 'ogbn-arxiv':
            common_att = nn.Parameter(torch.empty(1, num_head, 2 * c_out))
            for att_func in self.cfg.MODEL.NAS.ATT_TYPE:
                if att_func in ['cos', 'gat', 'gat_sym', 'linear', 'generalized_linear']:
                    self.att[att_func] = common_att#nn.Parameter(torch.empty(1, num_head, 2 * c_out))
        else:
            for att_func in self.cfg.MODEL.NAS.ATT_TYPE:
                if att_func in ['cos', 'gat', 'gat_sym', 'linear', 'generalized_linear']:
                    self.att[att_func] = nn.Parameter(torch.empty(1, num_head, 2 * c_out))
        if 'generalized_linear' in self.cfg.MODEL.NAS.ATT_TYPE:
            self.general_att_layer = nn.Linear(c_out, 1, bias=False)
        else:
            self.general_att_layer = None
        # self.att = nn.Parameter(torch.empty(1, num_head, 2 * c_out))
        self.alpha = 0.1
        # update
        bias = cfg.MODEL.BIAS
        if bias and concat:
            self.bias = Parameter(torch.Tensor(self.num_head * self.c_out))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(c_out))
        else:
            self.bias = None
        if 'mlp' in self.update_type:
            self.pool_layer = []
            if self.dataset == 'ppi' and concat:
                self.pool_layer.append(
                    torch.nn.Linear(self.c_out * num_head,
                                    128))
                self.pool_layer.append(torch.nn.ReLU(inplace=False))
                self.pool_layer.append(
                    torch.nn.Linear(128,
                                    self.c_out * num_head))
            else:
                self.pool_layer.append(
                    torch.nn.Linear(self.c_out, 128))
                self.pool_layer.append(torch.nn.ReLU(inplace=False))
                self.pool_layer.append(
                    torch.nn.Linear(128, self.c_out))
            self.pool_layer = nn.Sequential(*self.pool_layer)

    def forward(self, x: torch.tensor, adj_t:SparseTensor,
                structure: Dict[str, Dict[str, Dict[str, torch.tensor]]],
                h0: torch.tensor, beta: float):
        """
        :param x: node x features
        :param edge_index: 2 x edges
        :param alpha: 1 x N options
        """
        self._structure = structure
        # x = self.lin(x)
        if 'gcn' in structure['att_type'] and structure['att_type']['gcn'] > 0:
            support = self.propagate(edge_index=adj_t,
                                     x=x,
                                     num_nodes=x.size(0))  # + x.view(-1, self.num_head * self.c_out)
        else:
            row, col, edge_weights = adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)
            x = x.view(-1, self.num_head, self.c_out)
            support = self.propagate(edge_index=edge_index,
                                     edge_weights=edge_weights,
                                     x=x,
                                     num_nodes=x.size(0)) #+ x.view(-1, self.num_head * self.c_out)
        if h0 is not None:
            support = (1 - self.alpha) * support + self.alpha * h0
        if self.cfg.MODEL.ID_MAP:
            out = beta * torch.matmul(support, self.weight) + (1 - beta) * support
        else:
            out = support
        return out

    def message(self, x_i, x_j, edge_index, edge_weights,
                num_nodes) -> torch.Tensor:
        args = [x_i, x_j, edge_index, edge_weights, num_nodes]
        kwargs = self.get_att_kwargs()
        neighbor = None
        for op, weight in self._structure['att_type'].items():
            if weight >0:
                if self.cfg.MODEL.IS_SEARCH:
                    neighbor = att_ops[op](*args, **kwargs) * weight
                else:
                    neighbor = att_ops[op](*args, **kwargs)
                break
        assert neighbor is not None, print(self._structure['att_type'])
        return neighbor

    def message_and_aggregate(self, adj_t , x, num_nodes) :
        args = [x, adj_t, num_nodes, self.aggr]
        kwargs = self.get_att_kwargs()
        neighbor = None
        for op, weight in self._structure['att_type'].items():
            if weight >0:
                if self.cfg.MODEL.IS_SEARCH:
                    neighbor = att_ops[op](*args, **kwargs) * weight
                else:
                    neighbor = att_ops[op](*args, **kwargs)
                break
        assert neighbor is not None, print(self._structure['att_type'])
        return neighbor

    def get_att_kwargs(self):
        ret = {}
        ret['out_channels'] = self.c_out
        ret['general_att_layer'] = self.general_att_layer
        ret['negative_slope'] = self.negative_slope
        ret['att'] = self.att
        ret['dropout'] = self.dropout
        ret['training'] = self.training
        ret['heads'] = self.num_head
        return ret

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        structure = self._structure
        intake1 = intake2 = inputs

        for op, weight in structure['update_type'].items():
            if weight == 0:
                continue
            else:
                if op == 'identity':
                    if self.concat:
                        output = inputs.view(-1, self.num_head * self.c_out)
                    else:
                        for size, w in self._structure['num_heads'].items():
                            if w > 0.0:
                                output = inputs.sum(dim=1).div(size)
                                break

                    output = output + self.bias
                    return output * weight
                else:
                    output = inputs
                    if self.dataset != "ppi":
                        output = self.pool_layer(inputs)
                    if self.concat:
                        output = output.view(-1, self.num_head * self.c_out)
                    else:
                        for size, w in self._structure['num_heads'].items():
                            if w > 0.0:
                                output = inputs.sum(dim=1).div(size)
                                break
                    output = output + self.bias
                    if self.dataset == 'ppi':
                        output = self.pool_layer(output)
                    return output * weight

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.c_in, self.c_out,
                                             self.num_head)

    def generate_mask(self, structure):
        hidden_sizes = self.cfg.MODEL.NAS.HIDDEN_DIM
        num_heads = self.cfg.MODEL.NAS.NUM_HEADS

        feature_mask = torch.ones(self.num_head, self.c_out).cuda()
        for size in hidden_sizes:
            if structure['hidden_dim'][size] > 0.0:
                feature_mask[:, size::] = 0
                feature_mask = feature_mask * structure['hidden_dim'][size]
                break

        head_mask = torch.ones(self.num_head, 1).cuda()
        for size in num_heads:
            if structure['num_heads'][size] > 0.0:
                head_mask[size::] = 0
                head_mask = head_mask * structure['num_heads'][size]
                break

        return feature_mask * head_mask
