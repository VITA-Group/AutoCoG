import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_scatter import scatter_add

from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.utils import add_remaining_self_loops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops
from yacs.config import CfgNode as CN
from easydict import EasyDict as edict
from typing import Union
from .operators import att_ops, gdas_decode
import logging
from torch_scatter import scatter
from torch_geometric.utils import add_remaining_self_loops
from typing import Dict

class GeoLayer(MessagePassing):
    def __init__(self,
                 agg_type,
                 in_channels,
                 out_channels,
                 heads=1,
                 pooling_dim=128,
                 concat=True,
                 cfg=None
                 ):
        self.cfg = cfg

        if agg_type in ["add", "mean", "max"]:
            super(GeoLayer, self).__init__(aggr=agg_type, node_dim=0)
        else:
            raise Exception("Wrong attention type:", self.agg_type)

        self.in_channels = in_channels
        self.c_out = self.out_channels = out_channels
        self.heads = self.num_head=  heads
        self.concat = concat
        self.negative_slope = cfg.MODEL.NEGATIVE_SLOPE
        self.dropout = cfg.MODEL.DROPOUT
        self.gcn_weight = None   # GCN weight
        self.dataset = cfg.DATA.NAME
        bias= cfg.MODEL.BIAS

        self.weight = Parameter(
            torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        self.general_att_layer = torch.nn.Linear(out_channels, 1, bias=False)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        pool_dim = 128
        self.pool_dim = pool_dim
        if self.dataset == 'ppi':
            self.pool_layer = [] #torch.nn.ModuleList()
            if self.concat:
                self.pool_layer.append(torch.nn.Linear(self.out_channels*self.heads, self.pool_dim))
                self.pool_layer.append(torch.nn.ReLU())
                self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels*self.heads))
            else:
                self.pool_layer.append(torch.nn.Linear(self.out_channels, self.pool_dim))
                self.pool_layer.append(torch.nn.ReLU())
                self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels))
        else:
            self.pool_layer = []#torch.nn.ModuleList()
            self.pool_layer.append(torch.nn.Linear(self.out_channels, self.pool_dim))
            self.pool_layer.append(torch.nn.ReLU())
            self.pool_layer.append(torch.nn.Linear(self.pool_dim, self.out_channels))
        self.pool_layer= nn.Sequential(*self.pool_layer)
    @staticmethod
    def norm(edge_index , num_nodes , edge_weight , improved=False , dtype=None , add_self_loops=False):
        num_nodes = maybe_num_nodes(edge_index , num_nodes)
        fill_value = 2. if improved else 1.

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1) ,) , dtype=dtype ,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index , tmp_edge_weight = add_remaining_self_loops(
                edge_index , edge_weight , fill_value , num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row , col = edge_index[0] , edge_index[1]
        deg = scatter_add(edge_weight , col , dim=0 , dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf') , 0)
        return edge_index , deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, structure):
        """"""
        self.structure = structure
        edge_index , tmp_edge_weight = add_remaining_self_loops(
            edge_index , None , 1.0 , x.size(0))
        # prepare
        x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)
        if self.dataset == 'ppi':
            x = F.normalize(x, p=2, dim=-1)
        else:
            pass
        return self.propagate(edge_index, x=x, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes) -> torch.Tensor:
        neighbor = 0
        for op, weight in self.structure['att_type'].items():
            neighbor = neighbor + self.get_neighbor(edge_index, num_nodes, x_i, x_j, op) * weight
        return neighbor

    def get_neighbor(self, edge_index, num_nodes, x_i, x_j, att_type):
        if att_type == "const":
            if self.training and self.dropout > 0:
                x_j = F.dropout(x_j, p=self.dropout, training=True)
            neighbor = x_j
        else:
            alpha = self.apply_attention(edge_index, num_nodes,x_i, x_j, att_type)
            if att_type == 'gcn':
                neighbor = alpha.view(-1, 1, 1) * x_j
            else:
                # Compute attention coefficients.
                alpha = softmax(alpha, edge_index[0], num_nodes=num_nodes)
                # Sample attention coefficients stochastically.
                if self.training and self.dropout > 0:
                    alpha = F.dropout(alpha, p=self.dropout, training=True)
                neighbor = x_j * alpha.view(-1, self.heads, 1)

        return neighbor

    def apply_attention(self, edge_index, num_nodes, x_i, x_j, att_type):
        if att_type == 'gcn':
            _, norm = self.norm(edge_index, num_nodes, None)
            if self.training and self.dropout > 0 :
                alpha = F.dropout(norm, p=self.dropout, training=True)
            else:
                alpha = norm
        elif att_type == "gat":
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)

        elif att_type == "gat_sym":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
            alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope) + F.leaky_relu(alpha_2, self.negative_slope)

        elif att_type == "linear":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            al = x_j * wl
            ar = x_j * wr
            alpha = al.sum(dim=-1) + ar.sum(dim=-1)
            alpha = torch.tanh(alpha)
        elif att_type == "cos":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            alpha = x_i * wl * x_j * wr
            alpha = alpha.sum(dim=-1)

        elif att_type == "generalized_linear":
            wl = self.att[:, :, :self.out_channels]  # weight left
            wr = self.att[:, :, self.out_channels:]  # weight right
            al = x_i * wl
            ar = x_j * wr
            alpha = al + ar
            alpha = torch.tanh(alpha)
            alpha = self.general_att_layer(alpha)
        else:
            raise Exception("Wrong attention type:" , att_type)
        return alpha

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        structure = self.structure
        intake1 = intake2 =  inputs
        if self.dataset != 'ppi':
            intake1 = self.pool_layer(intake1)

        if self.concat:
            intake1 = intake1.view(-1, self.num_head * self.c_out)
            intake2 = intake2.view(-1, self.num_head * self.c_out)
        else:
            intake1 = intake1.mean(dim=1)
            intake2 = intake2.mean(dim=1)

        if self.bias is not None:
            intake1 = intake1 + self.bias
            intake2 = intake2 + self.bias

        if self.dataset == 'ppi':
            intake1 = self.pool_layer(intake1)

        return intake1 * structure['update_type']['mlp'] + intake2 * structure['update_type']['identity']


