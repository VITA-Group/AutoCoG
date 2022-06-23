import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv


class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class GCN(nn.Module):
    def __init__(self, num_feats, num_layers, num_classes, dim_hidden=128, dropout=0.6, cached=True,type_norm=None):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.num_feats = num_feats
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dim_hidden = dim_hidden
        self.cached = cached
        self.type_norm = type_norm
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])

        # self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        cur_dim = self.num_feats
        for _ in range(self.num_layers - 1):
            self.layers_GCN.append(
                GCNConv(cur_dim, self.dim_hidden, cached=self.cached))
            cur_dim = self.dim_hidden

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())
        self.layers_GCN.append(GCNConv(cur_dim, self.num_classes, cached=self.cached))

        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())
        # self.optimizer = torch.optim.Adam(self.parameters(),
        #                                   lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, edge_index, edge_weight):

        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py

        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index, edge_weight)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.layers_GCN[-1](x, edge_index)
        return x
