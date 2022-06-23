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
    def __init__(self, num_feats, num_classes, cfg):
        super(GCN, self).__init__()
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.dim_hidden = cfg.MODEL.HIDDEN_DIM
        self.lr = cfg.TRAIN.LR
        self.weight_decay = cfg.TRAIN.WEIGHT_DECAY
        self.cached = False
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.type_norm = None
        self.num_layers = cfg.MODEL.NUM_LAYERS
        self.dropout = cfg.MODEL.DROPOUT


        # self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())
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
        self.prompt = None
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=self.lr, weight_decay=self.weight_decay)
        self.tau = 0
        self._is_search = False

    def forward(self, x, edge_index, edge_weight, val=False):

        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py

        for i in range(self.num_layers - 1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index, edge_weight)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)
        self.final_embeddings = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)
        return x


    def decay(self):
        pass

    def increase(self):
        pass

    def get_current_arch(self):
        ret = {}
        return ret

    def arch_state_dict(self):
        ret = {}
        return ret  # {"arch_alpha": self.arch_alpha, "arch_beta": self.arch_beta}

    def arch_parameters(self):
        return []
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
