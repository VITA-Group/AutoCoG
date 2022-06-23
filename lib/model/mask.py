import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .GCN import GCN
from torch_geometric.nn import VGAE, GCNConv


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index, weight):
        x = self.conv1(x, edge_index, weight).relu()
        return self.conv_mu(x, edge_index, weight), self.conv_logstd(x, edge_index, weight)

class Mask(nn.Module):
    def __init__(self, cfg, num_features, **kwargs):
        super(Mask, self).__init__()
        self.cfg = cfg
        self.hidden = 128 #cfg.MODEL.HIDDEN_DIM
        self.num_feats = num_features
        self.lin1 = nn.Linear(num_features, self.hidden)
        self.encoder = VGAE(encoder=Encoder(self.hidden, self.hidden))
        self.lin = nn.Linear(self.hidden*2, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.TRAIN.LR_MASK, weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    def forward(self, edge_index, x, weights=None):
        x = self.lin1(F.dropout(x, p=0.4, training=self.training))
        self.__pos_edge__ = edge_index
        self.__num_nodes__ = x.size(0)
        self.__z__ = embeddings = self.encoder.encode(x, edge_index, weights)
        emb_edges =  torch.cat([embeddings[edge_index[0]], embeddings[edge_index[1]]], dim=1)
        output = self.lin(F.dropout(emb_edges, p=0.6, training=self.training))
        output = torch.sigmoid(output).squeeze() # generate edge weights
        return output

    def binarize(self, output, percentile=5):
        mask = output.detach().clone().cpu().numpy()
        threshold = np.percentile(mask, percentile)
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        mask = torch.tensor(mask).cuda()
        return mask

    def random_drop(self, edge_weights):
        mask = torch.bernoulli(edge_weights)
        edge_weights[mask==0] = 0
        return edge_weights

    def get_kl_loss(self):
        return self.encoder.kl_loss()

    def get_recon_loss(self):
        return self.encoder.recon_loss(self.__z__, self.__pos_edge__)

    def get_loss(self):
        self.__loss__ =  1/(self.__num_nodes__)*self.get_kl_loss() + self.get_recon_loss()
        return self.__loss__


