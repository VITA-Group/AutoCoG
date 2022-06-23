import logging

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

from .gnn.operators import data_ops, act_ops, gdas_decode
from typing import Union
from yacs.config import CfgNode as CN
from easydict import EasyDict as edict
from .gnn.message_passing import GDASGnnConv
from typing import Callable
from torch_geometric.datasets import Planetoid
import os
import logging

class ArchGdas:
    def __init__(self , model:nn.Module , data:Planetoid ,
                 cfg: Union[CN, edict] , search_fn:Callable ,
                 loss_fn: Callable):
        self.cfg = cfg
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=cfg.SEARCH.LR_ARCH)
        self.search_fn = search_fn
        self.loss_fn = loss_fn
        self.data = data
        self.logger = logging.getLogger()
        self.acc_valid = self.loss_valid = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        self.acc_valid, self.loss_valid = self.search_fn(self.model, self.data, self.loss_fn)
        self.optimizer.zero_grad()
        self.loss_valid.backward()
        self.optimizer.step()

    def decay(self):
        self.model.decay()

    def status(self):
        ret = {}
        ret['tau'] = self.model.tau
        ret['acc_valid'] = self.acc_valid
        ret['loss_valid'] = self.loss_valid
        ret.update(self.get_entropy())
        return  ret

    def evaluate(self, output , labels , mask):
        _ , indices = torch.max(output , dim=1)
        correct = torch.sum(indices[mask] == labels[mask])
        return correct.item() * 1.0 / mask.sum().item()

    def get_entropy(self):
        arch_state_dict = self.model.arch_state_dict()
        entropies = {}
        for k , v in arch_state_dict.items():
            prob = F.softmax(v , dim=-1)
            log_prob = F.log_softmax(v , dim=-1)
            entropies[k] = -(log_prob * prob).sum(-1 ,
                                                  keepdim=False).mean().item()
        return entropies

    def load_architecture(self, path:str ):
        assert os.path.isfile(path)
        w = torch.load(path)
        self.model.load_state_dict(w, strict=False)
        self.summary()

    def save_architecture(self, path:str):
        state_dict = self.model.arch_state_dict()
        torch.save(state_dict, path)

    def summary(self):
        architecture = self.model.arch_state_dict()
        data_struct = gdas_decode(architecture['arch_beta'] , self.cfg.DATA)
        model_struct = []
        for i in range(architecture['arch_alpha'].size(0)):
            model_struct.append(gdas_decode(architecture['arch_alpha'][i] , self.cfg.MODEL))
        msg = f"Data sampling policy:\n"
        for k, v in data_struct.items():
            if k == 'arch': continue
            msg = msg + f"{k}: {v}\n"
        msg = msg + f"arch settings:\n"
        for i, layer in enumerate(model_struct):
            msg = msg + f"\t layer {i} settings:\n"
            for k, v in layer.items():
                if k == 'arch': continue
                msg = msg + f"\t\t{k}:{v}\n"
        self.logger.info(msg)

class GdasGNN(nn.Module):
    def __init__(self, c_in:int, c_out:int, cfg: Union[CN, edict], is_search:bool):
        super().__init__()
        self.logger = logging.getLogger()
        self.is_search = is_search
        self.cfg = cfg
        self.c_in = c_in
        self.c_out = c_out
        self.data_name = cfg.DATA.NAME.lower()
        settings = cfg.MODEL
        # non-nas settings
        self.bn = settings.BATCHNORM # bool
        self.residual = settings.RESIDUAL # bool
        self.n_layers = settings.NUM_LAYERS # int
        self.dropout = settings.DROPOUT # float
        self.n_heads = settings.N_HEADS # int
        self.hidden_dim = settings.HIDDEN_DIM
        # nas settings
        self.n_states = self.total_states(settings.NAS)
        self.tau_max, self.tau_min = self.cfg.SEARCH.TAU
        decay_factor = self.cfg.SEARCH.TAU_DECAY_FACTOR
        self.tau = self.tau_max
        self.tau_decay = (self.tau_max- self.tau_min) / ((cfg.SEARCH.END_EPOCH - cfg.SEARCH.BEGIN_EPOCH) * decay_factor)
        #
        self.construct_model()
        self.init()

    def init(self)->None:
        for name, m in self.named_parameters():
            self.logger.debug(f"init {name} {m.shape}")
            if 'arch' in name:
                torch.nn.init.constant_(m, 1e-3)
            elif 'bias' in name:
                nn.init.zeros_(m)
            else:
                nn.init.xavier_normal_(m)

    def construct_model(self) -> None:
        if not self.cfg.MODEL.SIMPLIFIED:
            self.arch_alpha = nn.Parameter(torch.empty(self.n_layers , self.n_states))
            self._arch_alpha = None
        else:
            self.arch_alpha = nn.Parameter(torch.empty(1, self.n_states))
            self._arch_alpha = None

        #
        if self.cfg.DATA.MODE == 'policy':
            self.arch_beta = nn.Parameter(torch.empty(len(self.cfg.DATA.NAS.POLICY)))
            self._arch_beta = None
        else:
            raise NotImplementedError

        self.layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        c_in = self.c_in
        c_out = self.hidden_dim
        concat = True
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                c_out = self.c_out
                concat = False
                self.skip_layers.append(nn.Linear(c_in, c_out))
            else:
                self.skip_layers.append(nn.Linear(c_in, c_out*self.n_heads))
            #l = nn.ModuleDict()
            #for aggr_type in self.aggr_type:
            l = nn.ModuleDict()
            for aggr in self.aggr_type:
                l[aggr] = GDASGnnConv(aggr, c_in, c_out, self.n_heads, 128, concat, self.cfg )
            self.layers.append(l)
            if self.bn:
                self.bn_layers.append(nn.BatchNorm1d(c_in))
            c_in = c_out * self.n_heads

    def forward(self, x, edge, val=False):
        if self.is_search and self.training:
            self.gumbel_softmax(True)
        else:
            self.softmax(True)
            self._arch_alpha = self._arch_alpha.detach().clone()
            self._arch_beta = self._arch_beta.detach().clone()
        return self._forward(x, edge)

    def _forward(self, x:torch.Tensor, edge_index_all: torch.Tensor):
        prev =  x
        output = x
        sampled_edges = edge_index_all
        for i, layer in enumerate(self.layers):
            alpha_idx = i if not self.cfg.MODEL.SIMPLIFIED else 0
            structure = gdas_decode(self._arch_alpha[alpha_idx] , self.cfg.MODEL)
            self.logger.debug(msg=f"arch structure {structure}")
            if self.training:
                output, sampled_edges = data_ops[structure['data_policy']](output, edge_index_all)
            if self.bn:
                output = self.bn_layers[i](output)
            if self.data_name == 'ppi':
                raise NotImplementedError
            else:
                output = F.dropout(output, p=0.6, training=self.training)
                output = layer[structure['aggr_type']](output, sampled_edges, self._arch_alpha[alpha_idx]) * structure['arch']['weight']
                self.logger.debug(f"layer {i}: output {output.shape}")
            if structure['residual']:
                output = output + self.skip_layers[i](prev)
            output = act_ops[structure['act_type']](output) * structure['arch']['weight']
            prev = output

        return output

    def gumbel_softmax(self, onehot=True) -> None :
        self._arch_alpha = self._gumbel_softmax(onehot, self.arch_alpha)
        self._arch_beta = self._gumbel_softmax(onehot, self.arch_beta)

    def _gumbel_softmax(self, onehot, arch):
        return F.gumbel_softmax(F.log_softmax(arch , dim=-1) , tau=self.tau , hard=onehot , dim=-1)

    def softmax(self, onehot=True):
        self._arch_alpha = self._softmax(onehot, self.arch_alpha)
        self._arch_beta = self._softmax(onehot, self.arch_beta)

    def _softmax(self, onehot, arch):
        if onehot:
            max_idx = torch.argmax(arch , -1 , keepdim=True)
            one_hot = torch.zeros_like(arch)
            one_hot.scatter_(-1 , max_idx ,
                             1)  # onehot but detach from graph
            return one_hot - arch.detach() + arch  # attach to gradient
        else:
            return F.softmax(self.arch_alpha, dim=-1)

    def decay(self):
        self.tau = max(self.tau_min, self.tau - self.tau_decay)

    def total_states(self, config: Union[CN, edict]) -> int:
        config = dict(config)
        ret = 1
        for name, states in config.items():
            ret *= len(states)
            setattr(self, name.lower(), states)
        return ret

    def arch_state_dict(self):
        return {"arch_alpha": self.arch_alpha, "arch_beta": self.arch_beta}

    def arch_parameters(self):
        return [self.arch_alpha, self.arch_beta]

    def net_parameters(self):
        for name, m in self.named_parameters():
            if 'arch' in name: continue
            yield m



