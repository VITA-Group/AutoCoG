import logging
import math
import os.path as osp
import random
import time
from importlib import reload

import numpy as np
import torch
import torch.nn.functional as F
# from easydict import EasyDict as edict
# from ogb.nodeproppred import Evaluator
# from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.metrics import roc_auc_score
# from torch_geometric.utils import add_self_loops
# from torch_geometric.utils import coalesce 
# from torch_geometric.utils import negative_sampling
# from torch_geometric.utils import to_undirected
# from torch_geometric.utils import train_test_split_edges

from .dataloader import load_data
from .GCN import GCN



def seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def logger_setup(path: str, mode='INFO'):
    time_str = time.asctime(time.localtime()).split(' ')
    time_str = '-'.join(time_str)
    name = f"{time_str}-log.txt"
    mode = getattr(logging, mode)
    format = "%(levelname)s %(asctime)s - %(message)s"
    logging.shutdown()
    reload(logging)
    logging.basicConfig(level=mode,
                        format=format,
                        handlers=[
                            logging.FileHandler(osp.join(path, name)),
                            logging.StreamHandler()
                        ])
    return logging.getLogger()


def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset
    edge_idx = data.edge_index
    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_features_t
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'val': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index
    split_edge['train']['edge_neg'] = data.train_neg_edge_index
    split_edge['val']['edge'] = data.val_pos_edge_index
    split_edge['val']['edge_neg'] = data.val_neg_edge_index
    split_edge['test']['edge'] = data.test_pos_edge_index
    split_edge['test']['edge_neg'] = data.test_neg_edge_index
    data.edge_index = edge_index
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def evaluate_auc(val_pred, val_true, test_pred, test_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)

    return valid_auc, test_auc

def edge_add(nodes, edges, edge_weight=None, p=0.1, p_weight=1):
    if p == 0.0:
        return nodes, edges, edge_weight
    node_list = [i for i in range(nodes.size(0))]
    num_possible_candidates = int(p * edges.size(1))
    candidates_edges = torch.tensor([
        random.sample(node_list, 2) for _ in range(num_possible_candidates)
    ]).cuda().t()
    candidates_weights = torch.ones(candidates_edges.size(1)).cuda() * p_weight
    edges = torch.cat([edges, candidates_edges], dim=-1)
    edge_weight = torch.cat([edge_weight, candidates_weights], dim=-1)
    edges, edge_weight = coalesce(edges, edge_weight)
    return nodes, edges, edge_weight


def model_setup(model_fn, arch_fn, validate_fn, mask_fn, arch, cfg, arg, run_num=0):
    is_search = cfg.MODEL.IS_SEARCH
    loss_fn = F.nll_loss
    graph_learner = None
    data = load_data(cfg.DATA.NAME, which_run=run_num).cuda()

    data.edge_weight = torch.ones(data.edge_index.size(1)).cuda()
    # legacy shit
    # data.loss_graph = torch.tensor(0)

    settings = cfg.TRAIN
    lr = settings.LR

    num_edges = data.edge_index.shape[1]
    if not arg.disable_model:
        model = model_fn(data.x.size(1),
                         data.y.max().item() + 1, cfg, num_edges, is_search).cuda()
        architecture = arch_fn(model, data, cfg, validate_fn, loss_fn)
        if arch is not None:
            architecture.load_architecture(arch)
            if not is_search:
                architecture = None

        optimizer = torch.optim.Adam([{
            'params': model.net_reg_parameters(),
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY_1
        }, {
            'params': model.net_non_reg_parameters(),
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY_2
        }],
            lr=lr)
    else:
        model = GCN(data.x.size(1), data.y.max().item()+1, cfg).cuda()
        optimizer = model.optimizer
        architecture = None


    nodes, noisy_edges, noisy_edge_weights = edge_add(data.x, data.edge_index, data.edge_weight,
                                                      p=arg.noise)
    if arg.noise > 0.0:
        print(f"number of random edges: {noisy_edges.size() - data.edge_index.size()}")
    data.edge_index = noisy_edges
    data.edge_weight = noisy_edge_weights
    return data, model, architecture, optimizer, loss_fn
