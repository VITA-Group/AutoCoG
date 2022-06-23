import logging
import os
import os.path as osp
import random
from collections import defaultdict
from typing import *
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as U
import torch_scatter
import tqdm
from torch import nn
from torch import Tensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter


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
    idx = torch.arange(0, nnz).cuda().sub_(
        mask.logical_not_().cumsum(dim=0))

    if isinstance(edge_attr, Tensor):
        edge_attr = scatter(edge_attr, idx, 0, None, dim_size, reduce)
    else:
        edge_attr = [
            scatter(e, idx, 0, None, dim_size, reduce) for e in edge_attr
        ]

    return edge_index, edge_attr


def node_drop(nodes: torch.Tensor,
              edge_index: torch.Tensor,
              edge_weight=None,
              p=0.2,
              p_weight=1):
    num_nodes = maybe_num_nodes(edge_index, nodes.size(0))
    _nodes = torch.arange(num_nodes, dtype=torch.int64)
    mask = torch.full_like(_nodes, 1 - p, dtype=torch.float32)
    mask = torch.bernoulli(mask).to(torch.bool)
    subnodes = _nodes[mask]
    edges, attr = subgraph(subnodes,
                           edge_index,
                           edge_attr=edge_weight,
                           num_nodes=num_nodes)
    attr = attr * p_weight
    return nodes, edges.to(nodes.device), attr


def edge_perturbation(nodes, edges, edge_weight=None, p=0.2, p_weight=1):
    edges, attr = U.dropout_adj(edges, edge_attr=edge_weight, p=p)
    attr = attr * p_weight
    return nodes, edges, attr


def edge_add(nodes, edges, edge_weight=None, p=0.1, p_weight=1):
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


def attr_masking(nodes, edges, edge_weight=None, p=0.5, p_weight=1):
    nodes = torch.nn.functional.dropout(nodes, p=p)
    return nodes, edges, edge_weight


def none(node, edges, edge_weight=None):
    return node, edges, edge_weight


class EdgeGenerator(nn.Module):
    def __init__(self, cfg, num_nodes, num_edges, edges, nodes):
        super().__init__()

        self.cfg = cfg
        self.allow_add = cfg.MODEL.GENERATE_EDGES.ALLOW_ADD
        self.allow_minus = cfg.MODEL.GENERATE_EDGES.ALLOW_MINUS
        self.num_edges = num_edges
        self.num_nodes = num_nodes

        if self.allow_minus:
            self.arch_edges_alpha = nn.Parameter(torch.empty(num_edges, 2))
        if self.allow_add:
            self.arch_edges_beta = nn.Parameter(
                torch.empty(num_nodes, num_nodes, 2))
        self._arch_edges_alpha = self._arch_edges_beta = None

        self.sampling_policy = cfg.MODEL.GENERATE_EDGES.SAMPLING_METHOD  # 'random'
        self.sample_range = cfg.MODEL.GENERATE_EDGES.SAMPLE_RANGE
        self.derive_range = cfg.MODEL.GENERATE_EDGES.DERIVE_RANGE
        self.flag = False

        edges = edges.t()
        self.gt_edges_hash = defaultdict(set)
        for i in range(edges.shape[0]):
            self.gt_edges_hash[edges[i][0].item()].add(edges[i][-1].item())

        if self.sampling_policy == 'distance':
            distance_mode = cfg.MODEL.GENERATE_EDGES.D_M
            tar = f"_cached_{cfg.DATA.NAME}_distance.pth.tar"
            if osp.isfile(tar):
                nodes_distance = torch.load(tar)
            else:
                nodes_distance = list()
                pdist = nn.PairwiseDistance(p=2)
                for i in tqdm.tqdm(range(nodes.shape[0])):
                    emb_i = nodes[i, :].unsqueeze(0)
                    for j in range(i + 1, nodes.shape[0]):
                        if i == j: continue
                        emb_j = nodes[j, :].unsqueeze(0)
                        dist = pdist(emb_i, emb_j).squeeze().item()
                        nodes_distance.append([i, j, dist])
                nodes_distance = torch.tensor(nodes_distance)
                torch.save(nodes_distance, tar)

            if distance_mode == 'proximity':
                nodes_distance = nodes_distance[nodes_distance[:, -1].le(
                    nodes_distance[:, -1].mean())]
            elif distance_mode == 'centroid':
                mean = nodes_distance[:, -1].mean()
                stdev = nodes_distance[:, -1].std()
                mask = nodes_distance[:, -1].ge(
                    mean - stdev) * nodes_distance[:, -1].le(mean + stdev)
                nodes_distance = nodes_distance[mask.bool()]
            self.nodes_distance = defaultdict(list)
            for i, j, dist in nodes_distance:
                self.nodes_distance[i.int().item()].append(
                    [j.int().item(), dist.item()])

            for k, v in self.nodes_distance.items():
                sorted_v = sorted(v, key=lambda x: x[-1])
                v = []
                for v_j, v_d in sorted_v:
                    if v_j in self.gt_edges_hash[k]:
                        continue
                    else:
                        v.append(v_j)
                self.nodes_distance[k] = v
            # if self.allow_add:
            #     self.arch_edges_beta = nn.Parameter(torch.empty(num_nodes, self.sample_range, 2))

    def forward(self, nodes, edge_index_all, is_searching=True):
        if is_searching:
            return self.search_forward(nodes, edge_index_all)
        else:
            return self.inference(nodes, edge_index_all)

    def inference(self, nodes, edge_index_all):
        if not self.flag:
            edge_index_all = edge_index_all.t()
            if self.allow_minus:
                edge_index_all = edge_index_all[self.arch_edges_alpha[:, 0].ge(
                    self.arch_edges_alpha[:, 1])]
                # import pdb; pdb.set_trace()

            # edge_weight =  torch.ones(edge_index_all.shape[1])#._arch_edges_alpha[self._arch_edges_alpha[:,0].gt(0)][:, 0]
            if self.allow_add:
                edge_index_all = self.add_permanent_edges(
                    nodes, edge_index_all)
            self.flag = True
            self._edges = edge_index_all.t()
        # print(self._edges)
        return self._edges, None
        # drop edge

    def search_forward(self, nodes, edge_index_all):
        self.flag = False
        edge_index_all = edge_index_all.t()
        edge_weight = None
        if self.allow_minus:
            edge_index_all = edge_index_all[self._arch_edges_alpha[:, 0].gt(
                0).t()]
            edge_weight = self._arch_edges_alpha[
                self._arch_edges_alpha[:, 0].gt(0)][:, 0]
            # edge_weight = self._arch_edges_alpha[:, 0]
        else:
            edge_weight = torch.ones(edge_index_all.shape[0]).cuda()
        # add edges
        if self.allow_add:
            edge_index_all, edge_weight = self.add_edges(
                nodes, edge_index_all, edge_weight)

        # print(edge_weight.shape, edge_index_all.shape)
        # print(edge_index_all)
        return edge_index_all.t(), edge_weight

    def add_permanent_edges(self, nodes, edge_index_all):
        pair = self._get_new_pair(self.arch_edges_beta)
        pair = torch.tensor(pair).cuda()
        edge_index_all = torch.cat([edge_index_all, pair], dim=0).long()
        return edge_index_all

    def _get_new_pair(self, beta):
        hash_table = defaultdict(set)
        pair = []
        # if self.sampling_policy == 'random':
        for i in range(beta.shape[0]):
            w = nn.functional.softmax(beta[i, :, 0], dim=-1)
            values, indicies = w.topk(self.derive_range)
            if values[0] == 1 / w.shape[0]: continue
            indicies = indicies.tolist()
            for index in indicies:
                if i == index:
                    continue
                elif index in hash_table[i]:
                    continue
                elif i in hash_table[index]:
                    continue
                elif index in self.gt_edges_hash[i]:
                    continue
                elif i in self.gt_edges_hash[index]:
                    continue
                else:
                    pair.append([i, index])
                    hash_table[i].add(index)
                    hash_table[index].add(i)
        # elif self.sampling_policy == 'distance':

        return pair

    def add_edges(self, nodes, edge_index_all, edge_weight):
        fro, to = 0, 0
        if self.sampling_policy == 'random':
            fro = random.randint(0, self.num_nodes - self.sample_range - 1)
            to = random.randint(0, self.num_nodes - self.sample_range - 1)
            index_fro = [i for i in range(fro, fro + self.sample_range)]
            index_to = [i for i in range(to, to + self.sample_range)]
            pair = []
            weight = []
            hash_table = defaultdict(set)
            for i in index_fro:
                for j in index_to:
                    if i == j: continue
                    if i in hash_table[j]: continue
                    if j in hash_table[i]: continue
                    w = self._arch_edges_beta[i][j][0]
                    if j not in self.gt_edges_hash[i] and w > 0:
                        pair.append([i, j])
                        weight.append(w)
                        hash_table[i].add(j)
                        hash_table[j].add(i)
        elif self.sampling_policy == 'distance':
            pair = []
            weight = []
            hash_table = defaultdict(set)
            for i, neighbors in self.nodes_distance.items():
                if len(neighbors) > self.sample_range:
                    neighbors = random.sample(neighbors, self.sample_range)

                w = self._arch_edges_beta[i, neighbors, 0]
                # pair.append(*[[i, j] for j in neighbors])
                # weight.append(w)
                for w_i, j in enumerate(neighbors):
                    if i == j: continue
                    if i in hash_table[j]: continue
                    if j in hash_table[i]: continue
                    # w = self._arch_edges_beta[i][j][0]
                    if j not in self.gt_edges_hash[i] and w[w_i] > 0:
                        pair.append([i, j])
                        weight.append(w[w_i])
                    hash_table[i].add(j)
                    hash_table[j].add(i)

        else:
            raise NotImplementedError
        pair = torch.tensor(pair).cuda()
        weight = torch.stack(weight, dim=0)  #.cuda()
        # pair = pair[weight.gt(0)]
        # weight = weight[weight.gt(0)]
        # import pdb; pdb.set_trace()

        edge_index_all = torch.cat([edge_index_all, pair], dim=0)
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, weight], dim=-1)
        else:
            edge_weight = weight
        return edge_index_all, edge_weight

        # grid = self._arch_edges_beta[fro:fro+self.range, to:to+self.range, :]


OPS = {
    'node_drop': node_drop,
    'edge_perturbation': edge_perturbation,
    'attr_masking': attr_masking,
    'none': none,
    'edge_add': edge_add
}
