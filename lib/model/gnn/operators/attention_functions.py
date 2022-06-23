import torch
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]



def const(*args , **kwargs):
    x_i , x_j , edge_index , edge_weight, num_nodes = args
    dropout = kwargs['dropout']
    training = kwargs['training']
    if training and dropout > 0:
        return F.dropout(x_j , p=dropout)
    if edge_weight is not None:
        return edge_weight.view(-1 , 1 , 1) * x_j
    return x_j


def att_wrapper(func):
    def wrapper(*args , **kwargs):
        x_i , x_j , edge_index , edge_weight , num_nodes = args
        training = kwargs['training']
        dropout = kwargs['dropout']
        heads = kwargs['heads']
        alpha = func(*args , **kwargs)
        alpha = softmax(alpha , edge_index[0] , num_nodes=num_nodes)

         # edge_weight = norm(edge_index, num_nodes=num_nodes, edge_weight=edge_weight)
        # edge_weight = normalized_cut(edge_index, edge_weight, num_nodes=num_nodes)
        if training and dropout > 0:
            alpha = F.dropout(alpha , p=dropout , training=True)

        return x_j * alpha.view(-1 , heads , 1)

    return wrapper


def gcn(*args , **kwargs):
    if len(args) == 5:
        x_i , x_j , edge_index , edge_weight, num_nodes = args
        _ , normalized = gcn_norm(edge_index , num_nodes=num_nodes , edge_weight=edge_weight)
        gcn_weight = normalized
        alpha = gcn_weight
        return alpha.view(-1 , 1 , 1) * x_j
    else:
        x, edge_index, num_nodes, aggr = args
        norm = gcn_norm(edge_index, num_nodes=num_nodes)
        return matmul(norm, x, reduce=aggr)

@att_wrapper
def gat(*args , **kwargs):
    x_i , x_j , edge_index , edge_weight, num_nodes = args
    att , negative_slope = kwargs['att']['gat'] , kwargs['negative_slope']
    alpha = (torch.cat([x_i , x_j] , dim=-1) * att).sum(dim=-1)
    alpha = F.leaky_relu(alpha , negative_slope)
    if edge_weight is not None:
        alpha = alpha.squeeze() * edge_weight
        alpha = alpha.unsqueeze(-1)
    return alpha
    # x = x_i + x_j
    # x = F.leaky_relu(x, negative_slope)
    # att = att.view(1, -1, 2, kwargs['out_channels'])
    # att = att.mean(2)
    # alpha = (x * att).sum(dim=-1)
    # alpha = F.softmax(alpha, -1)# index, ptr, size_i)
    # alpha = F.dropout(alpha, p=kwargs['dropout'], training=kwargs['training'])
    # return alpha.unsqueeze(-1)

@att_wrapper
def gat_sym(*args , **kwargs):
    x_i , x_j , edge_index , edge_weight, num_nodes = args
    out_channels , negative_slope , att = kwargs['out_channels'] , kwargs['negative_slope'] , kwargs['att']['gat_sym']
    wl = att[: , : , :out_channels]  # weight left
    wr = att[: , : , out_channels:]  # weight right
    alpha = (x_i * wl).sum(dim=-1) + (x_j * wr).sum(dim=-1)
    alpha_2 = (x_j * wl).sum(dim=-1) + (x_i * wr).sum(dim=-1)
    alpha = F.leaky_relu(alpha , negative_slope) + F.leaky_relu(alpha_2 , negative_slope)
    if edge_weight is not None:
        alpha = alpha.squeeze() * edge_weight
        alpha = alpha.unsqueeze(-1)
    return alpha


@att_wrapper
def linear(*args , **kwargs):
    x_i , x_j , edge_index , edge_weight, num_nodes = args
    out_channels , negative_slope , att = kwargs['out_channels'] , kwargs['negative_slope'] , kwargs['att']['linear']
    wl = att[: , : , :out_channels]  # weight left
    wr = att[: , : , out_channels:]  # weight right
    al = x_j * wl
    ar = x_j * wr
    alpha = al.sum(dim=-1) + ar.sum(dim=-1)
    alpha = torch.tanh(alpha)
    if edge_weight is not None:
        alpha = alpha.squeeze() * edge_weight
        alpha = alpha.unsqueeze(-1)
    return alpha


@att_wrapper
def cos(*args , **kwargs):
    x_i , x_j , edge_index , edge_weight, num_nodes = args
    out_channels , att = kwargs['out_channels'] , kwargs['att']['cos']
    wl = att[: , : , :out_channels]  # weight left
    wr = att[: , : , out_channels:]  # weight right
    alpha = x_i * wl * x_j * wr
    alpha = alpha.sum(dim=-1)

    if edge_weight is not None:
        alpha = alpha.squeeze() * edge_weight
        alpha = alpha.unsqueeze(-1)
    return alpha


@att_wrapper
def generalized_linear(*args , **kwargs):
    x_i , x_j , edge_index , edge_weight, num_nodes = args
    out_channels , general_att_layer , att = kwargs['out_channels'] , kwargs['general_att_layer'] , kwargs['att']['generalized_linear']
    wl = att[: , : , :out_channels]  # weight left
    wr = att[: , : , out_channels:]  # weight right
    al = x_i * wl
    ar = x_j * wr
    alpha = al + ar
    alpha = torch.tanh(alpha)
    alpha = general_att_layer(alpha)
    if edge_weight is not None:
        alpha = alpha.squeeze() * edge_weight
        alpha = alpha.unsqueeze(-1)
    return alpha


OPS = {'const': const , 'gcn': gcn , 'gat': gat , 'gat_sym': gat_sym , 'linear': linear ,
       'cos': cos , 'generalized_linear': generalized_linear}
