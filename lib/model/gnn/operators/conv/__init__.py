from .agnn import AGNNConv
from .gat import GATv2Conv
from .gcn import GCNConv
from .linear import Linear



def construct_attention(name, c_in, c_out, num_head, concat, cfg):

    if name == 'agnn':
        return AGNNConv()
    elif name == 'gat':
        return GATv2Conv(in_channels=c_in, out_channels=c_out, heads=num_head,
                         concat=concat, negative_slope=cfg.MODEL.NEGATIVE_SLOPE,
                         dropout=cfg.MODEL.DROPOUT)
    elif name == 'gcn':
        return GCNConv(in_channels=c_in, out_channels=c_out*num_head,)
    elif name == 'linear':
        return Linear(in_channels=c_in, out_channels=c_out*num_head)
    else:
        raise NotImplementedError
