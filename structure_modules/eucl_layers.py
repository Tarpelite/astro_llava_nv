"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, SAGEConv, GATConv





class EucLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, out_dim, dropout=0., act=nn.ReLU(),use_norm=True):
        super(EucLayer, self).__init__()
        self.linear = EucLinear(in_dim, out_dim, dropout)
        self.sph_act = EucAct(act)
        self.norm = EucNorm(out_dim)
        self.use_norm = use_norm
    def forward(self, x):
        x = self.linear(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.sph_act(x)
        return x


class EucGCLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(EucGCLayer, self).__init__()
        self.linear = EucLinear(in_dim, out_dim, manifold_in, dropout)
        self.conv1 = GCNConv(out_dim, out_dim)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = EucNorm(out_dim, manifold_in,use_norm)
        self.hyp_act = EucAct(manifold_in, manifold_out, act)

    def forward(self, input):
        x, adj = input
        # print('in:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.linear(x)
        # print('linear:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        # if self.use_norm != 'none':
        #     x = self.norm(x)
        x = self.conv1(x, adj)
        # print('agg:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        if self.use_norm != 'none':
            x = self.norm(x)
            # print('norm:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.hyp_act(x)
        # print('HypAct:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        return x


class EucSageLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(EucSageLayer, self).__init__()
        self.linear = EucLinear(in_dim, out_dim, dropout)
        self.sage = SAGEConv(out_dim, out_dim)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = EucNorm(out_dim, use_norm)
        self.hyp_act = EucAct(act)

    def forward(self, input):
        x, adj = input
        # print('in:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.linear(x)
        # print('linear:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        # if self.use_norm != 'none':
        #     x = self.norm(x)
        x = self.sage(x, adj)
        # print('agg:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        if self.use_norm != 'none':
            x = self.norm(x)
            # print('norm:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.hyp_act(x)
        # print('HypAct:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        return x

class EucATLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(EucATLayer, self).__init__()
        self.linear = EucLinear(in_dim, out_dim, dropout)
        self.gat = GATConv(out_dim, out_dim,heads=4)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = EucNorm(out_dim, use_norm)
        self.hyp_act = EucAct(act)

    def forward(self, input):
        x, adj = input
        # print('in:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.linear(x)
        # print('linear:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        # if self.use_norm != 'none':
        #     x = self.norm(x)
        x = self.gat(x, adj)
        # print('agg:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        if self.use_norm != 'none':
            x = self.norm(x)
            # print('norm:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        x = self.hyp_act(x)
        # print('HypAct:', torch.max(x.view(-1)), torch.min(x.view(-1)))
        return x



class EucLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_dim, out_dim, dropout):
        super(EucLinear, self).__init__()

        self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dp = nn.Dropout(dropout)

        self.scale = 1.
        self.reset_parameters()
    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):

        x = self.linear(x) * self.scale
        x = self.dp(x)
        return x



class EucAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, act):
        super(EucAct, self).__init__()

        self.act = act

    def forward(self, x):
        x = self.act(x)

        return x


class EucNorm(nn.Module):

    def __init__(self, in_features, method='ln'):
        super(EucNorm, self).__init__()

        if method == 'ln':
            self.norm = nn.LayerNorm(in_features)

    def forward(self, h):

        h = self.norm(h)
        return h


# def proj_tan0(u, manifold):
#     if manifold.name == 'Lorentz':
#         narrowed = u.narrow(-1, 0, 1)
#         vals = torch.zeros_like(u)
#         vals[:, 0:1] = narrowed
#         return u - vals
#     else:
#         return u
#

# def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
#     """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
#         Normalization: 'sum' or 'mean'.
#     """
#     result_shape = (num_segments, data.size(1))
#     result = data.new_full(result_shape, 0)  # Init empty result tensor.
#     segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
#     result.scatter_add_(0, segment_ids, data)
#     if aggregation_method == 'sum':
#         result = result / normalization_factor
#
#     if aggregation_method == 'mean':
#         norm = data.new_zeros(result.shape)
#         norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
#         norm[norm == 0] = 1
#         result = result / norm
#     return result
