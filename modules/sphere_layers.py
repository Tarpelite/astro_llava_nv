"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from geoopt import Sphere
# from manifolds import Sphere
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, SAGEConv, GATConv



def get_dim_act_curv(config, num_layers, enc=True):
    """
    Helper function to get dimension and activation at every layer.
    :param args:
    :return:
    """
    model_config = config.model
    act = getattr(nn, model_config.act)
    if isinstance(act(),nn.LeakyReLU):
        acts = [act(0.5)] * (num_layers)
    else:
        acts = [act()] * (num_layers)  # len=args.num_layers
    if enc:
        dims = [model_config.hidden_dim] * (num_layers+1)  # len=args.num_layers+1
    else:
        dims = [model_config.dim]+[model_config.hidden_dim] * (num_layers)  # len=args.num_layers+1

    manifold_class = {'PoincareBall': PoincareBall, 'Lorentz': Lorentz}

    if enc:
        manifolds = [manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c)
                     for _ in range(num_layers)]+[manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c)]
    else:
        manifolds = [manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c)]+\
                    [manifold_class[model_config.manifold](model_config.c, learnable=model_config.learnable_c) for _ in
                    range(num_layers)]

    return dims, acts, manifolds


class SphNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(),use_norm=True):
        super(SphNNLayer, self).__init__()
        self.linear = SphLinear(in_dim, out_dim, manifold_in, manifold_out, dropout)
        self.sph_act = SphAct(manifold_out, manifold_out, act)
        self.norm = SphNorm(out_dim, manifold_out)
        self.use_norm = use_norm
    def forward(self, x):
        x = self.linear(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.sph_act(x)
        return x


class SphGCLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(SphGCLayer, self).__init__()
        self.linear = SphNNLayer(in_dim, out_dim, manifold_in, manifold_out, dropout)
        self.conv1 = GCNConv(out_dim, out_dim)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = SphNorm(out_dim, manifold_out,use_norm)
        self.hyp_act = SphAct(manifold_out, manifold_out, act)

    def forward(self, x,adj):

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


class SphSageLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(SphSageLayer, self).__init__()
        self.linear = SphNNLayer(in_dim, out_dim, manifold_in,manifold_out, dropout)
        self.sage = SAGEConv(out_dim, out_dim)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = SphNorm(out_dim, manifold_out,use_norm)
        self.hyp_act = SphAct(manifold_out, manifold_out, act)

    def forward(self, x, adj):

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

class SphATLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(), edge_dim=1, normalization_factor=1,
                 aggregation_method='sum', msg_transform=True, sum_transform=True,use_norm='ln'):
        super(SphATLayer, self).__init__()
        self.linear = SphNNLayer(in_dim, out_dim, manifold_in, manifold_out,dropout)
        self.gat = GATConv(out_dim, out_dim,heads=4)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = SphNorm(out_dim, manifold_out,use_norm)
        self.hyp_act = SphAct(manifold_out, manifold_out, act)

    def forward(self, x,adj):
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



class SphLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_dim, out_dim, manifold_in,manifold_out, dropout):
        super(SphLinear, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.bias = nn.Parameter(torch.Tensor(1, out_dim))
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.dp = nn.Dropout(dropout)

        self.scale = 1.
        self.reset_parameters()

    def reset_parameters(self):
        # init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        x = self.manifold_in.logmap0(x)
        x = self.linear(x) * self.scale
        x = self.dp(x)
        x = self.manifold_out.proju0(x,)
        x = self.manifold_out.expmap0(x)
        bias = self.manifold_out.proju0(self.bias.view(1, -1))
        bias = self.manifold_out.proju(x, bias)
        x = self.manifold_out.expmap(x, bias)
        return x



class SphAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(SphAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        x = self.act(self.manifold_in.logmap0(x))
        x = self.manifold_in.proju0(x)
        x = self.manifold_out.expmap0(x)
        return x


class SphNorm(nn.Module):

    def __init__(self, in_features, manifold, method='ln'):
        super(SphNorm, self).__init__()
        self.manifold = manifold
        if method == 'ln':
            self.norm = nn.LayerNorm(in_features)

    def forward(self, h):
        h = self.manifold.logmap0(h)

        h = self.norm(h)
        h = self.manifold.expmap0(h)
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

def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result
