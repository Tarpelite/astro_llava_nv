"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from geoopt import PoincareBall
from geoopt import Lorentz

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


class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(),use_norm=True):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_dim, out_dim, manifold_in,manifold_out, dropout)
        self.hyp_act = HypAct(manifold_out, manifold_out, act)
        self.norm = HypNorm(out_dim, manifold_out)
        self.use_norm = use_norm
    def forward(self, x):
        x = self.linear(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.hyp_act(x)
        return x


class HGCLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(),use_norm='ln'):
        super(HGCLayer, self).__init__()
        self.linear = HNNLayer(in_dim, out_dim, manifold_in, manifold_out, dropout)
        self.conv1 = GCNConv(out_dim, out_dim)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = HypNorm(out_dim, manifold_out,use_norm)
        self.hyp_act = HypAct(manifold_out, manifold_out, act)

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

class HSageLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0., act=nn.ReLU(),use_norm='ln'):
        super(HSageLayer, self).__init__()
        self.linear = HNNLayer(in_dim, out_dim, manifold_in,manifold_out, dropout)
        self.sage = SAGEConv(out_dim, out_dim)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = HypNorm(out_dim, manifold_out,use_norm)
        self.hyp_act = HypAct(manifold_out, manifold_out, act)

    def forward(self, x,adj):

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

class HATLayer(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout=0.,heads=4, act=nn.ReLU(),use_norm='ln'):
        super(HATLayer, self).__init__()
        self.linear = HNNLayer(in_dim, out_dim, manifold_in,manifold_out, dropout)
        self.gat = GATConv(out_dim, out_dim,heads=heads)
        self.use_norm = use_norm
        if use_norm != 'none':
            self.norm = HypNorm(out_dim*heads, manifold_out,use_norm)
        self.hyp_act = HypAct(manifold_out, manifold_out, act)

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


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    input in manifold
    output in manifold
    """

    def __init__(self, in_dim, out_dim, manifold_in, manifold_out, dropout):
        super(HypLinear, self).__init__()
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
        x = proj_tan0(x, self.manifold_out)
        x = self.manifold_out.expmap0(x)
        bias = proj_tan0(self.bias.view(1, -1), self.manifold_out)
        bias = self.manifold_out.transp0(x, bias)
        x = self.manifold_out.expmap(x, bias)
        return x




class HypAct(Module):
    """
    Hyperbolic activation layer.
    input in manifold
    output in manifold
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        x = self.act(self.manifold_in.logmap0(x))
        x = proj_tan0(x, self.manifold_in)
        x = self.manifold_out.expmap0(x)
        return x


class HypNorm(nn.Module):

    def __init__(self, in_features, manifold, method='ln'):
        super(HypNorm, self).__init__()
        self.manifold = manifold
        if self.manifold.name == 'Lorentz':
            in_features = in_features - 1
        if method == 'ln':
            self.norm = nn.LayerNorm(in_features)

    def forward(self, h):
        h = self.manifold.logmap0(h)
        if self.manifold.name == 'Lorentz':
            h[..., 1:] = self.norm(h[..., 1:].clone())
        else:
            h = self.norm(h)
        h = self.manifold.expmap0(h)
        return h


def proj_tan0(u, manifold):
    if manifold.name == 'Lorentz':
        narrowed = u.narrow(-1, 0, 1)
        vals = torch.zeros_like(u)
        vals[:, 0:1] = narrowed
        return u - vals
    else:
        return u



