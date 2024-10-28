import torch
import torch.nn as nn
from structure_modules.hyp_layers import HNNLayer,HATLayer,HGCLayer,HSageLayer
from structure_modules.sphere_layers import SphNNLayer,SphGCLayer,SphATLayer,SphSageLayer
from structure_modules.eucl_layers import EucATLayer,EucSageLayer,EucGCLayer
# from manifolds.lorentz import Lorentz
from geoopt.tensor import ManifoldParameter
import math

class HATencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, heads=4, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(HATencoder, self).__init__()
        self.hat1 = HATLayer(in_dim, hidden_dim, manifold_in,manifold_out, dropout,heads=heads)
        self.hat2 = HATLayer(hidden_dim*heads,out_dim, manifold_out,manifold_out, dropout)

    def forward(self, x,adj):
        x = self.hat1(x,adj[0])
        x = self.hat2(x,adj[1])
        return x

class HGCencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(HGCencoder, self).__init__()
        self.hgc1 = HGCLayer(in_dim, hidden_dim, manifold_in,manifold_out, dropout)
        self.hgc2 = HGCLayer(hidden_dim,out_dim, manifold_out,manifold_out, dropout)

    def forward(self, x,adj):
        x = self.hgc1(x,adj[0])
        x = self.hgc2(x,adj[1])
        return x

class HSageencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(HSageencoder, self).__init__()
        self.hsage1 = HSageLayer(in_dim, hidden_dim, manifold_in,manifold_out, dropout)
        self.hsage2 = HSageLayer(hidden_dim,out_dim, manifold_out,manifold_out, dropout)

    def forward(self, x,adj):
        x = self.hsage1(x,adj[0])
        x = self.hsage2(x,adj[1])
        return x

class SphATencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(SphATencoder, self).__init__()
        self.sphat1 = SphATLayer(in_dim, hidden_dim, manifold_in, manifold_out, dropout)
        self.sphat2 = SphATLayer(hidden_dim,out_dim, manifold_out, manifold_out, dropout)

    def forward(self, x,adj):
        x = self.sphat1(x,adj[0])
        x = self.sphat2(x,adj[1])
        return x

class SphGCencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(SphGCencoder, self).__init__()
        self.sphgc1 = SphGCLayer(in_dim, hidden_dim, manifold_in,manifold_out, dropout)
        self.sphgc2 = SphGCLayer(hidden_dim,out_dim, manifold_out,manifold_out, dropout)

    def forward(self, x,adj):
        x = self.sphgc1(x,adj[0])
        x = self.sphgc2(x,adj[1])
        return x

class SphSageencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(SphSageencoder, self).__init__()
        self.sphsage1 = SphSageLayer(in_dim, hidden_dim, manifold_in,manifold_out, dropout)
        self.sphsage2 = SphSageLayer(hidden_dim,out_dim, manifold_out,manifold_out, dropout)

    def forward(self, x,adj):
        x = self.sphsage1(x,adj[0])
        x = self.sphsage2(x,adj[1])
        return x


class EucATencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(EucATencoder, self).__init__()
        self.eucphat1 = EucATLayer(in_dim, hidden_dim,  dropout)
        self.eucphat2 = EucATLayer(hidden_dim,out_dim,  dropout)

    def forward(self, x,adj):
        x = self.eucphat1(x,adj[0])
        x = self.eucphat2(x,adj[1])
        return x

class EucGCencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(EucGCencoder, self).__init__()
        self.eucgc1 = EucGCLayer(in_dim, hidden_dim,  dropout)
        self.eucgc2 = EucGCLayer(hidden_dim,out_dim,  dropout)

    def forward(self, x,adj):
        x = self.eucgc1(x,adj[0])
        x = self.eucgc2(x,adj[1])
        return x

class EucSageencoder(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, manifold_in, manifold_out, dropout=0.1, act=nn.ReLU(),use_norm=True):
        super(EucSageencoder, self).__init__()
        self.Eucsage1 = EucSageLayer(in_dim, hidden_dim, dropout)
        self.Eucsage2 = EucSageLayer(hidden_dim,out_dim, dropout)

    def forward(self, x,adj):
        x = self.Eucsage1(x,adj[0])
        x = self.Eucsage2(x,adj[1])
        return x

