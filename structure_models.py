from structure_modules.models import HSageencoder,SphSageencoder,EucGCencoder,HGCencoder,SphGCencoder,EucSageencoder
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import geoopt
from manifolds.euclidean import Euclidean
from manifolds.sphere import Sphere


class StructureEncoder(nn.Module):
    def __init__(self,
                in_dim=1433,
                hidden_dim=512,
                out_dim=256):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.euc_encoder = EucSageencoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, manifold_in=Euclidean(),manifold_out=Euclidean())
        self.sph_encoder = SphSageencoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, manifold_in=Euclidean(),manifold_out=Sphere())
        self.hgc_encoder = HSageencoder(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, manifold_in=Euclidean(),manifold_out= geoopt.PoincareBall())

    
    def forward(self,
            node_features=None, #[n_node, in_dim]
            edge_index_list = None, # here is a list containing [spectrum edge, ra/dec edge]
            target_node_idx=None #[2, num_edges]
            ):
        """Note: No support for batchify, one graph each time"""
        
        euc_features = self.euc_encoder(node_features, edge_index_list) #
        target_euc_feature = euc_features[target_node_idx] #[out_dim]
        sph_features = self.sph_encoder(node_features, edge_index_list)
        target_sph_feature = sph_features[target_node_idx] #[out_dim]
        hgc_features = self.hgc_encoder(node_features, edge_index_list)
        target_hgc_feature = hgc_features[target_node_idx] #[out_dim]
        out_features = torch.stack([euc_features, sph_features, hgc_features])

        return out_features


