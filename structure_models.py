from structure_modules.models import HSageencoder,SphSageencoder,EucGCencoder,HGCencoder,SphGCencoder,EucSageencoder
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import geoopt
from manifolds.euclidean import Euclidean
from manifolds.sphere import Sphere


class StructureEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.euc_encoder = EucSageencoder(in_dim=1433, hidden_dim=512, out_dim=256, manifold_in=Euclidean(),manifold_out=Euclidean())
        self.sph_encoder = SphGCencoder(in_dim=1433, hidden_dim=512, out_dim=256, manifold_in=Euclidean(),manifold_out=Sphere())
        self.hgc_encoder = HGCencoder(in_dim=1433, hidden_dim=512, out_dim=256, manifold_in=Euclidean(),manifold_out= geoopt.PoincareBall())

    
    def forward(self,
            node_features=None, #[n_node, 1433]
            euc_edge_index=None, #[2, num_edges]
            sph_edge_index=None, #[2, num_edges]
            hgc_edge_index=None, #[2, num_edges]
            target_node_idx=None #[2, num_edges]
            ):
        """Note: No support for batchify, one graph each time"""
        
        euc_features = self.euc_encoder(node_features, euc_edge_index) #
        target_euc_feature = euc_features[target_node_idx] #[out_dim]
        sph_features = self.sph_encoder(node_features, sph_edge_index)
        target_sph_feature = sph_features[target_node_idx] #[out_dim]
        hgc_features = self.hgc_encoder(node_features, hgc_edge_index)
        target_hgc_feature = hgc_features[target_node_idx] #[out_dim]
        out_features = torch.stack([euc_features, sph_features, hgc_features])

        return out_features


