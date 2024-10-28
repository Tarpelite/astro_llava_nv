import torch
from structure_modules.models import HSageencoder,SphSageencoder,EucGCencoder,HGCencoder,SphGCencoder,EucSageencoder
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import geoopt
from manifolds.euclidean import Euclidean
from manifolds.sphere import Sphere
dataset = Planetoid("/data/Cora", name="Cora", transform=T.NormalizeFeatures())
print(dataset)
print(dataset.x.shape)
print(dataset.edge_index.shape)
# manifold_out = geoopt.PoincareBall()
manifold_in = Euclidean()
manifold_out = Sphere()
model = SphGCencoder(in_dim=1433, hidden_dim=512, out_dim=256, manifold_in=manifold_in,manifold_out=manifold_out)
manifold_in = Euclidean()
manifold_out = geoopt.PoincareBall()
model = HGCencoder(in_dim=1433, hidden_dim=512, out_dim=256, manifold_in=manifold_in,manifold_out=manifold_out)
manifold_in = Euclidean()
manifold_out = Euclidean()
model = EucSageencoder(in_dim=1433, hidden_dim=512, out_dim=256, manifold_in=manifold_in,manifold_out=manifold_out)
# print(manifold.logmap0(dataset.x))
#
y = model(dataset.x, dataset.edge_index)
print(y.shape)