import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(name="data", root="Cora", normalize_features=True):
    """Load the Cora dataset with the standard Planetoid split, row-normalized features."""
    if normalize_features:
        transform = T.NormalizeFeatures()
    else:
        transform = None

    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset.to(DEVICE)
    return data, dataset.num_node_features, dataset.num_classes