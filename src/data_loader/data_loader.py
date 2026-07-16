"""Cora dataset loader with standard Planetoid split and optional feature normalization."""
from __future__ import annotations

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def get_data(name="Cora", root="data", normalize_features=True):
    """Load a Planetoid dataset with the standard split, optionally row-normalized features.

    Data is returned on CPU; the caller (trainer/runner) is responsible for
    moving it to the appropriate device.
    """
    transform = T.NormalizeFeatures() if normalize_features else None
    dataset = Planetoid(root=root, name=name, transform=transform)
    data = dataset[0]
    return data, dataset.num_node_features, dataset.num_classes


def print_statistics(data, num_features: int, num_classes: int, name: str = "Cora") -> None:
    """Print dataset summary: nodes, edges, features, classes, and mask sizes."""
    print(f"Dataset: {name}")
    print(f"  Nodes:       {data.num_nodes}")
    print(f"  Edges:       {data.num_edges}")
    print(f"  Features:    {num_features}")
    print(f"  Classes:     {num_classes}")
    print(f"  Train mask:  {int(data.train_mask.sum())}")
    print(f"  Val mask:    {int(data.val_mask.sum())}")
    print(f"  Test mask:   {int(data.test_mask.sum())}")