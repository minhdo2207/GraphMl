"""Two-layer GraphSAGE (Hamilton et al., 2017).

Aggregates neighbour representations and combines them with each node's own
representation. Provides a second message-passing architecture to compare
against GCN and GAT.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64,
                 dropout: float = 0.5, aggr: str = "mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr=aggr)
        self.conv2 = SAGEConv(hidden_dim, out_dim, aggr=aggr)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
