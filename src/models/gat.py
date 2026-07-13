"""Vanilla two-layer Graph Attention Network (Velickovic et al., 2018).

    h_i^{l+1} = ||_k  sigma( sum_{j in N(i)} alpha_ij^k  W^k h_j^l )

Layer 1 uses ``heads`` attention heads (concatenated); the output layer uses a
single averaged head projecting to the class logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gat_hidden: int = 8,
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_dim, gat_hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(gat_hidden * heads, out_dim, heads=1,
                             concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
