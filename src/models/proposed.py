"""Residual Multi-Head Mixing GAT (proposed model).

Combines multi-head attention with an input-feature skip connection to counteract
over-smoothing, plus optional DropEdge for structural regularization:

    H0 = X
    H1 = ELU( GATConv_multi(H0, E) )     # K heads, concatenated (one-hop mixing)
    H2 = GATConv_single(H1, E)           # single head          (two-hop)
    Z  = H2 + H0 @ W_skip                # residual from raw features -> logits

The ``residual`` and ``drop_edge`` switches exist so the ablation study can turn
each contribution on/off independently (Person 4).

Note: ``Z`` is returned as raw logits (no softmax); the training loss is
``F.cross_entropy``, which applies log-softmax internally.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge


class ResidualMixingGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, gat_hidden: int = 8,
                 heads: int = 8, dropout: float = 0.6, drop_edge: float = 0.2,
                 residual: bool = True):
        super().__init__()
        self.dropout = dropout
        self.drop_edge = drop_edge
        self.residual = residual

        # H1: multi-head attention over one-hop neighbours (concatenated heads)
        self.gat_multi = GATConv(in_dim, gat_hidden, heads=heads, dropout=dropout)
        # H2: single-head attention projecting to class logits
        self.gat_single = GATConv(gat_hidden * heads, out_dim, heads=1,
                                  concat=False, dropout=dropout)
        # H0 @ W_skip: linear skip connection carrying raw features to the output
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if residual else None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h0 = x
        ei = edge_index
        # DropEdge: randomly remove a fraction of edges during training only.
        if self.training and self.drop_edge > 0:
            ei, _ = dropout_edge(edge_index, p=self.drop_edge)

        h = F.dropout(h0, p=self.dropout, training=self.training)
        h1 = F.elu(self.gat_multi(h, ei))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = self.gat_single(h1, ei)

        if self.residual and self.skip is not None:
            return h2 + self.skip(h0)
        return h2
