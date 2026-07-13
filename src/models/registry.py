"""Model factory — build any model by name from a Config.

Keeps experiment scripts decoupled from concrete model constructors so new
architectures only need to be registered here.
"""
from __future__ import annotations

import torch.nn as nn

from ..utils.config import Config
from .mlp import MLP
from .gcn import GCN
from .graphsage import GraphSAGE
from .gat import GAT
from .proposed import ResidualMixingGAT

MODEL_NAMES = ["mlp", "gcn", "graphsage", "gat", "proposed"]


def build_model(name: str, in_dim: int, out_dim: int, cfg: Config) -> nn.Module:
    """Instantiate a model by name using hyperparameters from ``cfg``."""
    name = name.lower()
    if name == "mlp":
        return MLP(in_dim, out_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    if name == "gcn":
        return GCN(in_dim, out_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    if name in ("graphsage", "sage"):
        return GraphSAGE(in_dim, out_dim, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    if name == "gat":
        return GAT(in_dim, out_dim, gat_hidden=cfg.gat_hidden, heads=cfg.heads,
                   dropout=cfg.dropout)
    if name == "proposed":
        return ResidualMixingGAT(in_dim, out_dim, gat_hidden=cfg.gat_hidden,
                                 heads=cfg.heads, dropout=cfg.dropout,
                                 drop_edge=cfg.drop_edge, residual=cfg.residual)
    raise ValueError(f"Unknown model {name!r}. Choose from {MODEL_NAMES}.")
