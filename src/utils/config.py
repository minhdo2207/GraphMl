"""Experiment configuration.

A single dataclass holds every hyperparameter so experiments stay reproducible
and self-documenting. Defaults mirror ``configs/default.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any, Dict


@dataclass
class Config:
    # --- data ---
    dataset: str = "Cora"
    data_root: str = "data"
    normalize_features: bool = True

    # --- model ---
    model: str = "gcn"
    hidden_dim: int = 64          # hidden size for MLP/GCN/SAGE
    heads: int = 8               # attention heads for GAT / proposed
    gat_hidden: int = 8          # per-head hidden size for GAT / proposed
    dropout: float = 0.6
    drop_edge: float = 0.2       # DropEdge probability (proposed model)
    residual: bool = True        # enable H0 @ W_skip residual (proposed model)

    # --- optimisation ---
    lr: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 100          # early-stopping patience on val accuracy (0 = off)

    # --- experiment ---
    seed: int = 42
    seeds: int = 5               # number of seeds for multi-run averaging
    device: str = "auto"         # "auto" | "cpu" | "cuda"

    def update(self, **kwargs: Any) -> "Config":
        """Return a copy with the given fields overridden."""
        valid = {f.name for f in fields(self)}
        data = asdict(self)
        for k, v in kwargs.items():
            if v is None:
                continue
            if k not in valid:
                raise KeyError(f"Unknown config field: {k!r}")
            data[k] = v
        return Config(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_DEFAULT_YAML = "configs/default.yaml"


def load_config(path: str | None = _DEFAULT_YAML, **overrides: Any) -> Config:
    """Load a Config from a YAML file (default: configs/default.yaml), then apply overrides.

    Falls back to dataclass defaults silently when the default YAML is missing.
    Raises FileNotFoundError only when an explicit path does not exist.
    """
    import os

    cfg = Config()
    explicit = path is not None and path != _DEFAULT_YAML
    path = path or _DEFAULT_YAML

    if os.path.isfile(path):
        try:
            import yaml  # optional dependency
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pyyaml is required to load YAML configs") from exc
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg = cfg.update(**data)
    elif explicit:
        raise FileNotFoundError(f"Config file not found: {path}")

    return cfg.update(**overrides)
