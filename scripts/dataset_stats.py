"""Print Cora dataset statistics and mask sanity checks.

Usage:
    python scripts/dataset_stats.py
"""
from __future__ import annotations

import os
import sys

# make ``src`` importable when run as a standalone script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data, print_statistics  # noqa: E402


def main() -> None:
    data, num_features, num_classes = get_data(name="Cora")
    print_statistics(data, num_features, num_classes, name="Cora")


if __name__ == "__main__":
    main()
