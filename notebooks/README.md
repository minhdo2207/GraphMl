# Notebooks

Exploratory and Colab notebooks live here.

To use the `src/` package from a Colab notebook:

```python
import sys, os
sys.path.append("/content/GraphMl")   # repo root after `git clone`

from src.data import print_statistics
from src.utils import load_config
from src.training import run_seeds

print_statistics(name="Cora")

cfg = load_config()
summary = run_seeds(cfg, model_name="proposed", verbose=True)
print(summary.as_row())
```

Reference Colab (shared in the proposal):
https://colab.research.google.com/drive/1snPK7GouMrluqdbNmxaZ9li-g0S-hQtx
