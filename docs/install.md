# Installation Guide

`da4ml` is available on PyPI and can be installed using pip.

```bash
pip install da4ml
```

## Requirements

  * **Python:** `da4ml` requires `Python>=3.10`.
  * **Numba:** The project relies on `numba>=0.61` for its just-in-time (JIT) compilation to achieve high performance. If you encounter compilation issues, try upgrading `numba` and `llvmlite` to their latest versions: `pip install --upgrade numba llvmlite`.
