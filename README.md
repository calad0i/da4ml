# da4ml: Distributed Arithmetic for Machine Learning

This project performs Constant Matrix-Vector Multiplication (CMVM) with Distributed Arithmetic (DA) for Machine Learning (ML) on a Field Programmable Gate Arrays (FPGAs).

CMVM optimization is done through greedy CSE of two-term subexpressions, with possible Delay Constraints (DC). The optimization is done in jitted Python (Numba), and a list of optimized operations is generated as traced Python code.

At the moment, the project only generates Vitis HLS C++ code for the FPGA implementation of the optimized CMVM kernel. HDL code generation is planned for the future. Currently, the major use of this repository is through the `distributed_arithmetic` strategy in the [`hls4ml`](https://github.com/fastmachinelearning/hls4ml/) project.


## Installation

The project is available on PyPI and can be installed with pip:

```bash
pip install da4ml
```

Notice that `numba>=6.0.0` is required for the project to work. The project does not work with `python<3.10`. If the project fails to compile, try upgrading `numba` and `llvmlite` to the latest versions.

## `hls4ml`

The major use of this project is through the `distributed_arithmetic` strategy in the `hls4ml`:

```python
model_hls = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config={
        'Model': {
            ...
            'Strategy': 'distributed_arithmetic',
        },
        ...
    },
    ...
)
```

Currently, `Dense/Conv1D/Conv2D` layers are supported for both `io_parallel` and `io_stream` dataflows. However, notice that distributed arithmetic implies `reuse_factor=1`, as the whole kernel is implemented in combinational logic.

### Notice

Currently, only the `da4ml-v3` branch of `hls4ml` supports the `distributed_arithmetic` strategy. The `da4ml-v3` branch is not yet merged into the `main` branch of `hls4ml`, so you need to install it from the GitHub repository.
