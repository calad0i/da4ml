# da4ml: Distributed Arithmetic for Machine Learning

This project performs Constant Matrix-Vector Multiplication (CMVM) with Distributed Arithmetic (DA) for Machine Learning (ML) on a Field Programmable Gate Arrays (FPGAs).

CMVM optimization is done through greedy CSE of two-term subexpressions, with possible Delay Constraints (DC). The optimization is done in jitted Python (Numba), and a list of optimized operations is generated as traced Python code.

The project generates Verilog or Vitis HLS code for the optimized CMVM operations. This project can be used in conjunction with [`hls4ml`](https://github.com/fastmachinelearning/hls4ml/) for optimizing the neural networks deployed on FPGAs. For a subset of neural networks, the full design can be generated standalone in Verilog or Vitis HLS.


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

## Standalone usage

### `HGQ2`

For some models trained with `HGQ2`, the `da4ml` can be used to generate the whole model in Verilog or Vitis HLS:

```python
from da4ml.codegen import HLSModel, VerilogModel
from da4ml.converter.hgq2.parser import trace_model
from da4ml.trace import comb_trace

inp, out = trace_model(hgq2_model)
comb_logic = comb_trace(inp[0], out[0]) # Currently, only models with 1 input and 1 output are supported

# Pipelined Verilog model generation
# `latency_cutoff` is used to control auto piplining behavior. To disable pipelining, set it to -1.
verilog_model = VerilogModel(sol, prj_name='barbar', path='/tmp/barbar', latency_cutoff=5)
verilog_model.compile() # write and verilator binding
verilog_model.predict(inputs)

vitis_hls_model = HLSModel(sol, prj_name='foo', path='/tmp/foo', flavor='vitis') # Only vitis is supported for now
vitis_hls_model.compile() # write and hls binding
vitis_hls_model.predict(inputs)
```

### Functional Definition
For generic operations, one can define a combinational logic with the functional API:

```python
from da4ml.trace import FixedVariableArray, HWConfig, comb_trace
from da4ml.trace.ops import einsum, relu, quantize, conv, pool

# k, i, f are numpy arrays of integers: keep_negative (0/1), integer bits (excl. sign), fractional bits
inp = FixedVariableArray.from_kif(k, i, f, HWConfig(1, -1, -1), solver_options={'hard_dc':2})
out = inp @ kernel
out = relu(out)
out = einsum(equation, out, weights)
...

comb = comb_trace(inp, out)
```

`+`, `-`, `@` are supported as well as `einsum`, `relu`, `quantize` (WRAP, with TRN or RND), `conv`, `pool` (average only). For multiplications, only power-of-two multipliers are supported, otherwise use `einsum` or `@` operators.

The `comb_trace` returns a `Solution` objects that contains a list of low-level operations that are used to implement the combinational logic, which in turn can be used to generate Verilog or Vitis HLS code.
