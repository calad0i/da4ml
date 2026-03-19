# Getting Started with da4ml

da4ml can be used in multiple ways. When standalone code generation, it is recommended to use the functional API or HGQ2 integration. See [FAQ](faq.md) for more details on when to use which flow.

## functional API:

The most flexible way to use da4ml is through its functional API/Explicit symbolic tracing. This allows users to define arbitrary operations using numpy-like syntax, and then trace the operations to generate synthesizable HDL or HLS code.

```python
# da4ml standalone example
import numpy as np

from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import einsum, quantize, relu
from da4ml.codegen import HLSModel, RTLModel

w = np.random.randint(-2**7, 2**7, (4, 5, 6)) / 2**7

def operation(inp):
   inp = quantize(inp, 1, 7, 0) # Input must be quantized before any non-trivial operation
   out1 = relu(inp) # Only activation supported for now; can attach quantization at the same time

   # many native numpy operations are supported
   out2 = inp[:, 1:3].transpose()
   out2 = quantize(np.sin(out2), 1, 0, 7, 'SAT', 'RND')
   out2 = np.repeat(out2, 2, axis=0) * 3 + 4
   out2 = np.amax(np.stack([out2, -out2 * 2], axis=0), axis=0)

   out3 = quantize(out2 @ out1, 1, 10, 2) # can also be einsum here
   out = einsum('ijk,ij->ik', w, out3) # CMVM optimization is performed for all
   return out

# Replay the operation on symbolic tensor
inp = FixedVariableArrayInput((4, 5))
out = operation(inp)

# Generate pipelined Verilog code form the traced operation
# flavor can be 'verilog' or 'vhdl'. VHDL code generated will be in 2008 standard.
comb_logic = comb_trace(inp, out)
rtl_model = RTLModel(comb_logic, '/tmp/rtl', flavor='verilog', latency_cutoff=5)
rtl_model.write()
# rtl_model.compile() # compile the generated Verilog code with verilator (with GHDL, if using vhdl)
# rtl_model.predict(data_inp) # run inference with the compiled model; bit-accurate

# Run bit-exact (all int64 arithmetic) inference with the combinational logic model
# Backed by C++-based DAIS interpreter for speed
# comb_logic.predict(data_inp)
```

## Using external plugins:

da4ml supports a plugin system for external frameworks. A plugin can implement the logic for tracing models defined in a specific framework (e.g., Keras3/HGQ2, PyTorch, etc.) and register itself under the `dais_tracer.plugins` entry point. When tracing a model, da4ml will automatically discover the appropriate plugin based on the model type and use it for tracing. See [Conversion Plugin](plugin.md) for more details. Below are some examples.


## HGQ2/Keras3 integration:

For models defined in [HGQ2](https://github.com/calad0i/HGQ2) (Keras3 based), da4ml can trace the model operations automatically when the supported layers/operations are used (i.e., most HGQ2 layers without general non-linear activations). In this way, one can easily convert existing HGQ2 models to HDL or HLS code in seconds. The plugin is built-in in HGQ2, so installing HGQ2 is sufficient to enable the integration. No additional configuration is needed.

> **Note**: HGQ2 support requires installing the [HGQ2](https://github.com/calad0i/HGQ2) package separately. HGQ2 registers its own `dais_tracer.plugins` entry point under the `keras` key, which da4ml discovers automatically. `trace_model()` auto-detects the framework from `type(model).__module__.split('.', 1)[0]`, so a Keras model resolves to `'keras'` and the HGQ2 plugin is used. See [Conversion Plugin](plugin.md) for how the plugin system works.

```python
# da4ml with HGQ2
import numpy as np
import keras
from hgq.layers import QEinsumDenseBatchnorm, QMaxPool1D
from da4ml.codegen import HLSModel, RTLModel
from da4ml.converter import trace_model
from da4ml.trace import comb_trace

inp = keras.Input((4, 5))
out = QEinsumDenseBatchnorm('bij,jk->bik', (4,6), bias_axes='k', activation='relu')(inp)
out1 = QMaxPool1D(pool_size=2)(out)
out = keras.ops.concatenate([out, out1], axis=1)
out1, out2 = out[:, :3], out[:, 3:]
out = keras.ops.einsum('bik,bjk->bij', out1, out1 - out2[:,:1])
model = keras.Model(inp, out)

# Automatically replay the model operation on symbolic tensors
inp, out = trace_model(model)

comb_logic = comb_trace(inp, out)

... # The rest is the same as above
```

## RTL/HLS backends:

### RTL (Verilog/VHDL)

`RTLModel` generates synthesizable RTL and wraps a compiled simulation emulator for bit-accurate inference:

```python
from da4ml.codegen import RTLModel

# flavor='verilog' uses Verilator for simulation;
# 'vhdl' uses GHDL and Verilator chained (GHDL for VHDL to Verilog conversion, then Verilator for simulation)
rtl_model = RTLModel(comb_logic, '/tmp/rtl', flavor='verilog', latency_cutoff=5, clock_period=5.0)
rtl_model.write()        # write RTL project to disk
rtl_model.compile()      # compile simulation emulator (requires Verilator or GHDL)
y = rtl_model.predict(x) # bit-accurate inference via compiled emulator
```

The generated project includes TCL build scripts for Vivado (`build_vivado_prj.tcl`) and Quartus (`build_quartus_prj.tcl`). Both `CombLogic` and `Pipeline` are supported.

### HLS (Vitis / HLSlib / oneAPI)

`HLSModel` generates HLS C++ code and wraps a compiled emulator:

```python
from da4ml.codegen import HLSModel

# flavor='vitis' (ap_types), 'hlslib' (ac_types/Intel), or 'oneapi'
hls_model = HLSModel(comb_logic, '/tmp/hls', flavor='vitis', clock_period=5.0)
hls_model.write()        # write HLS project to disk
hls_model.compile()      # compile C++ emulator
y = hls_model.predict(x) # inference via compiled emulator
```

Note: only `CombLogic` is supported for HLS backends; `Pipeline` is not.


## XLS backend (experimental):


```{note}
`xls-python`, a python binding for `libxls.so`, is required for the XLS backend. It can be installed from PyPI with `pip install xls-python`, but only available for Linux-x86_64 for the binary wheel. For other platforms, you may need to build `libxls` and `xls-python` from source.
```

For generating Verilog through [XLS](https://google.github.io/xls/), an experimental backend is available. This requires the `pyxls` package (experimental) to be installed separately.

```python
from da4ml.codegen.xls import XLSModel

xls_model = XLSModel(comb_logic) # Converts DAIS to XLS IR.
_ = xls_model.jit() # JIT-compile the XLS IR
y = xls_model.predict(data_inp) # Batched inference; bit-exact. No multithreading support for now.
verilog_text = xls_model.compile('/tmp/xls_output.v')
```

## CLI usage:

da4ml provides a command-line interface for common workflows:

```bash
# Convert a Keras/HGQ2 model to an RTL project
da4ml convert model.keras /tmp/rtl_output --flavor verilog --latency-cutoff 5

# Convert a serialized DAIS model (JSON) to an RTL project
da4ml convert model.json /tmp/rtl_output --flavor vhdl

# Generate a resource/timing report from an existing RTL project
da4ml report /tmp/rtl_output
```

Use `da4ml convert --help` and `da4ml report --help` for full option details.

## hls4ml integration:

For existing uses of [hls4ml](https://github.com/fastmachinelearning/hls4ml), da4ml can be used as a strategy provider to enable the `distributed_arithmetic` strategy for supported layers (e.g., Dense, Conv, EinsumDense). This leverages the HLS codegen backend in da4ml to generate only the CMVM part of the design, while still using hls4ml for the rest of the design and integration. For any design aiming for `II>1` (i.e., not-fully unrolled), this is the recommended way to use da4ml.

```python
# da4ml with hls4ml
from hls4ml.converters import convert_from_keras_model

model_hls = convert_from_keras_model(
   model,
   hls_config={'Model': {'Strategy': 'distributed_arithmetic', ...}, ...},
   ...
)

model_hls.write()
```
