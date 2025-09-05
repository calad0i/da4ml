# Getting Started with da4ml

da4ml can be used in three different ways. When standalone code generation, it is recommended to use the functional API or HGQ2 integration. See [FAQ](./faq.html) for more details on when to use which flow.

## functional API:

The most flexible way to use da4ml is through its functional API/Explicit symbolic tracing. This allows users to define arbitrary operations using numpy-like syntax, and then trace the operations to generate synthesizable HDL or HLS code.

```python
# da4ml standalone example
import numpy as np

from da4ml.trace import FixedVariableArrayInput, comb_trace
from da4ml.trace.ops import einsum, quantize, relu
from da4ml.codegen import HLSModel, VerilogModel

w = np.random.randint(-2**7, 2**7, (4, 5, 6)) / 2**7

def operation(inp):
   inp = quantize(inp, 1, 7, 0) # Input must be quantized before any non-trivial operation
   out1 = relu(inp) # Only activation supported for now; can attach quantization at the same time

   # many native numpy operations are supported
   out2 = inp[:, 1:3].transpose()
   out2 = np.repeat(out2, 2, axis=0) * 3 + 4
   out2 = np.amax(np.stack([out2, -out2 * 2], axis=0), axis=0)

   out3 = quantize(out2 @ out1, 1, 10, 2) # can also be einsum here
   out = einsum('ijk,ij->ik', w, out3) # CMVM optimization is performed for all
   return out

# Replay the operation on symbolic tensor
inp = FixedVariableArrayInput((4, 5))
out = operation(inp)

# Generate pipelined Verilog code form the traced operation
comb_logic = comb_trace(inp, out)
verilog_model = VerilogModel(comb_logic, 'vmodel', '/tmp/verilog', latency_cutoff=5) # can also be HLSModel
verilog_model.write()
# verilog_model.compile() # compile the generated Verilog code with verilator
# verilog_model.predict(data_inp) # run inference with the compiled model; bit-accurate
```

## HGQ2/Keras3 integration:

For models defined in [HGQ2](https://github.com/calad0i/HGQ2) (Keras3 based), da4ml can trace the model operations automatically when the supported layers/operations are used (i.e., most HGQ2 layers without general non-linear activations). In this way, one can easily convert existing HGQ2 models to HDL or HLS code in seconds.

```python
# da4ml with HGQ2
import numpy as np
import keras
from hgq.layers import QEinsumDenseBatchnorm, QMaxPool1D
from da4ml.codegen import HLSModel, VerilogModel
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
