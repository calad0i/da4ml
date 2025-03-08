import numpy as np

from .api import cost, fn_from_kernel
from .cmvm import compile_kernel
from .codegen import PyCodegenBackend
from .graph_compile import graph_compile_states
from .utils import DAState, OpCode, Score

d_in = 2
d_out = 2
kernel = np.ones((d_in, d_out), dtype=np.float32)
signs = [False] * d_in
bits = [8] * d_in
int_bits = [0] * d_in
symmetrics = [False] * d_in
depths = [0] * d_in

print('The da4ml library is compiling. This will take a while...', end=' ')
_ = fn_from_kernel(
    kernel=kernel,
    signs=signs,
    bits=bits,
    int_bits=int_bits,
    symmetrics=symmetrics,
    depths=depths,
    n_beams=1,
    dc=None,
    n_inp_max=-1,
    n_out_max=-1,
    codegen_backend=PyCodegenBackend(),
)
print('Done')


__all__ = ['DAState', 'OpCode', 'Score', 'cost', 'compile_kernel', 'fn_from_kernel', 'graph_compile_states']
