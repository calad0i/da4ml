# Project Status

da4ml is currently **beta** software and is under active development. Its core functionality is stable, but some features/API regarding tracing individual operations may change in future releases. We welcome contributions and feedback from the community through issues and pull requests.

:::{note}
It is advised to always verify the results produced from the generated code.
:::

## Supported Operations

Most common high-level operations can be represented in [DAIS](dais.md) is supported, including but not limited to:
 - Dense/Convolutional/EinsumDense layers
 - ReLU
 - max/minimum of two tensors; max/min pooling
 - element-wise addition/subtraction/multiplication
 - rearrangement of tensors (reshape, transpose, slicing, etc.)
 - fixed-point quantization
 - Arbitrary unary mapping through logic lookup tables (no to be confused with LUT primitives)


## Unsupported Operations
 - Anything requires stateful operations/time dependencies: due to the SSA nature of DAIS, we do not plan to support stateful operations (e.g., not-unrolled RNNs) within the da4ml framework. We believe these operations shall be implemented in higher-level frameworks that orchestrate the DAIS programs.
 - Bit-exact integer division: We do not support real division or bit-exact integer division. Users are advised to approximate division with multiplication and inverse table lookup where necessary.
