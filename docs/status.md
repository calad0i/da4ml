# Project Status

da4ml is currently **beta** software and is under active development. Its core functionality is stable, but some features/API regarding tracing individual operations may change in future releases. We welcome contributions and feedback from the community through issues and pull requests.

```info
It is adviced to always verify the results produced from the generated code.
```

## Supported Operations

Most common high-level operations can be represented in [DAIS](dais.html) is supported, including but not limited to:
 - Dense/Convolutional/EinsumDense layers
 - ReLU
 - max/minimum of two tensors; max/min pooling
 - element-wise addition/subtraction/multiplication
 - rearrangement of tensors (reshape, transpose, slicing, etc.)
 - fixed-point quantization


## Unsupported Operations
 - general non-linear activation functions: each instruction in DAIS supports only a 64-bit data size, so general non-linear functions are not supported yet. In the future, general non-linear unary operations may be added as an external table lookup attached to the DAIS program.
 - Anything requires stateful operations/time dependencies: due to the SSA nature of DAIS, we do not plan to support stateful operations (e.g., not-unrolled RNNs) within the da4ml framework. We believe these operations shall be implemented in higher-level frameworks that orchestrate the DAIS programs.
 - Division: division is not yet supported in DAIS. Albeit not impossible, we do not have short term plans to support it.
