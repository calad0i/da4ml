# Project Status

da4ml is currently **beta** software and is under active development. Its core functionality is stable, but some features/API regarding tracing individual operations may change in future releases. We welcome contributions and feedback from the community through issues and pull requests.

```{note}
It is advised to always verify the results produced from the generated code.
```

## Supported Operations

Most common high-level operations can be represented in [DAIS](dais.md) is supported, including but not limited to:
 - Dense/Convolutional/EinsumDense layers
 - ReLU
 - max/minimum of two tensors; max/min pooling
 - element-wise addition/subtraction/multiplication
 - rearrangement of tensors (reshape, transpose, slicing, etc.)
 - fixed-point quantization
 - Arbitrary unary mapping through logic lookup tables (not to be confused with LUT primitives)
 - Sorting operations (sorting networks via bitonic/odd-even merge sort)
 - Bitwise operations (AND, OR, XOR, NOT, reduce-any, reduce-all)

```{note}
An experimental [XLS](https://google.github.io/xls/) backend is available for generating Verilog through the XLS toolchain. See the [XLS backend documentation](xls.md) for details.
```
