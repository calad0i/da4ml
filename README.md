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

Currently, only the `da4ml-v2` branch of `hls4ml` supports the `distributed_arithmetic` strategy. The `da4ml-v2` branch is not yet merged into the `main` branch of `hls4ml`, so you need to install it from the GitHub repository.

## Direct Usage

If you want to use it directly, you can use the `da4ml.api.fn_from_kernel` function, which creates a Python function from a 2x2 kernel `float[n_in, n_out]` and its corresponding code. The function signature is:

```python
def fn_from_kernel(
    kernel: np.ndarray,
    signs: list[bool],
    bits: list[int],
    int_bits: list[int],
    symmetrics: list[bool],
    depths: list[int] | None = None,
    n_beams: int = 1,
    dc: int | None = None,
    n_inp_max: int = -1,
    n_out_max: int = -1,
    codegen_backend: PyCodegenBackend = PyCodegenBackend(),
    signed_balanced_reduction: bool = True,
) -> tuple[Callable[[list[T]], list[T]], str]:
    """Compile a CMVM operation, with the constant kernel, into a function with only accumulation/subtraction/shift operations.

    Parameters
    ----------
    kernel : np.ndarray
        The kernel to compile. Must be of shape (n_inp, n_out).
    signs : list[bool]
        If the input is signed. Must be of length n_inp.
    bits : list[int]
        The bitwidth of the inputs. Must be of length n_inp.
    int_bits : list[int]
        The number of integer bits in the inputs (incl. sign bit!). Must be of length n_inp.
    symmetrics : list[bool]
        If the input is symmetricly quantized. Must be of length n_inp.
    depths : list[int]|None, optional
        The depth associated with each input. Must be of length n_inp. Defaults to [0]*n_inp.
    n_beams : int, optional
        Number of beams to use in beam search. Defaults to 1. (Currently disabled!)
    dc : int | None, optional
        Delay constraint. Not implemented yet. Defaults to None.
    n_inp_max : int, optional
        Number of inputs to process in one block. Defaults to -1 (no limit). Decrease to improve performance, but result will be less optimal.
    n_out_max : int, optional
        Number of outputs to process in one block. Defaults to -1 (no limit). Decrease to improve performance, but result will be less optimal.
    codegen_backend : PyCodegenBackend, optional
        The codegen backend to be used. Defaults to PyCodegenBackend().
    signed_balanced_reduction : bool, optional
        If the reduction tree should isolate the plus and minus terms. Set to False to improve latency. Defaults to True.

    Returns
    -------
    tuple[Callable[[list[T]], list[T]], str]
        fn : Callable[[list[T]], list[T]]
            The compiled python function. It takes a list of inputs and returns a list of outputs with only accumulation/subtraction/powers of 2 operations.
        fn_str : str
            The code of the compiled function, depending on the codegen_backend used.
    """
```
