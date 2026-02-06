# da4ml: Distributed Arithmetic for Machine Learning

<!-- [![LGPLv3](https://img.shields.io/badge/License-LGPLv3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) -->
[![Tests](https://img.shields.io/github/actions/workflow/status/calad0i/da4ml/unit-test.yml?label=test)](https://github.com/calad0i/da4ml/actions/workflows/unit-test.yml)
[![Documentation](https://img.shields.io/github/actions/workflow/status/calad0i/da4ml/sphinx-build.yml?label=doc)](https://calad0i.github.io/da4ml/)
[![PyPI version](https://img.shields.io/pypi/v/da4ml)](https://pypi.org/project/da4ml/)
[![ArXiv](https://img.shields.io/badge/arXiv-2507.04535-b31b1b.svg)](https://arxiv.org/abs/2507.04535)
[![Cov](https://img.shields.io/codecov/c/github/calad0i/da4ml)](https://codecov.io/gh/calad0i/da4ml)

da4ml is a static computation graph to RTL/HLS design compiler targeting ultra-low latency applications on FPGAs. It as two major components:
 - A fast and performant constant-matrix-vector multiplications (CMVM) optimizer to implement them as
   efficient adder trees. Common sub-expressions elimination (CSE) with graph-based pre-optimization are
   performed to reduce the firmware footprint and improve the performance.
 - Low-level symbolic tracing frameworks for generating combinational/fully pipelined logics in HDL or HLS
   code. da4ml can generate the firmware for almost all fully pipelined networks standalone.
   Alternatively, da4ml also be used as a plugin in hls4ml to optimize the CMVM operations in the network.

Key Features
------------

- **Optimized Algorithms**: Comparing to hls4ml's latency strategy, da4ml's CMVM implementation uses no DSO and consumes up to 50% less LUT usage.
- **Fast code generation**: da4ml can generate HDL for a fully pipelined network in seconds. For the same models, high-level synthesis tools like Vivado/Vitis HLS can take up to days to generate the HDL code.
- **Low-level symbolic tracing**: As long as the operation can be expressed by a combination of the low-level operations supported, adding new operations is straightforward by "replaying" the operation on the symbolic tensor provided. In most cases, adding support for a new operation/layer takes just a few lines of code in numpy flavor.
- **Automatic model conversion**: da4ml can automatically convert models trained in `HGQ2 <https://github.com/calad0i/hgq2>`_.
- **Bit-accurate Emulation**: All operation in da4ml is bit-accurate, meaning the generated HDL code will produce the same output as the original model. da4ml's computation is converted to a RISC-like, instruction set level intermediate representation, distributed arithmetic instruction set (DAIS), which can be easily simulated in multiple ways. da4ml also provides a fast C++ based DAIS interpreter to run bit-exact inference on the traced models for verification and benchmarking.
- **hls4ml integration**: da4ml can be used as a extension in hls4ml to optimize the CMVM operations in the network by setting `strategy='distributed_arithmetic'` for the strategy of the Dense, EinsumDense, or Conv1/2D layers.

Installation
------------

```bash
pip install da4ml
```

Note: da4ml is now released as binary wheels on PyPI for Linux X86_64 and MacOS ARM64 platforms. For other platforms, please install from source.

Getting Started
---------------

- See the [Getting Started](https://calad0i.github.io/da4ml/getting_started.html) guide for a quick introduction to using da4ml.
- See [JEDI-linear](https://github.com/calad0i/JEDI-linear) project which is based on da4ml

## License

LGPLv3. See the [LICENSE](LICENSE) file for details.

## Citation

If you use da4ml in a publication, please cite our [TRETS'25 paper](https://doi.org/10.1145/3777387) with the following bibtex entry:

```bibtex
@article{sun2025da4ml,
    author = {Sun, Chang and Que, Zhiqiang and Loncar, Vladimir and Luk, Wayne and Spiropulu, Maria},
    title = {da4ml: Distributed Arithmetic for Real-time Neural Networks on FPGAs},
    year = {2025},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    issn = {1936-7406},
    url = {https://doi.org/10.1145/3777387},
    doi = {10.1145/3777387},
    journal = {ACM Trans. Reconfigurable Technol. Syst.},
    month = nov,
}
```
