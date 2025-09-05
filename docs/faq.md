# FAQs

## What does da4ml do?
Two things:
1. Converting constant-matrix-vector multiplications (CMVMs) into optimized adder graphs with distributed arithmetic for FPGA implementation.
2. Converting (a part of) neural networks to fully parallel HDL or HLS with the CMVM optimization above.

## Should I use the standalone flow or the hls4ml-integrated flow?
If the network is supported by da4ml standalone, it is **recommended to use the standalone flow**. In most cases, the standalone flow gives better latency and timing, and is orders of magnitude faster in synthesis time. However, in some occasions, the hls4ml-integrated flow could provide better timing when the routing is highly challenging for the standalone flow. If the network is not supported by da4ml standalone (e.g., contains unsupported layers or operations), then the hls4ml-integrated flow is the only option.

## So does da4ml only work with neural networks with II=1?
No. When integrated with hls4ml, da4ml only requires that **each CMVM operation is unrolled (II=1)**. This is different from unrolling the whole model, e.g., convolution layers can still have II>1 by reusing the same CMVM kernel for different input windows.

If you are using da4ml standalone, then the answer is yes: DAIS describes only fully parallel logics.

## What the advantage over hls4ml?
1. The CMVM implementation provided by da4ml is more optimized than hls4ml's default implementation (latency strategy). In most cases, using da4ml can reduce the LUT usage, eliminate DSP usage, while achieving lower latency/better timing. This all comes free with no accuracy loss (bit-exact throughout).
2. If the model targets `II=1` (common for L1 trigger applications), using da4ml's direct HDL generation could reduce the synthesis time (network to HDL) from **hours/days to seconds/minutes**.
3. With the low-level symbolic tracing, da4ml supports more flexible operations (e.g., arbitrary array slicing/rearranging, etc.), all by just "replaying" the operations without writing custom templates in C++.

## When should I use da4ml standalone vs. hls4ml+da4ml?
In general, use standalone if possible, as it is simpler and faster. In the following cases, you may want to use hls4ml+da4ml:
1. Some resource sharing is wanted
2. General-non-linear operations (i.e., activations that are not piecewise linear) are needed

## What is latency_cutoff in da4ml?
This is a threshold to decide when a new pipeline stage is needed. When the estimated latency of the network exceeds this threshold, a new pipeline stage is added.

The estimated latency is mostly based on the layer of adders/substractors, each contributing to 1 of abstract latency units. One can also set `carry_size` in `HWConfig` to positive value such that the latency of each adder/subtractor will be estimated by the number of carriers used.

It is always advised to use retiming in the synthesis tool for better timing performance.

## How to use the generated code?
- For Verilog generation, the generated code is a self-contained module that can be instantiated in any Verilog project. We provide a simple `build_prj.tcl` script for Vivado to run OOC synthesis and implementation. No automatic standalone testbench is provided, and we recommend performing functional verification with the Verilator bindings in Python.
- For HLS generation, the generated code is a self-contained C++ function that can be synthesized by the corresponding HLS tool. It is expected to be integrated into a larger HLS project. No build script is provided at the moment.
