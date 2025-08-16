DA Instruction Set (DAIS)
=============================================

In da4ml, all operations are converted to a RISC-like, instruction set level intermediate representation, distributed arithmetic instruction set (DAIS, pounces as "dice"). DAIS is designed to be minimal and lightweight, while being extensible and satisfying the requirements for representing neural networks required in the framework. Specifically, each DAIS program is in SSA form and contains one block of logic that are fully parallelizable (i.e., all combinational), and resource multiplexing shall be performed on a higher level.

One program represented in DAIS consists of the following components:

## Program Structure

- `shape`: tuple<int, int>
    - The number of inputs and outputs of the program.
- `inp_shift`: vector<int>
    - The shifts required to interpret the input data. (i.e., number of integers in the fixed-point representation)
- `out_idxs`: vector<int>
    - The indices of the output data shall be read from the buffer
- `out_shifts`: vector<int>
    - The shifts required to interpret the output data.
- `out_negs`: vector<bool>
    - The signs of the output data.
- `ops`: vector<Op>
    - The core list of operations for populating the full buffer.

Each operation is represented as a `Op` object, consists of the following components:
- `opcode`: int
    - The operation code, see [OpCode](#opcode).
- `id0`, `id1`: int
    - The first and second operand indices in the buffer. Unused operands must be set to `-1`.
- `data`: int64
    - Extra integer data for the operation, functionality depends on the opcode.
- `dtype`: tuple<float, float, float> OR tuple<int/bool, int, int>
    - Annotates the datatype of the output buffer as a quantization interval.
    - (min, max, step) or (signed, integer_bits (excl sign bit), fractional_bits). If using (min, max, step), format, it is assumed that the minimal fixed-point representation that contains the full range of the quantization interval is used. (e.g., (-3., 3., 1.) is the same as (-4., 3., 1.): both are (1, 2, 0) in fixed point representation). Step **must** be of a power of two.
    - **Must** cause no overflow if the operation itself does not imply quantization.

The program is executed as follows:
1. Instantiate an empty buffer of size `len(ops)`.
2. Go through the list of operations in `ops`. Fill the i-th index of the buffer with the result of the i-th operation: buf[i] = ops[i](buf, inp)
3. Instantiate the output buffer of size `shape[1]`.
4. Fill output buffer:
  - `output_buf[i] = buf[out_idxs[i]] * 2^out_shifts[i] * (-1 if out_negs[i] else 1)`

### OpCode
The operation codes are defined as follows:
- `-1`: Copy from input buffer (**implies quantization**)
  - `buf[i] = input[id0]`
- `0/1`: Addition/Subtraction
  - `buf[i] = buf[id0] +/- buf[id1] * 2^data`
- `2/-2`: ReLU (**implies quantization**)
  - `buf[i] = quantize(relu(+/- buf[id0]))`
- `3/-3`: Quantization (**implies quantization**)
  - `buf[i] = quantize(+/- buf[id0])`
- `4`: Add a constant
  - `buf[i] = buf[id0] + data * qint.step`
- `5`: Define a constant
  - `buf[i] = data * qint.step`
- `6/-6`: Mux by MSB
  - `buf[i] = MSB(buf[int32(data_lower_i32)]) ? buf[id0] : +/- buf[id1] * 2^int32(data_higher_i32)`
- `*`: Multiplication
  - `buf[i] = buf[id0] * buf[id1]`

In all cases, unused id0 or id1 **must** be set to `-1`; id0, id1 (and data for opcode=+/-6) **must** be smaller than the index of the operation itself to ensure causality. All quantization are direct bit-drop in binary format (i.e., WRAP for overflow and TRUNC for rounding).

### Binary Representation
The binary representation of the program is as follows, in order:
- `shape`: int32[2]
- `len(ops)`: int32
- `inp_shift`: int32[shape[0]]
- `out_idxs`: int32[shape[1]]
- `out_shifts`: int32[shape[1]]
- `out_negs`: int32[shape[1]]
- `ops`: Op[len(ops)]
    - `opcode`: int32
    - `id0`: int32
    - `id1`: int32
    - `data_higher`: int32
    - `data_lower`: int32
    - `dtype`: int32[3] (only (signed, integer_bits, fractional_bits) format for binary representation)

In execution, the internal buffer **must** have larger bitwidth than the maximum bitwidth appears in any of the operations. When an operation implies quantization, the program **must** apply the quantization explicitly. When an operation does not imply quantization, the program **may** apply quantization and verify no value change is incurred as a result.

`interperter/DAISInterpreter.cc` and `interperter/DAISInterpreter.hh` contains a reference implementation of a DAIS interpreter in C++, which runs the program in a straightforward manner with `int64_t` for the internal buffer. The program is represented in a `int32_t` array, which can be obtained by `comb_logic.to_binary()` or `comb_logic.save_binary(path)`.
