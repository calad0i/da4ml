`timescale 1ns / 1ps


module shift_adder #(
    parameter BW_INPUT0 = 32,
    parameter BW_INPUT1 = 32,
    parameter SIGNED0 = 0,
    parameter SIGNED1 = 0,
    parameter BW_OUT = 32,
    parameter SHIFT1 = 0,
    parameter IS_SUB = 0
) (
    input [BW_INPUT0-1:0] in0,
    input [BW_INPUT1-1:0] in1,
    output [BW_OUT-1:0] out
);

  localparam IN0_NEED_BITS = (SHIFT1 < 0) ? BW_INPUT0 - SHIFT1 : BW_INPUT0;
  localparam IN1_NEED_BITS = (SHIFT1 > 0) ? BW_INPUT1 + SHIFT1 : BW_INPUT1;
  localparam EXTRA_PAD = (SIGNED0 != SIGNED1) ? IS_SUB+1 : IS_SUB+0;
  localparam BW_ADD = (IN0_NEED_BITS > IN1_NEED_BITS) ? IN0_NEED_BITS + EXTRA_PAD + 1 : IN1_NEED_BITS + EXTRA_PAD + 1;
  localparam IN0_PAD_LEFT = (SHIFT1 < 0) ? BW_ADD - BW_INPUT0 + SHIFT1 : BW_ADD - BW_INPUT0;
  localparam IN0_PAD_RIGHT = (SHIFT1 < 0) ? -SHIFT1 : 0;
  localparam IN1_PAD_LEFT = (SHIFT1 > 0) ? BW_ADD - BW_INPUT1 - SHIFT1 : BW_ADD - BW_INPUT1;
  localparam IN1_PAD_RIGHT = (SHIFT1 > 0) ? SHIFT1 : 0;

  wire [BW_ADD-1:0] in0_ext;
  wire [BW_ADD-1:0] in1_ext;

  // verilator lint_off UNUSEDSIGNAL
  wire [BW_ADD-1:0] accum;
  // verilator lint_on UNUSEDSIGNAL

  generate
    if (SIGNED0 == 1) begin : in0_is_signed
      assign in0_ext = {{IN0_PAD_LEFT{in0[BW_INPUT0-1]}}, in0, {IN0_PAD_RIGHT{1'b0}}};
    end else begin : in0_is_unsigned
      assign in0_ext = {{IN0_PAD_LEFT{1'b0}}, in0, {IN0_PAD_RIGHT{1'b0}}};
    end
    if (SIGNED1 == 1) begin : in1_is_signed
      assign in1_ext = {{IN1_PAD_LEFT{in1[BW_INPUT1-1]}}, in1, {IN1_PAD_RIGHT{1'b0}}};
    end else begin : in1_is_unsigned
      assign in1_ext = {{IN1_PAD_LEFT{1'b0}}, in1, {IN1_PAD_RIGHT{1'b0}}};
    end
  endgenerate

  generate
    if (IS_SUB == 1) begin : is_sub
      assign accum = in0_ext - in1_ext;
    end else begin : is_add
      assign accum = in0_ext + in1_ext;
    end
  endgenerate
  assign out = accum[BW_OUT-1:0];

endmodule
