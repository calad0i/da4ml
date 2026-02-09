`timescale 1ns / 1ps


module binop #(
    parameter BW_INPUT0 = 32,
    parameter BW_INPUT1 = 32,
    parameter SIGNED0 = 0,
    parameter SIGNED1 = 0,
    parameter BW_OUT = 32,
    parameter SHIFT1 = 0,
    parameter OP = 0
) (
    input [BW_INPUT0-1:0] in0,
    input [BW_INPUT1-1:0] in1,
    output [BW_OUT-1:0] out
);

  localparam IN0_NEED_BITS = (SHIFT1 < 0) ? BW_INPUT0 - SHIFT1 : BW_INPUT0;
  localparam IN1_NEED_BITS = (SHIFT1 > 0) ? BW_INPUT1 + SHIFT1 : BW_INPUT1;
  localparam EXTRA_PAD = (SIGNED0 != SIGNED1) ? 1 : 0;
  localparam BW_OUT_TMP = (IN0_NEED_BITS > IN1_NEED_BITS) ? IN0_NEED_BITS + EXTRA_PAD: IN1_NEED_BITS + EXTRA_PAD;
  localparam IN0_PAD_LEFT = (SHIFT1 < 0) ? BW_OUT_TMP - BW_INPUT0 + SHIFT1 : BW_OUT_TMP - BW_INPUT0;
  localparam IN0_PAD_RIGHT = (SHIFT1 < 0) ? -SHIFT1 : 0;
  localparam IN1_PAD_LEFT = (SHIFT1 > 0) ? BW_OUT_TMP - BW_INPUT1 - SHIFT1 : BW_OUT_TMP - BW_INPUT1;
  localparam IN1_PAD_RIGHT = (SHIFT1 > 0) ? SHIFT1 : 0;

  wire [BW_OUT_TMP-1:0] in0_ext;
  wire [BW_OUT_TMP-1:0] in1_ext;

  // verilator lint_off UNUSEDSIGNAL
  wire [BW_OUT_TMP-1:0] out_buf;
  // verilator lint_on UNUSEDSIGNAL

  generate
    if (SIGNED0 == 1) begin : in0_is_signed
      assign in0_ext = {{IN0_PAD_LEFT{in0[BW_INPUT0-1]}}, in0, {IN0_PAD_RIGHT{1'b0}}};
    end else begin : in0_is_unsigned
      assign in0_ext = {{IN0_PAD_LEFT{1'b0}}, in0, {IN0_PAD_RIGHT{1'b0}}};
    end
  endgenerate

  generate
    if (SIGNED1 == 1) begin : in1_is_signed
      assign in1_ext = {{IN1_PAD_LEFT{in1[BW_INPUT1-1]}}, in1, {IN1_PAD_RIGHT{1'b0}}};
    end else begin : in1_is_unsigned
      assign in1_ext = {{IN1_PAD_LEFT{1'b0}}, in1, {IN1_PAD_RIGHT{1'b0}}};
    end
  endgenerate

  generate // AND, OR, XOR
    if (OP == 0) begin : is_and
      assign out_buf = in0_ext & in1_ext;
    end else if (OP == 1) begin : is_or
      assign out_buf = in0_ext | in1_ext;
    end else if (OP == 2) begin : is_xor
      assign out_buf = in0_ext ^ in1_ext;
    end
  endgenerate
  assign out = out_buf[BW_OUT-1:0];

endmodule
