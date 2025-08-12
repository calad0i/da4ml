`timescale 1ns / 1ps


module multiplier #(
    parameter BW_INPUT0 = 32,
    parameter BW_INPUT1 = 32,
    parameter SIGNED0 = 0,
    parameter SIGNED1 = 0,
    parameter BW_OUT = 32
) (
    input [BW_INPUT0-1:0] in0,
    input [BW_INPUT1-1:0] in1,
    output [BW_OUT-1:0] out
);

  localparam BW_BUF = BW_INPUT0 + BW_INPUT1;

  // verilator lint_off UNUSEDSIGNAL
  wire [BW_BUF - 1:0] buffer;
  // verilator lint_on UNUSEDSIGNAL

  generate
    if (SIGNED0 == 1 && SIGNED1 == 1) begin : signed_signed
      assign buffer[BW_BUF-1:0] = $signed(in0) * $signed(in1);
    end else if (SIGNED0 == 1 && SIGNED1 == 0) begin : signed_unsigned
      assign buffer[BW_BUF-1:0] = $signed(in0) * $signed({{1'b0,in1}});
      // assign buffer[BW_BUF-1] = in0[BW_INPUT0-1];
    end else if (SIGNED0 == 0 && SIGNED1 == 1) begin : unsigned_signed
      assign buffer[BW_BUF-1:0] = $signed({{1'b0,in0}}) * $signed(in1);
      // assign buffer[BW_BUF-1] = in1[BW_INPUT1-1];
    end else begin : unsigned_unsigned
      assign buffer[BW_BUF-1:0] = in0 * in1;
    end
  endgenerate

  assign out[BW_OUT-1:0] = buffer[BW_OUT-1:0];
endmodule
