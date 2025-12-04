`timescale 1ns / 1ps


module negative #(
    parameter BW_IN = 32,
    parameter BW_OUT = 32,
    parameter IN_SIGNED = 0
) (
    // verilator lint_off UNUSEDSIGNAL
    input  [ BW_IN-1:0] in,
    // verilator lint_off UNUSEDSIGNAL
    output [BW_OUT-1:0] out
);
  /* verilator lint_off WIDTHTRUNC */
  generate
    if (BW_IN < BW_OUT) begin : in_is_smaller
      wire [BW_OUT-1:0] in_ext;
      if (IN_SIGNED == 1) begin : is_signed
        assign in_ext = {{BW_OUT - BW_IN{in[BW_IN-1]}}, in};
      end else begin : is_unsigned
        assign in_ext = {{BW_OUT - BW_IN{1'b0}}, in};
      end
      assign out = -in_ext;
    end else begin : in_is_bigger
      wire [BW_IN-1:0] out_ext;
      assign out_ext = -in;
      assign out = out_ext[BW_OUT-1:0];
    end
  endgenerate
  /* verilator lint_on WIDTHTRUNC */
endmodule
