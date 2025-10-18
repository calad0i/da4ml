`timescale 1ns / 1ps


module lookup_table #(
    parameter BW_IN = 8,
    parameter BW_OUT = 8,
    parameter mem_file = "whatever.mem"
) (
    input [BW_IN-1:0] in,
    output [BW_OUT-1:0] out
);

  (*rom_style = "distributed" *)
  reg [BW_OUT-1:0] lut_rom [0:(1<<BW_IN)-1];
  reg [BW_OUT-1:0] readout;

  initial begin
    $readmemh(mem_file, lut_rom);
  end

  assign out[BW_OUT-1:0] = readout[BW_OUT-1:0];

  always @(*) begin
    readout = lut_rom[in];
  end

endmodule
