`timescale 1ns / 1ps


module lookup_table #(
    parameter BW_IN = 8,
    parameter BW_OUT = 8,
    parameter MEM_FILE = "whatever.mem"
) (
    input [BW_IN-1:0] in,
    output [BW_OUT-1:0] out
);

  (* rom_style = (BW_IN <= 999) ? "distributed" : "block" *)
  reg [BW_OUT-1:0] lut_rom [0:(1<<BW_IN)-1];
  reg [BW_OUT-1:0] readout;

  initial begin
    $readmemh(MEM_FILE, lut_rom);
  end

  assign out[BW_OUT-1:0] = readout[BW_OUT-1:0];

  always @(*) begin
    readout = lut_rom[in];
  end

endmodule
