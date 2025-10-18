library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity lookup_table is
	generic (
		BW_IN    : positive := 8;
		BW_OUT   : positive := 8;
		MEM_FILE : string   := "whatever.mem"
	);
	port (
		inp  : in  std_logic_vector(BW_IN - 1 downto 0);
		outp : out std_logic_vector(BW_OUT - 1 downto 0)
	);
end entity lookup_table;

architecture rtl of lookup_table is
	subtype rom_index_t is natural range 0 to (2 ** BW_IN) - 1;
	type rom_array_t is array (rom_index_t) of std_logic_vector(BW_OUT - 1 downto 0);

	-- Load the ROM contents from an external hex file.
	impure function init_rom return rom_array_t is
		file rom_file : text;
		variable rom_data : rom_array_t := (others => (others => '0'));
		variable line_in  : line;
		variable idx      : integer := 0;
		variable data_val : std_logic_vector(BW_OUT - 1 downto 0);
		variable temp_val : std_logic_vector(((BW_OUT + 3) / 4) * 4 - 1 downto 0);
	begin
		file_open(rom_file, MEM_FILE, read_mode);

	while not endfile(rom_file) loop
		exit when idx > rom_index_t'high;
		readline(rom_file, line_in);
		hread(line_in, temp_val);
		rom_data(idx) := temp_val(BW_OUT - 1 downto 0);
		idx := idx + 1;
	end loop;

	file_close(rom_file);
	return rom_data;
	end function init_rom;

	signal ROM_CONTENTS : rom_array_t := init_rom;

	attribute rom_style : string;
	attribute rom_style of ROM_CONTENTS : signal is "distributed";
begin
	outp <= ROM_CONTENTS(to_integer(unsigned(inp)));
end architecture rtl;
