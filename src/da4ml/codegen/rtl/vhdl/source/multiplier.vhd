library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multiplier is
    generic (
        BW_INPUT0 : integer := 32;
        BW_INPUT1 : integer := 32;
        SIGNED0   : integer := 0;
        SIGNED1   : integer := 0;
        BW_OUT    : integer := 32
    );
    port (
        in0 : in  std_logic_vector(BW_INPUT0-1 downto 0);
        in1 : in  std_logic_vector(BW_INPUT1-1 downto 0);
        result : out std_logic_vector(BW_OUT-1 downto 0)
    );
end entity multiplier;

architecture rtl of multiplier is
    constant BW_BUF : integer := BW_INPUT0 + BW_INPUT1;
    signal mult_buffer : std_logic_vector(BW_BUF-1 downto 0);
begin

    gen_mult : process(in0, in1)
    begin
        if SIGNED0 = 1 and SIGNED1 = 1 then
            mult_buffer <= std_logic_vector(resize(signed(in0) * signed(in1), BW_BUF));
        elsif SIGNED0 = 1 and SIGNED1 = 0 then
            mult_buffer <= std_logic_vector(resize(signed(in0) * signed('0' & in1), BW_BUF));
        elsif SIGNED0 = 0 and SIGNED1 = 1 then
            mult_buffer <= std_logic_vector(resize(signed('0' & in0) * signed(in1), BW_BUF));
        else
            mult_buffer <= std_logic_vector(resize(unsigned(in0) * unsigned(in1), BW_BUF));
        end if;
    end process;

    result <= mult_buffer(BW_OUT-1 downto 0);

end architecture rtl;
