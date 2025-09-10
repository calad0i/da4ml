library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity negative is
    generic (
        BW_IN     : integer := 32;
        BW_OUT    : integer := 32;
        IN_SIGNED : integer := 0
    );
    port (
        neg_in  : in  std_logic_vector(BW_IN-1 downto 0);
        neg_out : out std_logic_vector(BW_OUT-1 downto 0)
    );
end entity negative;

architecture rtl of negative is
    signal in_ext : std_logic_vector(BW_OUT-1 downto 0);
begin

    gen_lt : if BW_IN < BW_OUT generate
        gen_signed : if IN_SIGNED = 1 generate
            in_ext <= std_logic_vector(resize(signed(neg_in), BW_OUT));
        end generate;
        gen_unsigned : if IN_SIGNED = 0 generate
            in_ext <= std_logic_vector(resize(unsigned(neg_in), BW_OUT));
        end generate;
        neg_out <= std_logic_vector(-signed(in_ext));
    end generate;

    gen_ge : if BW_IN >= BW_OUT generate
        neg_out <= std_logic_vector(-signed(neg_in(BW_OUT-1 downto 0)));
    end generate;

end architecture rtl;
