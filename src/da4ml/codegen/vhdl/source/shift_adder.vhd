library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity shift_adder is
    generic (
        BW_INPUT0 : integer := 32;
        BW_INPUT1 : integer := 32;
        SIGNED0   : integer := 0;
        SIGNED1   : integer := 0;
        BW_OUT    : integer := 32;
        SHIFT1    : integer := 0;
        IS_SUB    : integer := 0
    );
    port (
        in0 : in  std_logic_vector(BW_INPUT0-1 downto 0);
        in1 : in  std_logic_vector(BW_INPUT1-1 downto 0);
        result : out std_logic_vector(BW_OUT-1 downto 0)
    );
end entity shift_adder;

architecture rtl of shift_adder is
    function max(L, R: integer) return integer is
    begin
        if L > R then
            return L;
        else
            return R;
        end if;
    end function;

    function if_then_else(cond: boolean; val_true: integer; val_false: integer) return integer is
    begin
        if cond then
            return val_true;
        else
            return val_false;
        end if;
    end function;

    constant IN0_NEED_BITS : integer := if_then_else(SHIFT1 < 0, BW_INPUT0 - SHIFT1, BW_INPUT0);
    constant IN1_NEED_BITS : integer := if_then_else(SHIFT1 > 0, BW_INPUT1 + SHIFT1, BW_INPUT1);
    constant EXTRA_PAD     : integer := if_then_else(SIGNED0 /= SIGNED1, IS_SUB + 1, IS_SUB);
    constant BW_ADD        : integer := max(IN0_NEED_BITS, IN1_NEED_BITS) + EXTRA_PAD + 1;

    signal in0_ext : std_logic_vector(BW_ADD-1 downto 0);
    signal in1_ext : std_logic_vector(BW_ADD-1 downto 0);
    signal accum   : std_logic_vector(BW_ADD-1 downto 0);

begin

    -- Extension and shifting for input 0
    gen_in0_shift_neg: if SHIFT1 < 0 generate
        gen_in0_signed: if SIGNED0 = 1 generate
            in0_ext <= std_logic_vector(resize(signed(in0), BW_ADD)) sll (-SHIFT1);
        end generate;
        gen_in0_unsigned: if SIGNED0 = 0 generate
            in0_ext <= std_logic_vector(resize(unsigned(in0), BW_ADD)) sll (-SHIFT1);
        end generate;
    end generate;

    gen_in0_shift_pos: if SHIFT1 >= 0 generate
        gen_in0_signed: if SIGNED0 = 1 generate
            in0_ext <= std_logic_vector(resize(signed(in0), BW_ADD));
        end generate;
        gen_in0_unsigned: if SIGNED0 = 0 generate
            in0_ext <= std_logic_vector(resize(unsigned(in0), BW_ADD));
        end generate;
    end generate;

    -- Extension and shifting for input 1
    gen_in1_shift_pos: if SHIFT1 > 0 generate
        gen_in1_signed: if SIGNED1 = 1 generate
            in1_ext <= std_logic_vector(resize(signed(in1), BW_ADD)) sll SHIFT1;
        end generate;
        gen_in1_unsigned: if SIGNED1 = 0 generate
            in1_ext <= std_logic_vector(resize(unsigned(in1), BW_ADD)) sll SHIFT1;
        end generate;
    end generate;

    gen_in1_shift_neg: if SHIFT1 <= 0 generate
        gen_in1_signed: if SIGNED1 = 1 generate
            in1_ext <= std_logic_vector(resize(signed(in1), BW_ADD));
        end generate;
        gen_in1_unsigned: if SIGNED1 = 0 generate
            in1_ext <= std_logic_vector(resize(unsigned(in1), BW_ADD));
        end generate;
    end generate;

    -- Addition/subtraction logic
    gen_sub: if IS_SUB = 1 generate
        accum <= std_logic_vector(signed(in0_ext) - signed(in1_ext));
    end generate;

    gen_add: if IS_SUB = 0 generate
        accum <= std_logic_vector(signed(in0_ext) + signed(in1_ext));
    end generate;

    result <= accum(BW_OUT-1 downto 0);

end architecture rtl;
