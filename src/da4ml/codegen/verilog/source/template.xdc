set clock_period ${CLOCK_PERIOD}

# Clock uncertainty as percentage of clock period
set uncertainty_setup_r ${UNCERTAINITY_SETUP}
set uncertainty_hold_r ${UNCERTAINITY_HOLD}
set delay_max_r ${DELAY_MAX}
set delay_min_r ${DELAY_MIN}

# Calculate actual uncertainty values
set uncertainty_setup [expr {$clock_period * $uncertainty_setup_r}]
set uncertainty_hold [expr {$clock_period * $uncertainty_hold_r}]
set delay_max [expr {$clock_period * $delay_max_r}]
set delay_min [expr {$clock_period * $delay_min_r}]

# Create clock with variable period
create_clock -period $clock_period -name sys_clk [get_ports {clk}]

# Input/Output constraints
set_input_delay -clock sys_clk -max $delay_max [get_ports {inp[*]}]
set_input_delay -clock sys_clk -min $delay_min [get_ports {inp[*]}]

set_output_delay -clock sys_clk -max $delay_max [get_ports {out[*]}]
set_output_delay -clock sys_clk -min $delay_min [get_ports {out[*]}]

# Apply calculated uncertainty values
set_clock_uncertainty -setup $uncertainty_setup [get_clocks sys_clk]
set_clock_uncertainty -hold $uncertainty_hold [get_clocks sys_clk]

set_property HD.CLK_SRC BUFG_X0Y0 [get_ports clk]
