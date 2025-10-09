set clock_period $::env(CLOCK_PERIOD)

# Clock uncertainty as percentage of clock period
set uncertainty_setup_r $::env(UNCERTAINITY_SETUP)
set uncertainty_hold_r $::env(UNCERTAINITY_HOLD)
set delay_max_r $::env(DELAY_MAX)
set delay_min_r $::env(DELAY_MIN)

# Calculate actual uncertainty values
set uncertainty_setup [expr {$clock_period * $uncertainty_setup_r}]
set uncertainty_hold [expr {$clock_period * $uncertainty_hold_r}]
set delay_max [expr {$clock_period * $delay_max_r}]
set delay_min [expr {$clock_period * $delay_min_r}]

# Create clock with variable period
create_clock -period $clock_period -name sys_clk [get_ports {clk}]

# Input/Output constraints
set_input_delay -clock sys_clk -max $delay_max [get_ports {model_inp[*]}]
set_input_delay -clock sys_clk -min $delay_min [get_ports {model_inp[*]}]

set_output_delay -clock sys_clk -max $delay_max [get_ports {model_out[*]}]
set_output_delay -clock sys_clk -min $delay_min [get_ports {model_out[*]}]

# Apply calculated uncertainty values
set_clock_uncertainty -setup -to [get_clocks sys_clk] $uncertainty_setup
set_clock_uncertainty -hold -to [get_clocks sys_clk] $uncertainty_hold
