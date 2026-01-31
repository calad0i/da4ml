set project_name "$::env(PROJECT_NAME)"
set device "$::env(DEVICE)"
set clock_period "$::env(CLOCK_PERIOD)"
set clock_uncertainty "$::env(CLOCK_UNCERTAINTY)"

set prj_root [file normalize [file dirname [info script]]]

open_project "${prj_root}/output_${project_name}"

add_files "${prj_root}/utils/${project_name}_ooc.cc" -cflags "-std=c++0x -I${prj_root}/src"

set_top ${project_name}_fn

open_solution "test_ooc"

config_compile -name_max_length 80
set_part $device
config_schedule -enable_dsp_full_reg=false
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default

puts "***** C/RTL SYNTHESIS *****"

csynth_design

puts "***** IMPLEMENTATION *****"

exec vivado -mode batch -source build_vivado_hls_prj.tcl >@ stdout
