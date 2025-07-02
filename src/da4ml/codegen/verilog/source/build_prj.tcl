set project_name "${PROJECT_NAME}"
set device "${DEVICE}"

set top_module "${project_name}_wrapper"
set output_dir "./output_${project_name}"

create_project $project_name "${output_dir}/$project_name" -force -part $device

set_property TARGET_LANGUAGE Verilog [current_project]
set_property DEFAULT_LIB work [current_project]

read_verilog "${project_name}_wrapper.v"
read_verilog "${project_name}.v"
read_verilog "shift_adder.v"
foreach file [glob -nocomplain "${project_name}_stage*.v"] {
    read_verilog $file
}

read_xdc "${project_name}.xdc" -mode out_of_context

set_property top $top_module [current_fileset]

file mkdir $output_dir
file mkdir "${output_dir}/reports"

# synth
synth_design -top $top_module -mode out_of_context -retiming \
    -flatten_hierarchy rebuilt -resource_sharing auto \
    -directive AlternateRoutability

write_checkpoint -force "${output_dir}/${project_name}_post_synth.dcp"

report_timing_summary -file "${output_dir}/reports/${project_name}_post_synth_timing.rpt"
report_power -file "${output_dir}/reports/${project_name}_post_synth_power.rpt"
report_utilization -file "${output_dir}/reports/${project_name}_post_synth_util.rpt"

# set_property CARRY_REMAP 3 [get_cells -hier -filter {ref_name == CARRY8}]

opt_design -directive ExploreSequentialArea
opt_design -directive ExploreWithRemap

report_design_analysis -congestion -file "${output_dir}/reports/${project_name}_post_opt_congestion.rpt"

# place
place_design -directive AltSpreadLogic_high -fanout_opt
report_design_analysis -congestion -file "${output_dir}/reports/${project_name}_post_place_congestion_initial.rpt"

phys_opt_design -directive AggressiveExplore
write_checkpoint -force "${output_dir}/${project_name}_post_place.dcp"

report_design_analysis -congestion -file "${output_dir}/reports/${project_name}_post_place_congestion_final.rpt"

report_timing_summary -file "${output_dir}/reports/${project_name}_post_place_timing.rpt"
report_utilization -hierarchical -file "${output_dir}/reports/${project_name}_post_place_util.rpt"

# route
route_design -directive NoTimingRelaxation
write_checkpoint -force "${output_dir}/${project_name}_post_route.dcp"


report_timing_summary -file "${output_dir}/reports/${project_name}_post_route_timing.rpt"
report_timing -sort_by group -max_paths 100 -path_type summary -file "${output_dir}/reports/${project_name}_post_route_timing_paths.rpt"
report_clock_utilization -file "${output_dir}/reports/${project_name}_post_route_clock_util.rpt"
report_utilization -file "${output_dir}/reports/${project_name}_post_route_util.rpt"
report_power -file "${output_dir}/reports/${project_name}_post_route_power.rpt"
report_drc -file "${output_dir}/reports/${project_name}_post_route_drc.rpt"

report_utilization -format xml -hierarchical -file "${output_dir}/reports/${project_name}_post_route_util.xml"
report_power -xpe "${output_dir}/reports/${project_name}_post_route_power.xml"

# Generate Verilog netlist for simulation
# write_verilog -force "${output_dir}/${project_name}_impl_netlist.v" -mode timesim -sdf_anno true

puts "Implementation complete. Results saved in ${output_dir}"
