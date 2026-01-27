set project_name "$::env(PROJECT_NAME)"
set device "$::env(DEVICE)"
set source_type "$::env(SOURCE_TYPE)"

set prj_root [file normalize [file dirname [info script]]]
set top_module "${project_name}"
set output_dir "${prj_root}/output_${project_name}"

file mkdir $output_dir
file mkdir "${output_dir}/reports"

cd $output_dir
project_new "${project_name}" -overwrite -revision "${project_name}"

set_global_assignment -name FAMILY [lindex [split "${device}" "-"] 0]
set_global_assignment -name DEVICE "${device}"

if { "${source_type}" != "vhdl" && "${source_type}" != "verilog" } {
    puts "Error: SOURCE_TYPE must be either 'vhdl' or 'verilog'."
    exit 1
}

# Add source files based on type
if { "${source_type}" == "vhdl" } {
    set_global_assignment -name VHDL_INPUT_VERSION VHDL_2008

    foreach file [glob -nocomplain "${prj_root}/src/static/*.vhd"] {
        set_global_assignment -name VHDL_FILE "${file}"
    }

    set_global_assignment -name VHDL_FILE "${prj_root}/src/${project_name}.vhd"
    foreach file [glob -nocomplain "${prj_root}/src/${project_name}_stage*.vhd"] {
        set_global_assignment -name VHDL_FILE "${file}"
    }
} else {
    foreach file [glob -nocomplain "${prj_root}/src/static/*.v"] {
        set_global_assignment -name VERILOG_FILE "${file}"
    }

    set_global_assignment -name VERILOG_FILE "${prj_root}/src/${project_name}.v"
    foreach file [glob -nocomplain "${prj_root}/src/${project_name}_stage*.v"] {
        set_global_assignment -name VERILOG_FILE "${file}"
    }
}

foreach f [glob -nocomplain "${prj_root}/src/memfiles/*.mem"] {
    file copy -force $f "${output_dir}/[file tail $f]"
}
set mems [glob -nocomplain "${output_dir}/*.mem"]

foreach f $mems {
    set_global_assignment -name MIF_FILE "${f}"
}

# Add SDC constraint file if it exists
if { [file exists "${prj_root}/src/${project_name}.sdc"] } {
    file copy -force "${prj_root}/src/${project_name}.sdc" "${output_dir}/${project_name}.sdc"
    set_global_assignment -name SDC_FILE "${output_dir}/${project_name}.sdc"
}

# Set top-level entity
set_global_assignment -name TOP_LEVEL_ENTITY "${top_module}"

# OOC
load_package flow

proc make_all_pins_virtual {} {
    execute_module -tool map

    set name_ids [get_names -filter * -node_type pin]

    foreach_in_collection name_id $name_ids {
        set pin_name [get_name_info -info full_path $name_id]
        post_message "Making VIRTUAL_PIN assignment to $pin_name"
        set_instance_assignment -to $pin_name -name VIRTUAL_PIN ON
    }
    export_assignments
}

make_all_pins_virtual

# Config
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"
set_global_assignment -name OPTIMIZATION_TECHNIQUE SPEED
set_global_assignment -name AUTO_RESOURCE_SHARING ON
set_global_assignment -name ALLOW_ANY_RAM_SIZE_FOR_RECOGNITION ON
set_global_assignment -name ALLOW_ANY_ROM_SIZE_FOR_RECOGNITION ON
set_global_assignment -name ALLOW_REGISTER_RETIMING ON

set_global_assignment -name TIMEQUEST_MULTICORNER_ANALYSIS ON
set_global_assignment -name TIMEQUEST_DO_CCPP_REMOVAL ON

set_global_assignment -name FITTER_EFFORT "STANDARD FIT"

set_global_assignment -name SYNTH_TIMING_DRIVEN_SYNTHESIS ON
set_global_assignment -name SYNTHESIS_EFFORT AUTO
set_global_assignment -name ADV_NETLIST_OPT_SYNTH_WYSIWYG_REMAP ON

set_global_assignment -name PROJECT_OUTPUT_DIRECTORY "${output_dir}"

# Run!!!
execute_flow -compile

foreach report [glob -nocomplain "${output_dir}/*.rpt"] {
    file copy -force $report "${output_dir}/reports/"
}

foreach f $mems {
    file delete $f
}

project_close
