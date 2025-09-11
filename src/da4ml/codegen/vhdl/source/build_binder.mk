default: slow

VERILATOR_ROOT = $(shell verilator -V | grep -a VERILATOR_ROOT | tail -1 | awk '{{print $$3}}')
INCLUDES = -I./obj_dir -I$(VERILATOR_ROOT)/include
WARNINGS = -Wl,--no-undefined
CFLAGS = -std=c++17 -fPIC
LINKFLAGS = $(INCLUDES) $(WARNINGS)
LIBNAME = lib$(VM_PREFIX)_$(STAMP).so
N_JOBS ?= $(shell nproc)

FILES = $(wildcard *.vhd)


$(VM_PREFIX).v: $(VM_PREFIX).vhd
	mkdir -p obj_dir
	ghdl -a --std=08 --workdir=obj_dir multiplier.vhd mux.vhd negative.vhd shift_adder.vhd $(wildcard *_stage*.vhd) $(wildcard test.vhd) $(VM_PREFIX).vhd
	ghdl synth --std=08 --workdir=obj_dir --out=verilog $(VM_PREFIX) > $(VM_PREFIX).v

./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a: $(VM_PREFIX).v
	verilator --cc -j $(N_JOBS) -build $(VM_PREFIX).v --prefix V$(VM_PREFIX) -CFLAGS "$(CFLAGS)"

$(LIBNAME): ./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a $(VM_PREFIX)_binder.cc
	$(CXX) $(CFLAGS) $(LINKFLAGS) $(CXXFLAGS2) -pthread -shared -o $(LIBNAME) $(VM_PREFIX)_binder.cc ./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a $(EXTRA_CXXFLAGS)


fast: CFLAGS += -O3
fast: $(LIBNAME)

slow: CFLAGS += -O
slow: $(LIBNAME)

clean:
	rm -rf obj_dir
	rm -f $(LIBNAME)
