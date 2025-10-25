default: slow

VERILATOR_ROOT = $(shell verilator -V | grep -a VERILATOR_ROOT | tail -1 | awk '{{print $$3}}')
INCLUDES = -I./obj_dir -I$(VERILATOR_ROOT)/include -I../src
WARNINGS = -Wl,--no-undefined
CFLAGS = -std=c++17 -fPIC
LINKFLAGS = $(INCLUDES) $(WARNINGS)
LIBNAME = lib$(VM_PREFIX)_$(STAMP).so
N_JOBS ?= $(shell nproc)
VERILATOR_FLAGS ?=

../src/$(VM_PREFIX).v: $(wildcard ../src/$(VM_PREFIX).vhd) $(wildcard ../src/$(VM_PREFIX)_stage*.vhd)
# vhdl specific - convert to verilog first for verilating
	mkdir -p obj_dir
	cp ../src/memfiles/* ./ 2>/dev/null || true
	ghdl -a --std=08 --workdir=obj_dir ../src/static/multiplier.vhd ../src/static/mux.vhd ../src/static/negative.vhd ../src/static/shift_adder.vhd ../src/static/lookup_table.vhd $(wildcard ../src/$(VM_PREFIX:_wrapper=)_stage*.vhd) $(wildcard ../src/$(VM_PREFIX:_wrapper=).vhd) ../src/$(VM_PREFIX).vhd
	ghdl synth --std=08 --workdir=obj_dir --out=verilog $(VM_PREFIX) > $(VM_PREFIX).v

./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a: ../src/$(VM_PREFIX).v $(wildcard ../src/$(VM_PREFIX)_stage*.v)
	verilator --cc -j $(N_JOBS) -build $(VM_PREFIX).v --prefix V$(VM_PREFIX) $(VERILATOR_FLAGS) -CFLAGS "$(CFLAGS)" -I../src -I../src/static

$(LIBNAME): ./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a $(VM_PREFIX)_binder.cc
	$(CXX) $(CFLAGS) $(LINKFLAGS) $(CXXFLAGS2) -pthread -shared -o $(LIBNAME) $(VM_PREFIX)_binder.cc ./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a $(EXTRA_CXXFLAGS)


fast: CFLAGS += -O3
fast: $(LIBNAME)

slow: CFLAGS += -O
slow: $(LIBNAME)

clean:
	rm -rf obj_dir
	rm -f $(LIBNAME)
