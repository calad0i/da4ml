default: slow

VERILATOR_ROOT = $(shell verilator -V | grep -a VERILATOR_ROOT | tail -1 | awk '{{print $$3}}')
INCLUDES = -I./obj_dir -I$(VERILATOR_ROOT)/include
WARNINGS = -Wl,--no-undefined
CFLAGS = -std=c++17 -fPIC
LINKFLAGS = $(INCLUDES) $(WARNINGS)
LIBNAME = lib$(VM_PREFIX)_$(STAMP).so


./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a: $(VM_PREFIX).v
	verilator --cc -j -Wall -build $(VM_PREFIX).v --prefix V$(VM_PREFIX) -CFLAGS "$(CFLAGS)"

$(LIBNAME): ./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a bridge.cc
	$(CXX) $(CFLAGS) $(LINKFLAGS) $(CXXFLAGS2) -pthread -shared -o $(LIBNAME) bridge.cc ./obj_dir/libV$(VM_PREFIX).a ./obj_dir/libverilated.a ./obj_dir/V$(VM_PREFIX)__ALL.a


fast: CFLAGS += -O3
fast: $(LIBNAME)

slow: CFLAGS += -O
slow: $(LIBNAME)

clean:
	rm -rf obj_dir
	rm -f $(LIBNAME)
