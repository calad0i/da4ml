default: slow
INCLUDES = -I ap_types -I .
CXXFLAGS = -fPIC
CFLAGS = -std=c++17 -fPIC
LIBNAME = lib$(PRJ_NAME)_$(STAMP).so

fast: CXXFLAGS += -O3
fast: $(LIBNAME)

slow: CXXFLAGS += -O
slow: $(LIBNAME)

$(PRJ_NAME)_$(STAMP).o: $(PRJ_NAME).cc
	$(CC) -c $(PRJ_NAME).cc -o $(PRJ_NAME)_$(STAMP).o $(INCLUDES) $(CXXFLAGS) $(EXTRA_CXXFLAGS)

$(LIBNAME): $(PRJ_NAME)_$(STAMP).o $(PRJ_NAME)_bridge.cc
	$(CXX) $(INCLUDES) $(CXXFLAGS) -shared -o $@ $(PRJ_NAME)_$(STAMP).o $(PRJ_NAME)_bridge.cc $(EXTRA_CXXFLAGS)

clean:
	rm -f $(LIBNAME) $(PRJ_NAME)_$(STAMP).o

.PHONY: clean
