srcdir := src
objdir := obj
bindir := bin
target_bin := OPS
libdir := lib
target_lib := tnnlib
so_name := $(target_lib)`python-config --extension-suffix`

headers := $(wildcard $(srcdir)/*.h)
src := $(wildcard $(srcdir)/*.cpp)
obj := $(src:$(srcdir)/%.cpp=$(objdir)/%.o)
#obj_without_python is a bit of a hack, should possibly reorganize 
obj_without_python := $(filter-out $(objdir)/TensorDLPackInterface.o, $(obj))
dep := $(obj:.o=.d)

third := thirdparty

# \todo only the library target depends on the pybind11 include, so do
# 		I have to include it to build the binary target?
# \todo change to -I$(third)/eigen3/include
incs := -I/usr/local/include/eigen3 \
		-I$(third)/dlpack/include \
		`python -m pybind11 --includes`


# not sure about the cleanest way to mix two main targets in a single makefile
target_bin_LDFLAGS :=
target_lib_LDFLAGS := -fPIC -shared -Wl,-install_name,$(so_name) -Wl,-undefined,dynamic_lookup
LDLIBS :=

CPPFLAGS :=
CXXFLAGS := $(incs) -Wall -Wextra -std=c++14 -g -O2


(srcdir)/%.cpp : $(srcdir)/%.h

# \todo perhaps all shouldn't be a phony target
# \todo make library should be a no-op afer the first time it is called but it isn't
.PHONY: all
all: binary library

.PHONY: binary
binary: $(bindir)/$(target_bin)

.PHONY: library
library: $(libdir)/$(target_lib)

$(bindir)/$(target_bin): $(obj) | $(bindir)
	$(CXX) -o $@ $(LDLIBS) $(target_bin_LDFLAGS) $(obj_without_python)

$(bindir):
	mkdir $(bindir)


$(libdir)/$(target_lib): $(obj) | $(libdir)
	$(CXX) -o $(libdir)/$(so_name) $(LDLIBS) $(target_lib_LDFLAGS) $(obj)

$(libdir):
	mkdir $(libdir)


-include $(dep)

$(objdir)/%.o: $(srcdir)/%.cpp | $(objdir)
	$(CXX) -o $@ -MMD -MP $(CPPFLAGS) $(CXXFLAGS) -c $<

$(objdir):
	mkdir $(objdir)

.PHONY: clean
clean:
	rm -rf $(bindir) $(libdir) $(objdir) 

.PHONY: cleanlink
cleanlink:
	rm -rf $(bindir) $(libdir)

