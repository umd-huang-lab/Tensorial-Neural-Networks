srcdir := src
objdir := obj
bindir := bin
target_bin := OPS
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
incs := -I$(third)/eigen/3.3.4/include/eigen3 \
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

# it makes more sense to put $(target_lib) in the binary directory, so all of this project's output products
# are in one place. Putting it in a directory called lib may confuse someone who might think lib contains third party
# libraries, and might accidentally delete a library they install there with a call to make clean
# (note third party libraries should be installed in thirdparty)
.PHONY: library
library: $(bindir)/$(target_lib)

$(bindir)/$(target_bin): $(obj) | $(bindir)
	$(CXX) -o $@ $(LDLIBS) $(target_bin_LDFLAGS) $(obj_without_python)

$(bindir):
	mkdir $(bindir)


$(bindir)/$(target_lib): $(obj) | $(bindir)
	$(CXX) -o $(bindir)/$(so_name) $(LDLIBS) $(target_lib_LDFLAGS) $(obj)

-include $(dep)

$(objdir)/%.o: $(srcdir)/%.cpp | $(objdir)
	$(CXX) -o $@ -MMD -MP $(CPPFLAGS) $(CXXFLAGS) -c $<

$(objdir):
	mkdir $(objdir)

.PHONY: clean
clean:
	rm -rf $(bindir) $(objdir) 

.PHONY: cleanlink
cleanlink:
	rm -rf $(bindir)

