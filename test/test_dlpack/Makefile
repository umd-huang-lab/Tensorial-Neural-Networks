srcdir := src
objdir := obj
bindir := lib
target := test_dlpack
so_name := $(target)`python-config --extension-suffix`

headers := $(wildcard $(srcdir)/*.h)
src := $(wildcard $(srcdir)/*.cpp)
obj := $(src:$(srcdir)/%.cpp=$(objdir)/%.o)
dep := $(obj:.o=.d)

third := ../../thirdparty

# \todo change to -I$(third)/eigen3/include
incs := -I/usr/local/include/eigen3 \
		-I$(third)/dlpack/include \
		`python -m pybind11 --includes`


LDFLAGS := -fPIC -shared -Wl,-install_name,$(so_name) -Wl,-undefined,dynamic_lookup
LDLIBS := 

CPPFLAGS :=  
CXXFLAGS := $(incs) -Wall -Wextra -std=c++14 -g -O2


(srcdir)/%.cpp : $(srcdir)/%.h


$(bindir)/$(target): $(obj) | $(bindir)	
	$(CXX) -o $(bindir)/$(so_name) $(LDLIBS) $(LDFLAGS) $(obj) 

$(bindir):
	mkdir $(bindir)


-include $(dep)

$(objdir)/%.o: $(srcdir)/%.cpp | $(objdir)
	$(CXX) -o $@ -MMD -MP $(CPPFLAGS) $(CXXFLAGS) -c $<

$(objdir):
	mkdir $(objdir)

.PHONY: clean
clean:
	rm -rf $(bindir) $(objdir) 


