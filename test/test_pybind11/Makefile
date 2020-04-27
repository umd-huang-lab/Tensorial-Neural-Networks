srcdir := src
objdir := obj
bindir := lib
target := test_python_binding
so_name := $(target)`python-config --extension-suffix`

headers := $(wildcard $(srcdir)/*.h)
src := $(wildcard $(srcdir)/*.cpp)
obj := $(src:$(srcdir)/%.cpp=$(objdir)/%.o)
dep := $(obj:.o=.d)

third := 

incs := `python -m pybind11 --includes`

LDFLAGS := -fPIC -shared -Wl,-install_name,$(so_name) -Wl,-undefined,dynamic_lookup 


LDLIBS :=

CPPFLAGS := 
CXXFLAGS := $(incs) -Wall -Wextra -std=c++14 -O2
			
$(warning before rules)

$(srcdir)/%.cpp : $(srcdir)/%.h


$(bindir)/$(target) : $(obj) | $(bindir)
	$(warning building target)	
	$(CXX) -o $(bindir)/$(so_name) $(LDLIBS) $(LDFLAGS) $(obj)


$(bindir):
	$(warning mkdir bindir)
	mkdir $(bindir)
	

-include $(dep)

$(objdir)/%.o : $(srcdir)/%.cpp | $(objdir) 
	$(warning building object file)
	$(CXX) -o $@ -MMD -MP $(CPPFLAGS) $(CXXFLAGS) -c $< 
	

$(objdir):
	$(warning mkdir obj)
	mkdir $(objdir)



.PHONY: clean
clean:
	rm -rf $(bindir) $(objdir)

