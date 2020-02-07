srcdir := src
objdir := obj
bindir := bin
target := OPS 
target_debug := OPSDEBUG

headers := $(wildcard $(srcdir)/*.h)
src := $(wildcard $(srcdir)/*.cpp)
obj := $(src:$(srcdir)/%.cpp=$(objdir)/%.o)

third := thirdparty
incs := 

LDFLAGS := 
LDLIBS := 

CPPFLAGS :=  
CXXFLAGS := ${incs} -Wall -Wextra -std=c++14 -g -O2


(srcdir)/%.cpp : $(srcdir)/%.h


$(bindir)/$(target): $(obj) | $(bindir)	
	$(CXX) -o $@ $(LDLIBS) $(LDFLAGS) $(obj) 

$(bindir):
	mkdir $(bindir)

$(objdir)/%.o: $(srcdir)/%.cpp | $(objdir)
	$(CXX) -o $@ -MD -MP $(CPPFLAGS) $(CXXFLAGS) -c $<

$(objdir):
	mkdir $(objdir)

.PHONY: clean
clean:
	rm -rf $(bindir) $(objdir) 


