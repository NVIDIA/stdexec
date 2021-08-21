BUILDDIR = build
IDIR =../include
CXX=clang++-12
CXXFLAGS=-I$(IDIR) -std=c++20

LIBS=-pthread

all: hello_world

execution: $(IDIR)/execution.hpp

setup:
	mkdir -p $(BUILDDIR)/

$(BUILDDIR)/%.o: %.cpp setup execution
	$(CXX) -c -o $@ $< $(CXXFLAGS)

hello_world: $(BUILDDIR)/hello_world.o
	$(CXX) -o $(BUILDDIR)/$@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm $(BUILDDIR)/*
	rmdir $(BUILDDIR)
