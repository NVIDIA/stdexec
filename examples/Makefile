BUILDDIR = build
IDIR =../include
CXX=clang++-12
CXXFLAGS=-I$(IDIR) -std=c++20 -stdlib=libc++ -ftemplate-backtrace-limit=0

LIBS=-pthread

all: hello_world hello_coro retry

execution: $(IDIR)/execution.hpp

setup:
	mkdir -p $(BUILDDIR)/

$(BUILDDIR)/%.o: %.cpp setup execution
	$(CXX) -c -o $@ $< $(CXXFLAGS)

hello_world: $(BUILDDIR)/hello_world.o
	$(CXX) -o $(BUILDDIR)/$@ $^ $(CXXFLAGS) $(LIBS)

hello_coro: $(BUILDDIR)/hello_coro.o
	$(CXX) -o $(BUILDDIR)/$@ $^ $(CXXFLAGS) $(LIBS)

retry: $(BUILDDIR)/retry.o
	$(CXX) -o $(BUILDDIR)/$@ $^ $(CXXFLAGS) $(LIBS)

let_value: $(BUILDDIR)/let_value.o
	$(CXX) -o $(BUILDDIR)/$@ $^ $(CXXFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm $(BUILDDIR)/*
	rmdir $(BUILDDIR)
