# User-customizable variables:
CXX ?= c++
CXX_STD ?= c++20
CXXFLAGS ?= -DSTDEXEC_RELACY -I relacy -I relacy/relacy/fakestd -O1 -std=$(CXX_STD) -I ../../include -I ../../test -g
DEPFLAGS ?= -MD -MF $(@).d -MP -MT $(@)
build_dir = build

.SECONDARY:

test_programs = split async_scope sync_wait

test_exe_files = $(foreach name,$(test_programs),$(build_dir)/$(name))

exe_files = $(test_exe_files)
o_files = $(exe_files:=.cpp.o)

ansi_term_csi = [
ansi_term_bold = $(ansi_term_csi)1m
ansi_term_green = $(ansi_term_csi)32m
ansi_term_red = $(ansi_term_csi)31m
ansi_term_reset = $(ansi_term_csi)m

COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS) -c
LINK.cpp = $(CXX) $(CXXFLAGS)

.PHONY: all
all: tests

.PHONY: tests
tests: $(test_exe_files)

$(build_dir)/%: $(build_dir)/%.cpp.o
	$(LINK.cpp) $(^) -o $(@)

$(build_dir)/%.cpp.o: %.cpp
	@mkdir -p $(dir $(@))
	$(COMPILE.cpp) -o $(@) $(<)

.PHONY: clean
clean:
	rm -fr -- $(build_dir)/

-include $(o_files:=.d)
