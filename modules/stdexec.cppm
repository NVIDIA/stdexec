module;

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if __has_include(<dispatch/dispatch.h>)
#  include <dispatch/dispatch.h>
#endif

#define STDEXEC_IN_MODULE_PURVIEW

#include <stdexec/__detail/__config.hpp>

#if STDEXEC_ENABLE_NUMA
#  include <numa.h>
#endif

export module stdexec;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"

#include <stdexec/execution.hpp>

#pragma clang diagnostic pop
