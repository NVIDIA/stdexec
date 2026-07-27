module;

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

export module stdexec;

#define STDEXEC_IN_MODULE_PURVIEW

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"

#include <stdexec/execution.hpp>

#pragma clang diagnostic pop
