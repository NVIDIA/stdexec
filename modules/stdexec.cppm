module;

#include <cassert>

export module stdexec;

import std;

#define STDEXEC_IN_MODULE_PURVIEW
#define STDEXEC_MODULE_EXPORT export

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"

#include <stdexec/execution.hpp>

#pragma clang diagnostic pop
