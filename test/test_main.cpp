#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include "stdexec/__detail/__config.hpp"

#if STDEXEC_USE_MODULES()
import std;
#endif
