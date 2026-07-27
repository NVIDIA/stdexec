#define CATCH_CONFIG_MAIN

#include "stdexec/__detail/__config.hpp"

#if STDEXEC_USE_MODULES()
#  include <catch2/catch_all.hpp>
import std;
#else
#  include <test_common/catch2.hpp>
#endif
