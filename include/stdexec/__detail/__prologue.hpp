
/*
 * Copyright (c) 2026 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// NO include guard or `#pragma once` here (this file is included multiple times)

#if defined(STDEXEC_PROLOGUE_INCLUDED)
#  error                                                                                           \
    "<stdexec/__detail/__epilogue.hpp> must be included before <stdexec/__detail/__prologue.hpp> is included again"
#endif
#define STDEXEC_PROLOGUE_INCLUDED() 1

#include <stdexec/__detail/__config.hpp>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(is_constant_evaluated_in_nonconstexpr_context)

// Windows.h macros:

#if defined(interface)
#  pragma push_macro("interface")
#  undef interface
#  define STDEXEC_POP_MACRO_interface
#endif  // defined(interface)

#if defined(min)
#  pragma push_macro("min")
#  undef min
#  define STDEXEC_POP_MACRO_min
#endif  // defined(min)

#if defined(max)
#  pragma push_macro("max")
#  undef max
#  define STDEXEC_POP_MACRO_max
#endif  // defined(max)

// sal.h on Windows

#if defined(__valid)
#  pragma push_macro("__valid")
#  undef __valid
#  define STDEXEC_POP_MACRO___valid
#endif  // defined(__valid)

#if defined(__callback)
#  pragma push_macro("__callback")
#  undef __callback
#  define STDEXEC_POP_MACRO___callback
#endif  // defined(__callback)
