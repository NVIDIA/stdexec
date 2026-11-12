
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

// NO include guard or `#pragma once` here (this file is included multiple
// times)

#if !defined(STDEXEC_PROLOGUE_INCLUDED)
#  error                                                                                           \
    "<stdexec/__detail/__prologue.hpp> must be included before <stdexec/__detail/__epilogue.hpp>"
#endif
#undef STDEXEC_PROLOGUE_INCLUDED

#if defined(STDEXEC_POP_MACRO___callback)
#  pragma pop_macro("__callback")
#  undef STDEXEC_POP_MACRO___callback
#endif  // defined(STDEXEC_POP_MACRO___callback)

#if defined(STDEXEC_POP_MACRO___valid)
#  pragma pop_macro("__valid")
#  undef STDEXEC_POP_MACRO___valid
#endif  // defined(STDEXEC_POP_MACRO___valid)

#if defined(STDEXEC_POP_MACRO_max)
#  pragma pop_macro("max")
#  undef STDEXEC_POP_MACRO_max
#endif  // defined(STDEXEC_POP_MACRO_max)

#if defined(STDEXEC_POP_MACRO_min)
#  pragma pop_macro("min")
#  undef STDEXEC_POP_MACRO_min
#endif  // defined(STDEXEC_POP_MACRO_min)

#if defined(STDEXEC_POP_MACRO_interface)
#  pragma pop_macro("interface")
#  undef STDEXEC_POP_MACRO_interface
#endif  // defined(STDEXEC_POP_MACRO_interface)

STDEXEC_PRAGMA_POP()
