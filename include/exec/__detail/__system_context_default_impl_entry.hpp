/*
 * Copyright (c) 2024 Lucian Radu Teodorescu
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
#pragma once

// Assumes STDEXEC_SYSTEM_CONTEXT_INLINE is defined

#if !defined(STDEXEC_SYSTEM_CONTEXT_INLINE)
#  error "STDEXEC_SYSTEM_CONTEXT_INLINE must be defined before including this header"
#endif

#include "__system_context_default_impl.hpp" // IWYU pragma: keep

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wattributes") // warning: inline function '[...]' declared weak

/// Gets the default system context implementation.
extern STDEXEC_SYSTEM_CONTEXT_INLINE STDEXEC_ATTRIBUTE((weak)) void*
  __query_system_context_interface(const __uuid& __id) noexcept {
  return exec::__system_context_default_impl::__default_query_system_context_interface(__id);
}

STDEXEC_PRAGMA_POP()
