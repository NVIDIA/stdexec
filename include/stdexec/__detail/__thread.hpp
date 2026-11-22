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
#pragma once

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__spin_loop_pause.hpp"

#  if !STDEXEC_USE_MODULES() && !STDEXEC_NO_STDCPP_THREADS()
#    include <thread>
#  endif

#  include "__prologue.hpp"

namespace STDEXEC
{
#  if !STDEXEC_NO_STDCPP_THREADS()

  STDEXEC_MODULE_EXPORT_AUTHORING
  using __thread_id = std::thread::id;

  STDEXEC_MODULE_EXPORT_AUTHORING
  inline auto __get_thread_id() noexcept -> __thread_id
  {
    return std::this_thread::get_id();
  }

  STDEXEC_MODULE_EXPORT_AUTHORING
  inline void __yield() noexcept
  {
    std::this_thread::yield();
  }

#  else  // ^^^ !STDEXEC_NO_STDCPP_THREADS() / vvv STDEXEC_NO_STDCPP_THREADS()

  // Single-threaded fallback when the C++ standard library has no <thread>
  // support (e.g., libc++ with _LIBCPP_HAS_THREADS == 0). Default-constructed
  // __thread_id{} represents "no thread" (0), distinct from __get_thread_id()
  // (1), preserving std::thread::id invariants.
  STDEXEC_MODULE_EXPORT_AUTHORING
  struct __thread_id
  {
    unsigned int __id_ = 0;

    friend constexpr bool operator==(__thread_id, __thread_id) noexcept = default;
  };

  STDEXEC_MODULE_EXPORT_AUTHORING
  STDEXEC_ATTRIBUTE(host, device)
  inline constexpr auto __get_thread_id() noexcept -> __thread_id
  {
    return __thread_id{1};
  }

  STDEXEC_MODULE_EXPORT_AUTHORING
  STDEXEC_ATTRIBUTE(host, device)
  inline void __yield() noexcept
  {
    STDEXEC::__spin_loop_pause();
  }

#  endif  // STDEXEC_NO_STDCPP_THREADS()
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
