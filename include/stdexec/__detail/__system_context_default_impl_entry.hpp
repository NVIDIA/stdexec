/*
 * Copyright (c) 2024 Lucian Radu Teodorescu
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

// This file assumes STDEXEC_SYSTEM_CONTEXT_INLINE is defined before including it. But clang-tidy
// and doxygen don't know that, so we need to include the header that defines it when clang-tidy and
// doxygen are invoked.
#if defined(STDEXEC_CLANG_TIDY_INVOKED) || defined(STDEXEC_DOXYGEN_INVOKED)
#  include "../../exec/system_context.hpp" // IWYU pragma: keep
#endif

#if !defined(STDEXEC_SYSTEM_CONTEXT_INLINE)
#  error "STDEXEC_SYSTEM_CONTEXT_INLINE must be defined before including this header"
#endif

#include "__system_context_default_impl.hpp" // IWYU pragma: keep

#define __STDEXEC_SYSTEM_CONTEXT_API extern STDEXEC_SYSTEM_CONTEXT_INLINE STDEXEC_ATTRIBUTE(weak)

namespace STDEXEC::system_context_replaceability {

  /// Get the backend for the parallel scheduler.
  /// Users might replace this function.
  auto query_parallel_scheduler_backend() -> std::shared_ptr<parallel_scheduler_backend> {
    return __system_context_default_impl::__parallel_scheduler_backend_singleton
      .__get_current_instance();
  }

  /// Set a factory for the parallel scheduler backend.
  /// Can be used to replace the parallel scheduler at runtime.
  /// Out of spec.
  auto set_parallel_scheduler_backend(__parallel_scheduler_backend_factory __new_factory)
    -> __parallel_scheduler_backend_factory {
    return __system_context_default_impl::__parallel_scheduler_backend_singleton
      .__set_backend_factory(__new_factory);
  }

} // namespace STDEXEC::system_context_replaceability
