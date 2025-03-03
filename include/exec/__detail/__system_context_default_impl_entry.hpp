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

#define __STDEXEC_SYSTEM_CONTEXT_API extern STDEXEC_SYSTEM_CONTEXT_INLINE STDEXEC_ATTRIBUTE((weak))

namespace exec::system_context_replaceability {
  /// The default implementation of the `query_system_context` function template.
  template <__queryable_interface _Interface>
  __STDEXEC_SYSTEM_CONTEXT_API std::shared_ptr<_Interface> query_system_context() {
    return {};
  }

  /// The default specialization of `query_system_context` for `parallel_scheduler_backend`.
  template <>
  std::shared_ptr<parallel_scheduler_backend> query_system_context<parallel_scheduler_backend>() {
    return __system_context_default_impl::__parallel_scheduler_backend_singleton
      .__get_current_instance();
  }

  /// The default implementation of the `set_system_context_backend_factory` function template.
  template <__queryable_interface _Interface>
  __STDEXEC_SYSTEM_CONTEXT_API __system_context_backend_factory<_Interface>
    set_system_context_backend_factory(__system_context_backend_factory<_Interface> __new_factory) {
    return nullptr;
  }

  /// The default specialization of `set_system_context_backend_factory` for `parallel_scheduler_backend`.
  template <>
  __system_context_backend_factory<parallel_scheduler_backend>
    set_system_context_backend_factory<parallel_scheduler_backend>(
      __system_context_backend_factory<parallel_scheduler_backend> __new_factory) {
    return __system_context_default_impl::__parallel_scheduler_backend_singleton
      .__set_backend_factory(__new_factory);
  }


} // namespace exec::system_context_replaceability
