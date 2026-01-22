/*
 * Copyright (c) 2025 Lucian Radu Teodorescu, Lewis Baker
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

#ifndef STDEXEC_SYSTEM_CONTEXT_REPLACEABILITY_API_H
#define STDEXEC_SYSTEM_CONTEXT_REPLACEABILITY_API_H

#include "../../stdexec/__detail/__execution_fwd.hpp"
#include "../../stdexec/__detail/__system_context_replaceability_api.hpp"

#include <cstddef>
#include <memory>

namespace exec::system_context_replaceability {
  using STDEXEC::system_context_replaceability::__parallel_scheduler_backend_factory;

  /// Interface for the parallel scheduler backend.
  using parallel_scheduler_backend [[deprecated(
    "Use STDEXEC::system_context_replaceability::parallel_scheduler_backend instead.")]] =
    STDEXEC::system_context_replaceability::parallel_scheduler_backend;

  /// Get the backend for the parallel scheduler.
  /// Users might replace this function.
  [[deprecated(
    "Use STDEXEC::system_context_replaceability::query_parallel_scheduler_backend instead.")]]
  inline auto query_parallel_scheduler_backend()
    -> std::shared_ptr<STDEXEC::system_context_replaceability::parallel_scheduler_backend> {
    return STDEXEC::system_context_replaceability::query_parallel_scheduler_backend();
  }

  /// Set a factory for the parallel scheduler backend.
  /// Can be used to replace the parallel scheduler at runtime.
  /// Out of spec.
  [[deprecated(
    "Use STDEXEC::system_context_replaceability::set_parallel_scheduler_backend instead.")]]
  inline auto set_parallel_scheduler_backend(__parallel_scheduler_backend_factory __new_factory)
    -> __parallel_scheduler_backend_factory {
    return STDEXEC::system_context_replaceability::set_parallel_scheduler_backend(__new_factory);
  }

  /// Interface for completing a sender operation. Backend will call frontend though this interface
  /// for completing the `schedule` and `schedule_bulk` operations.
  using receiver
    [[deprecated("Use STDEXEC::system_context_replaceability::receiver_proxy instead.")]] =
      STDEXEC::system_context_replaceability::receiver_proxy;

  /// Receiver for bulk scheduling operations.
  using bulk_item_receiver [[deprecated(
    "Use STDEXEC::system_context_replaceability::bulk_item_receiver_proxy instead.")]] =
    STDEXEC::system_context_replaceability::bulk_item_receiver_proxy;
} // namespace exec::system_context_replaceability

#endif
