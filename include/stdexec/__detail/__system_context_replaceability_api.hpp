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

#pragma once

#include "__execution_fwd.hpp"
#include "__parallel_scheduler_backend.hpp"

#include <memory>

namespace STDEXEC::system_context_replaceability {
  /// The type of a factory that can create `parallel_scheduler_backend` instances.
  /// TODO(ericniebler): NOT TO SPEC.
  using __parallel_scheduler_backend_factory = std::shared_ptr<parallel_scheduler_backend> (*)();

  /// Get the backend for the parallel scheduler.
  /// Users might replace this function.
  auto query_parallel_scheduler_backend() -> std::shared_ptr<parallel_scheduler_backend>;

  /// Set a factory for the parallel scheduler backend.
  /// Can be used to replace the parallel scheduler at runtime.
  /// TODO(ericniebler): NOT TO SPEC.
  auto set_parallel_scheduler_backend(__parallel_scheduler_backend_factory __new_factory)
    -> __parallel_scheduler_backend_factory;
} // namespace STDEXEC::system_context_replaceability
