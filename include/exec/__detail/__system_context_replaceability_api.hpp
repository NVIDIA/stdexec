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

#include "stdexec/__detail/__execution_fwd.hpp"

#include <typeindex>


/// Implementation-defined mechanism of querying a system context interface of type `id`.
extern void* __query_system_context_interface(std::type_index id) noexcept;

namespace exec::system_context_replaceability {
  /// Query the system context for an interface of type `_Interface`.
  template <typename _Interface>
  inline _Interface* query_system_context() {
    return static_cast<_Interface*>(__query_system_context_interface(typeid(_Interface)));
  }

  /// Interface for completing a sender operation.
  /// Backend will call frontend though this interface for completing the `schedule` and `schedule_bulk` operations.
  struct receiver {
    virtual ~receiver() = default;

    /// Called when the system scheduler completes successfully.
    virtual void set_value() noexcept = 0;
    /// Called when the system scheduler completes with an error.
    virtual void set_error(std::exception_ptr) noexcept = 0;
    /// Called when the system scheduler was stopped.
    virtual void set_stopped() noexcept = 0;
  };

  /// Interface for receiving bulk item signals.
  struct bulk_item_receiver {
    virtual ~bulk_item_receiver() = default;

    /// Called for each item of a bulk operation, possible on different threads.
    virtual void start(uint32_t) noexcept = 0;
  };

  /// Describes a storage space.
  /// Used to pass preallocated storage from the frontend to the backend.
  struct storage {
    void* __data;
    uint32_t __size;
  };

  /// Interface for the system scheduler
  struct system_scheduler {
    virtual ~system_scheduler() = default;

    /// Schedule work on system scheduler, calling `__r` when done and using `__s` for preallocated memory.
    virtual void schedule(storage __s, receiver* __r) noexcept = 0;
    /// Schedule bulk work of size `__n` on system scheduler, calling `__br` for each item, calling `__r` when done and using `__s` for preallocated memory.
    virtual void
      bulk_schedule(uint32_t __n, storage __s, receiver* __r, bulk_item_receiver* __br) noexcept = 0;
    /// Returns the maximum concurrency supported by the system scheduler.
    virtual uint32_t max_concurrency() noexcept = 0;
    /// Get the forward progress guarantee promised by the system scheduler.
    virtual stdexec::forward_progress_guarantee get_forward_progress_guarantee() noexcept = 0;
  };

  /// Implementation-defined mechanism for replacing the system scheduler backend at run-time.
  struct __system_context_replaceability {
    /// Globally replaces the system scheduler backend.
    /// This needs to be called within `main()` and before the system scheduler is accessed.
    virtual void __set_system_scheduler(system_scheduler*) noexcept = 0;
  };

} // namespace exec::system_context_replaceability

#endif