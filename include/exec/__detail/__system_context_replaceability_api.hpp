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

struct __uuid {
  uint64_t __parts1;
  uint64_t __parts2;

  friend bool operator==(__uuid, __uuid) noexcept = default;
};

namespace exec::system_context_replaceability {

  //! Helper for the `__queryable_interface` concept.
  template <__uuid X>
  using __check_constexpr_uuid = void;

  //! Concept for a queryable interface. Ensures that the interface has a `__interface_identifier` member.
  template <typename _T>
  concept __queryable_interface =
    requires() { typename __check_constexpr_uuid<_T::__interface_identifier>; };

  /// Query the system context for an interface of type `_Interface`.
  template <__queryable_interface _Interface>
  extern _Interface* query_system_context();

  /// Sets the system context backend for an interface of type `_Interface`.
  template <__queryable_interface _Interface>
  extern bool set_system_context_backend(_Interface* __backend);

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

  /// Receiver for bulk sheduling operations.
  struct bulk_item_receiver : receiver {
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
    static constexpr __uuid __interface_identifier{0x5ee9202498c4bd4f, 0xa1df2508ffcd9d7e};

    virtual ~system_scheduler() = default;

    /// Schedule work on system scheduler, calling `__r` when done and using `__s` for preallocated memory.
    virtual void schedule(storage __s, receiver* __r) noexcept = 0;
    /// Schedule bulk work of size `__n` on system scheduler, calling `__r` for each item and then when done, and using `__s` for preallocated memory.
    virtual void bulk_schedule(uint32_t __n, storage __s, bulk_item_receiver* __r) noexcept = 0;
  };

} // namespace exec::system_context_replaceability

#endif