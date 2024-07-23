/*
 * Copyright (c) 2024 Lee Howes, Lucian Radu Teodorescu
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

#ifndef STDEXEC_SYSTEM_CONTEXT_IF_H
#define STDEXEC_SYSTEM_CONTEXT_IF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#  define STDEXEC_SYSTEM_CONTEXT_NOEXCEPT noexcept
namespace exec {
  extern "C" {
#else
#  define STDEXEC_SYSTEM_CONTEXT_NOEXCEPT
#endif

  struct system_context_interface;
  struct system_scheduler_interface;

  union __system_context_interface_reserved {
    void* __object_ptr;
    void* (*__function_ptr)();
    uintptr_t __int;
  };

  /// Interface that allows interaction with the system context, allowing scheduling work on the system.
  struct system_context_interface {
    /// The supported version of the system context interface, in the form YYYYMM.
    uint32_t version;

    /// The ref count of the system context.
    uint32_t ref_count;

    /// Destroys an instance of the system scheduler.
    void (*destroy_fn)(struct system_context_interface* /*self*/) STDEXEC_SYSTEM_CONTEXT_NOEXCEPT;

    /// Returns an interface to the system scheduler.
    struct system_scheduler_interface* (*get_scheduler_fn)(
      struct system_context_interface* /*self*/) STDEXEC_SYSTEM_CONTEXT_NOEXCEPT;

    /// Unused slots for future expansion.
    union __system_context_interface_reserved __reserved_for_future_use[8];
  };

  /// Function pointer type for registering a new system context.
  typedef system_context_interface* (*new_system_context_handler)();

  /// Sets the handler for creating new system contexts.
  /// Usage:
  ///  auto old_handler = exec::set_new_system_context_handler(
  ///    []() -> exec::system_context_interface* {
  ///      return new my_system_context_impl{};
  ///    });
  new_system_context_handler set_new_system_context_handler(new_system_context_handler /*handler*/);

  /// Callback to be called by the scheduler when new work can start.
  typedef void (*system_context_completion_callback)(
    void*,  // data pointer passed to scheduler
    int,    // completion type: 0 for normal completion, 1 for cancellation, 2 for exception
    void*); // If completion type is 2, this is the exception pointer.

  /// Callback to be called by the scheduler for each bulk item.
  typedef void (*system_context_bulk_item_callback)(
    void*,     // data pointer passed to scheduler
    uint64_t); // the index of the work item that is starting

  struct system_operation_state { };

  struct system_bulk_operation_state { };

  struct system_scheduler_interface {
    /// The forward progress guarantee of the scheduler.
    ///
    /// 0 == concurrent, 1 == parallel, 2 == weakly_parallel
    uint32_t forward_progress_guarantee;

    /// The size of the operation state object on the implementation side.
    uint32_t schedule_operation_size;
    /// The alignment of the operation state object on the implementation side.
    uint32_t schedule_operation_alignment;

    /// Schedules new work on the system scheduler, calling `cb` with `data` when the work can start.
    /// Returns an object that should be passed to destroy_schedule_operation_fn when the operation completes.
    struct system_operation_state* (*schedule_fn)(
      struct system_scheduler_interface* /*self*/,
      void* /*__preallocated*/,
      uint32_t /*__psize*/,
      system_context_completion_callback /*cb*/,
      void* /*data*/);

    /// Destructs the operation state object.
    void (*destroy_schedule_operation_fn)(
      struct system_scheduler_interface* /*self*/,
      struct system_operation_state* /*operation*/) STDEXEC_SYSTEM_CONTEXT_NOEXCEPT;

    /// The size of the operation state object on the implementation side.
    uint32_t bulk_schedule_operation_size;
    /// The alignment of the operation state object on the implementation side.
    uint32_t bulk_schedule_operation_alignment;

    /// Schedules new bulk work of size `size` on the system scheduler, calling `cb_item` with `data`
    /// for indices in [0, `size`), and calling `cb` on general completion.
    /// Returns the operation state object that should be passed to destroy_bulk_schedule_operation_fn.
    struct system_bulk_operation_state* (*bulk_schedule_fn)(
      struct system_scheduler_interface* /*self*/,
      void* /*__preallocated*/,
      uint32_t /*__psize*/,
      system_context_completion_callback /*cb*/,
      system_context_bulk_item_callback /*cb_item*/,
      void* /*data*/,
      uint64_t /*size*/);

    /// Destructs the operation state object for a bulk_schedule.
    void (*destroy_bulk_schedule_operation_fn)(
      struct system_scheduler_interface* /*self*/,
      struct system_bulk_operation_state* /*operation*/) STDEXEC_SYSTEM_CONTEXT_NOEXCEPT;

    /// Unused slots for future expansion.
    union __system_context_interface_reserved __reserved_for_future_use[8];
  };

#ifdef __cplusplus
  } // extern "C"
} // namespace exec
#endif

#endif
