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

#ifndef __EXEC__SYSTEM_CONTEXT_IF_H__
#define __EXEC__SYSTEM_CONTEXT_IF_H__

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct __exec_system_context_interface;
struct __exec_system_scheduler_interface;

/// Interface that allows interaction with the system context, allowing scheduling work on the system.
struct __exec_system_context_interface {
  /// The supported version of the system context interface, in the form YYYYMM.
  uint32_t __version;

  /// Returns an interface to the system scheduler.
  struct __exec_system_scheduler_interface* (*__get_scheduler)(
    struct __exec_system_context_interface* /*self*/);
};

/// Callback to be called by the scheduler when new work can start.
typedef void (*__exec_system_context_completion_callback_t)(
  void*,  // data pointer passed to scheduler
  int,    // completion type: 0 for normal completion, 1 for cancellation, 2 for exception
  void*); // If completion type is 2, this is the exception pointer.

/// Callback to be called by the scheduler for each bulk item.
typedef void (*__exec_system_context_bulk_item_callback_t)(
  void*,          // data pointer passed to scheduler
  unsigned long); // the index of the work item that is starting

struct __exec_system_scheduler_interface {
  /// The forward progress guarantee of the scheduler.
  ///
  /// 0 == concurrent, 1 == parallel, 2 == weakly_parallel
  uint32_t __forward_progress_guarantee;

  /// The size of the operation state object on the implementation side.
  uint32_t __schedule_operation_size;
  /// The alignment of the operation state object on the implementation side.
  uint32_t __schedule_operation_alignment;

  /// Schedules new work on the system scheduler, calling `cb` with `data` when the work can start.
  /// Returns an object that should be passed to __destruct_schedule_operation when the operation completes.
  void* (*__schedule)(
    struct __exec_system_scheduler_interface* /*self*/,
    void* /*__preallocated*/,
    uint32_t /*__psize*/,
    __exec_system_context_completion_callback_t /*cb*/,
    void* /*data*/);

  /// Destructs the operation state object.
  void (*__destruct_schedule_operation)(
    struct __exec_system_scheduler_interface* /*self*/,
    void* /*operation*/);

  /// The size of the operation state object on the implementation side.
  uint32_t __bulk_schedule_operation_size;
  /// The alignment of the operation state object on the implementation side.
  uint32_t __bulk_schedule_operation_alignment;

  /// Schedules new bulk work of size `size` on the system scheduler, calling `cb_item` with `data`
  /// for indices in [0, `size`), and calling `cb` on general completion.
  /// Returns the operation state object that should be passed to __destruct_bulk_schedule_operation.
  void* (*__bulk_schedule)(
    struct __exec_system_scheduler_interface* /*self*/,
    void* /*__preallocated*/,
    uint32_t /*__psize*/,
    __exec_system_context_completion_callback_t /*cb*/,
    __exec_system_context_bulk_item_callback_t /*cb_item*/,
    void* /*data*/,
    unsigned long /*size*/);

  /// Destructs the operation state object for a bulk_schedule.
  void (*__destruct_bulk_schedule_operation)(
    struct __exec_system_scheduler_interface* /*self*/,
    void* /*operation*/);
};

#ifdef __cplusplus
}
#endif


#endif
