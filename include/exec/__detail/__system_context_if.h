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

#ifdef __cplusplus
extern "C" {
#endif

struct __exec_system_context_interface;
struct __exec_system_scheduler_interface;

/// Interface that allows interaction with the system context, allowing scheduling work on the system.
struct __exec_system_context_interface {
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
  void*, // data pointer passed to scheduler
  long); // the index of the work item that is starting

struct __exec_system_scheduler_interface {
  /// Gets the forward progress guarantee of the scheduler.
  ///
  /// 0 == concurrent, 1 == parallel, 2 == weakly_parallel
  int (*__get_forward_progress_guarantee)(struct __exec_system_scheduler_interface* /*self*/);

  /// Schedules new work on the system scheduler, calling `cb` with `data` when the work can start.
  void (*__schedule)(
    struct __exec_system_scheduler_interface* /*self*/,
    __exec_system_context_completion_callback_t /*cb*/,
    void* /*data*/);

  /// Schedules new bulk work of size `size` on the system scheduler, calling `cb_item` with `data`
  /// for indices in [0, `size`), and calling `cb` on general completion.
  void (*__bulk_schedule)(
    struct __exec_system_scheduler_interface* /*self*/,
    __exec_system_context_completion_callback_t /*cb*/,
    __exec_system_context_bulk_item_callback_t /*cb_item*/,
    void* /*data*/,
    long /*size*/);
};

#ifdef __cplusplus
}
#endif


#endif
