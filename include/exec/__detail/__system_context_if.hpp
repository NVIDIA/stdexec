/*
 * Copyright (c) 2023 Lee Howes, Lucian Radu Teodorescu
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

namespace exec { namespace __system_context_interface {

  // TODO: transform this into a C interface

  struct __exec_system_scheduler_interface;

  // Virtual interfaces to underlying implementations for initial simplicit
  // TODO: Potentially move to custom vtable implementations
  struct __exec_system_context_interface {
    virtual __exec_system_scheduler_interface* get_scheduler() noexcept = 0;
  };

  // bulk function for scheduler to transmit from, will wrap actual function stub stored in real type
  using __exec_system_bulk_shape = long;
  using __exec_system_bulk_fn = void(void*, __exec_system_bulk_shape);

  /// Callback to be called by the scheduler when new work can start.
  ///
  /// \param data The data pointer passed to the scheduler.
  /// \param completion_type 0 for normal completion, 1 for cancellation, 2 for exception
  /// \param exception If completion_type is 2, this is the exception pointer.
  using __exec_system_context_schedule_callback_t =
    void (*)(void* /*data*/, int /*completion_type*/, void* /*exception*/);

  /// Callback to be called by the scheduler for each bulk item.
  ///
  /// \param data The data pointer passed to the scheduler.
  /// \param index The index of the work item that is starting.
  using __exec_system_context_bulk_item_callback_t = void (*)(void* /*data*/, long /*index*/);

  struct __exec_system_scheduler_interface {
    virtual stdexec::forward_progress_guarantee get_forward_progress_guarantee() const = 0;

    /// Schedules new work on the system scheduler, calling `__cb` with `__data` when the work can start.
    virtual void schedule(__exec_system_context_schedule_callback_t __cb, void* __data) = 0;

    /// Schedules new bulk work of size `size` on the system scheduler, calling `__cb_item` with `__data` for indices in [0, `size`), and calling `__cb` on general completion.
    virtual void bulk_schedule(
      __exec_system_context_schedule_callback_t __cb,
      __exec_system_context_bulk_item_callback_t __cb_item,
      void* __data,
      long size) = 0;

    virtual bool equals(const __exec_system_scheduler_interface* __rhs) const = 0;
  };

}}
