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
  struct __exec_system_sender_interface;

  // Virtual interfaces to underlying implementations for initial simplicit
  // TODO: Potentially move to custom vtable implementations
  struct __exec_system_context_interface {
    virtual __exec_system_scheduler_interface* get_scheduler() noexcept = 0;
  };

  // bulk function for scheduler to transmit from, will wrap actual function stub stored in real type
  using __exec_system_bulk_shape = long;
  using __exec_system_bulk_fn = void(void*, __exec_system_bulk_shape);

  struct __exec_system_bulk_function_object {
    void* __fn_state = nullptr;
    __exec_system_bulk_fn* __fn = nullptr;
  };

  struct __exec_system_scheduler_interface {
    virtual stdexec::forward_progress_guarantee get_forward_progress_guarantee() const = 0;
    virtual __exec_system_sender_interface* schedule() = 0;
    // TODO: Move chaining in here to support chaining after a system_sender or other system_bulk_sender
    // or don't do anything that specific?
    virtual __exec_system_sender_interface*
      bulk(__exec_system_bulk_shape __shp, __exec_system_bulk_function_object __fn) = 0;
    virtual bool equals(const __exec_system_scheduler_interface* __rhs) const = 0;
  };

  struct __exec_system_operation_state_interface {
    virtual void start() noexcept = 0;
  };

  struct __exec_system_receiver {
    void* __cpp_recv_ = nullptr;
    void (*set_value)(void* __cpp_recv) noexcept;
    void (*set_stopped)(void* __cpp_recv) noexcept;
    // Type-erase the exception pointer for extern-c-ness
    void (*set_error)(void* __cpp_recv, void* __exception) noexcept;
  };

  struct __exec_system_sender_interface {
    virtual __exec_system_operation_state_interface*
      connect(__exec_system_receiver __recv) noexcept = 0;
    virtual __exec_system_scheduler_interface* get_completion_scheduler() noexcept = 0;
  };

}}
