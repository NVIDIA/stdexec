/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp" // IWYU pragma: keep

#include "__concepts.hpp"
#include "__receivers.hpp"
#include "__senders.hpp"
#include "__schedulers.hpp"
#include "__submit.hpp"
#include "__tag_invoke.hpp"
#include "__transform_sender.hpp"

#include <exception>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  namespace __execute_ {
    template <class _Fun>
    struct __as_receiver {
      using receiver_concept = receiver_t;
      _Fun __fun_;

      void set_value() noexcept {
        // terminates on exception:
        __fun_();
      }

      [[noreturn]]
      void set_error(std::exception_ptr) noexcept {
        std::terminate();
      }

      void set_stopped() noexcept {
      }
    };

    struct execute_t {
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const noexcept(false) {
        // Look for a legacy customization
        if constexpr (tag_invocable<execute_t, _Scheduler, _Fun>) {
          tag_invoke(execute_t{}, static_cast<_Scheduler&&>(__sched), static_cast<_Fun&&>(__fun));
        } else {
          auto __domain = query_or(get_domain, __sched, default_domain());
          stdexec::apply_sender(
            __domain,
            *this,
            schedule(static_cast<_Scheduler&&>(__sched)),
            static_cast<_Fun&&>(__fun));
        }
      }

      template <sender_of<set_value_t()> _Sender, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void apply_sender(_Sender&& __sndr, _Fun __fun) const noexcept(false) {
        __submit(static_cast<_Sender&&>(__sndr), __as_receiver<_Fun>{static_cast<_Fun&&>(__fun)});
      }
    };
  } // namespace __execute_

  using __execute_::execute_t;
  inline constexpr execute_t execute{};
} // namespace stdexec
