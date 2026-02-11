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

#include "../stdexec/__detail/__execution_fwd.hpp"

#include "../stdexec/__detail/__concepts.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__schedulers.hpp"
#include "../stdexec/__detail/__senders.hpp"
#include "../stdexec/__detail/__then.hpp"
#include "../stdexec/__detail/__transform_sender.hpp"

#include "start_detached.hpp"

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  struct __execute_t {
    template <STDEXEC::scheduler _Scheduler, class _Fun>
      requires STDEXEC::__callable<_Fun&> && STDEXEC::__std::move_constructible<_Fun>
    void operator()(_Scheduler&& __sched, _Fun __fun) const noexcept(false) {
      auto __domain = STDEXEC::get_domain(__sched);
      STDEXEC::apply_sender(
        __domain, *this, STDEXEC::schedule(static_cast<_Scheduler&&>(__sched)), static_cast<_Fun&&>(__fun));
    }

    template <STDEXEC::sender_of<STDEXEC::set_value_t()> _Sender, class _Fun>
      requires STDEXEC::__callable<_Fun&> && STDEXEC::__std::move_constructible<_Fun>
    void apply_sender(_Sender&& __sndr, _Fun __fun) const noexcept(false) {
      exec::start_detached(STDEXEC::then(static_cast<_Sender&&>(__sndr), static_cast<_Fun&&>(__fun)));
    }
  };

  inline constexpr __execute_t __execute{};

  using execute_t [[deprecated]] = __execute_t;
  [[deprecated]]
  inline constexpr const execute_t& execute = __execute;
} // namespace exec

