/*
 * Copyright (c) 2026 NVIDIA Corporation
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

// Built with STDEXEC_ENABLE_EXTRA_TYPE_CHECKING, -Wsystem-headers and
// -Werror=deprecated-declarations.

#include <stdexec/execution.hpp>

#include <utility>

namespace ex = STDEXEC;

// Declares set_value_t(int), but completes with an int lvalue.
struct int_lvalue_sender
{
  using sender_concept        = ex::sender_t;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(int)>;

  template <class Rcvr>
  struct opstate
  {
    using operation_state_concept = ex::operation_state_t;

    void start() & noexcept
    {
      int value = 42;
      ex::set_value(std::move(rcvr_), value);
    }

    Rcvr rcvr_;
  };

  template <class Rcvr>
  auto connect(Rcvr rcvr) const -> opstate<Rcvr>
  {
    return {std::move(rcvr)};
  }
};

auto main() -> int
{
  // build error: _SIGNAL_SENT_BY_SENDER_<int_lvalue_sender>
  [[maybe_unused]]
  auto result = ex::sync_wait(int_lvalue_sender{});
}
