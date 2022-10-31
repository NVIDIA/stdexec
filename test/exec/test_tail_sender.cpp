/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

#include <exec/tail_sender.hpp>

using namespace std;
namespace ex = stdexec;

//! Tail Sender
struct ATailSender {
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>;

  template <class Receiver>
  struct operation {
    Receiver rcvr_;

    friend auto tag_invoke(ex::start_t, operation& self) noexcept {
      return ex::set_value(std::move(self.rcvr_));
    }

    friend void tag_invoke(exec::unwind_t, operation& self) noexcept {
      ex::set_stopped(std::move(self.rcvr_));
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, ATailSender&& self, Receiver&& rcvr)
      -> operation<std::decay_t<Receiver>> {
    return {std::forward<Receiver>(rcvr)};
  }

  template<class _Env>
  friend constexpr bool tag_invoke(
      exec::always_completes_inline_t, exec::c_t<ATailSender>, exec::c_t<_Env>) noexcept {
    return true;
  }
};

TEST_CASE("Test ATailSender is a tail_sender", "[tail_sender]") {
  static_assert(exec::tail_sender<ATailSender>);
  CHECK(exec::tail_sender<ATailSender>);
}
