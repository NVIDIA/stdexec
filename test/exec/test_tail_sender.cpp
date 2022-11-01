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

    operation(Receiver __r) : rcvr_(__r) {}

    operation(const operation&) = delete;
    operation(operation&&) = delete;
    operation& operator=(const operation&) = delete;
    operation& operator=(operation&&) = delete;
    [[nodiscard]]
    friend auto tag_invoke(ex::start_t, operation& self) noexcept {
      return ex::set_value(std::move(self.rcvr_));
    }

    friend void tag_invoke(exec::unwind_t, operation& self) noexcept {
      ex::set_stopped(std::move(self.rcvr_));
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, ATailSender self, Receiver&& rcvr) noexcept
      -> operation<std::decay_t<Receiver>> {
    return {std::forward<Receiver>(rcvr)};
  }

  template<class _Env>
  friend constexpr bool tag_invoke(
      exec::always_completes_inline_t, exec::c_t<ATailSender>, exec::c_t<_Env>) noexcept {
    return true;
  }
};

struct ATailReceiver {
  int* called;
  friend void tag_invoke(ex::set_value_t, ATailReceiver&& __self, auto&&...) noexcept { ++*__self.called; }
  friend void tag_invoke(ex::set_stopped_t, ATailReceiver&& __self) noexcept { ++*__self.called; }
  friend ex::__debug_env_t<ex::__empty_env> tag_invoke(ex::get_env_t, const ATailReceiver&) {
    return {{}};
  }
};


template<class NestTailSender>
struct ANestTailReceiver {
  std::decay_t<NestTailSender> nested_tail_sender;
  int* called;
  [[nodiscard]]
  friend std::decay_t<NestTailSender> tag_invoke(ex::set_value_t, ANestTailReceiver&& __self, auto&&...) noexcept {
    ++*__self.called;
    return __self.nested_tail_sender;
  }
  [[nodiscard]]
  friend std::decay_t<NestTailSender> tag_invoke(ex::set_stopped_t, ANestTailReceiver&& __self) noexcept {
    ++*__self.called;
    return __self.nested_tail_sender;
  }
  friend ex::__debug_env_t<ex::__empty_env> tag_invoke(ex::get_env_t, const ANestTailReceiver&) {
    return {{}};
  }
};

TEST_CASE("Test ATailSender is a tail_sender", "[tail_sender]") {
  static_assert(exec::tail_sender<ATailSender>);
  static_assert(exec::__terminal_tail_sender_to<ATailSender, ATailReceiver>);
  CHECK(exec::tail_sender<ATailSender>);
  CHECK(exec::__terminal_tail_sender_to<ATailSender, ATailReceiver>);
}

TEST_CASE("Test __null_tail_sender is a tail_sender", "[tail_sender]") {
  static_assert(exec::tail_sender<exec::__null_tail_sender>);
  static_assert(exec::__terminal_tail_sender_to<exec::__null_tail_sender, exec::__null_tail_receiver>);
  CHECK(exec::tail_sender<exec::__null_tail_sender>);
  CHECK(exec::__terminal_tail_sender_to<exec::__null_tail_sender, exec::__null_tail_receiver>);
}

TEST_CASE("Test maybe_tail_sender is a tail_sender", "[tail_sender]") {
  static_assert(exec::tail_sender<exec::maybe_tail_sender<ATailSender>>);
  static_assert(exec::__nullable_tail_sender_to<exec::maybe_tail_sender<ATailSender>, ATailReceiver>);
  CHECK(exec::tail_sender<exec::maybe_tail_sender<ATailSender>>);
  CHECK(exec::__nullable_tail_sender_to<exec::maybe_tail_sender<ATailSender>, ATailReceiver>);
}

TEST_CASE("Test scoped_tail_sender", "[tail_sender]") {
  int called = 0;
  {
    exec::scoped_tail_sender<ATailSender, ATailReceiver> exit{ATailSender{}, ATailReceiver{&called}};
    CHECK(called == 0);
  }
  CHECK(called == 1);
}

TEST_CASE("Test __start_until_nullable()", "[tail_sender]") {
  static_assert(exec::__terminal_tail_sender_to<exec::maybe_tail_sender<ATailSender>, ATailReceiver>);
  CHECK(exec::__terminal_tail_sender_to<exec::maybe_tail_sender<ATailSender>, ATailReceiver>);

  int called = 0;
  CHECK(called == 0);
  exec::maybe_tail_sender<ATailSender> maybe =
    exec::__start_until_nullable(exec::maybe_tail_sender<ATailSender>{}, ATailReceiver{&called});
  CHECK(called == 0);
  auto op0 = ex::connect(std::move(maybe), ATailReceiver{&called});
  CHECK(called == 0);
  CHECK(!op0);

  called = 0;
  maybe =
    exec::__start_until_nullable(exec::maybe_tail_sender<ATailSender>{ATailSender{}}, ATailReceiver{&called});
  CHECK(called == 0);
  auto op1 = ex::connect(std::move(maybe), ATailReceiver{&called});
  CHECK(called == 0);
  CHECK(!!op1);
  ex::start(op1);
  CHECK(called == 1);

  called = 0;
  maybe =
    exec::__start_until_nullable(
      ATailSender{},
      ANestTailReceiver<exec::maybe_tail_sender<ATailSender>>{ATailSender{}, &called});
  CHECK(called == 1);
  auto op2 = ex::connect(std::move(maybe), ATailReceiver{&called});
  CHECK(called == 1);
  CHECK(!!op2);
  ex::start(op2);
  CHECK(called == 2);
}

#if 0
TEST_CASE("Test __resume_until_nullable()", "[tail_sender]") {
  int called = 0;
  CHECK(called == 0);
  exec::maybe_tail_sender<ATailSender> maybe =
    exec::__resume_until_nullable(exec::maybe_tail_sender<ATailSender>{}, ATailReceiver{&called});
  CHECK(called == 0);
  auto op0 = ex::connect(std::move(maybe), ATailReceiver{&called});
  CHECK(called == 0);
  CHECK(!op0);

  called = 0;
  maybe =
    exec::__resume_until_nullable(exec::maybe_tail_sender<ATailSender>{ATailSender{}}, ATailReceiver{&called});
  CHECK(called == 0);
  auto op1 = ex::connect(std::move(maybe), ATailReceiver{&called});
  CHECK(called == 0);
  CHECK(!!op1);
  ex::start(op1);
  CHECK(called == 1);

  called = 0;
  maybe =
    exec::__resume_until_nullable(
      ATailSender{},
      ANestTailReceiver<exec::maybe_tail_sender<ATailSender>>{ATailSender{}, &called});
  CHECK(called == 1);
  auto op2 = ex::connect(std::move(maybe), ATailReceiver{&called});
  CHECK(called == 1);
  CHECK(!!op2);
  ex::start(op2);
  CHECK(called == 2);
}
#endif
