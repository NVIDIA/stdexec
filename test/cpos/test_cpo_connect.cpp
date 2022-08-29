/*
 * Copyright (c) Lucian Radu Teodorescu
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
#include <execution.hpp>
#include "test_common/receivers.hpp"
#include "test_common/type_helpers.hpp"

namespace ex = std::execution;

template <typename R>
struct op_state : immovable {
  int val_;
  R recv_;

  friend void tag_invoke(ex::start_t, op_state& self) noexcept {
    ex::set_value((R &&) self.recv_, self.val_);
  }
};

struct my_sender  {
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(int)>;

  int value_{0};

  template <class R>
  friend op_state<R> tag_invoke(ex::connect_t, my_sender&& s, R&& r) {
    return {{}, s.value_, (R &&) r};
  }
};

struct my_sender_unconstrained {
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(int)>;

  int value_{0};

  template <class R> // accept any type here
  friend op_state<R> tag_invoke(ex::connect_t, my_sender_unconstrained&& s, R&& r) {
    return {{}, s.value_, (R &&) r};
  }
};

TEST_CASE("can call connect on an appropriate types", "[cpo][cpo_connect]") {
  auto op = ex::connect(my_sender{10}, expect_value_receiver{10});
  ex::start(op);
  // the receiver will check the received value
}

TEST_CASE("cannot connect sender with invalid receiver", "[cpo][cpo_connect]") {
  static_assert(ex::sender<my_sender_unconstrained>);
  REQUIRE_FALSE(std::invocable<ex::connect_t, my_sender_unconstrained, int>);
}

struct strange_receiver {
  bool* called_;

  friend inline op_state<strange_receiver> tag_invoke(
      ex::connect_t, my_sender, strange_receiver self) {
    *self.called_ = true;
    // NOLINTNEXTLINE
    return {{}, 19, std::move(self)};
  }

  friend inline void tag_invoke(ex::set_value_t, strange_receiver, int val) noexcept { REQUIRE(val == 19); }
  friend void tag_invoke(ex::set_stopped_t, strange_receiver) noexcept {}
  friend void tag_invoke(ex::set_error_t, strange_receiver, std::exception_ptr) noexcept {}
  friend empty_env tag_invoke(ex::get_env_t, const strange_receiver&) noexcept {
    return {};
  }
};

TEST_CASE("connect can be defined in the receiver", "[cpo][cpo_connect]") {
  static_assert(ex::sender_to<my_sender, strange_receiver>);
  bool called{false};
  auto op = ex::connect(my_sender{10}, strange_receiver{&called});
  ex::start(op);
  REQUIRE(called);
}

TEST_CASE("tag types can be deduced from ex::connect", "[cpo][cpo_connect]") {
  static_assert(std::is_same_v<const ex::connect_t, decltype(ex::connect)>, "type mismatch");
}
