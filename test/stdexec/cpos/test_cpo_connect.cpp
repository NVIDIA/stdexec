/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
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

#include "test_common/receivers.hpp"
#include "test_common/type_helpers.hpp"
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

namespace ex = STDEXEC;

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunneeded-internal-declaration")

namespace {

  template <typename R>
  struct op_state : immovable {
    int val_;
    R recv_;

    void start() & noexcept {
      ex::set_value(static_cast<R&&>(recv_), static_cast<int>(val_));
    }
  };

  struct my_sender {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(int)>;

    int value_{0};

    template <class R>
    [[nodiscard]]
    auto connect(R r) const -> op_state<R> {
      return {{}, value_, std::move(r)};
    }
  };

  struct my_sender_unconstrained {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(int)>;

    int value_{0};

    template <class R> // accept any type here
    [[nodiscard]]
    auto connect(R r) const -> op_state<R> {
      return {{}, value_, std::move(r)};
    }
  };

  TEST_CASE("can call connect on an appropriate types", "[cpo][cpo_connect]") {
    auto op = ex::connect(my_sender{10}, expect_value_receiver{10});
    ex::start(op);
    // the receiver will check the received value
  }

  TEST_CASE("cannot connect sender with invalid receiver", "[cpo][cpo_connect]") {
    static_assert(ex::sender<my_sender_unconstrained>);
    REQUIRE_FALSE(ex::sender_to<my_sender_unconstrained, int>);
  }

  TEST_CASE("tag types can be deduced from ex::connect", "[cpo][cpo_connect]") {
    static_assert(std::is_same_v<const ex::connect_t, decltype(ex::connect)>, "type mismatch");
  }
} // namespace

STDEXEC_PRAGMA_POP()
