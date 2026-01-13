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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

namespace {

  struct my_oper : immovable {
    bool started_{false};

    void start() noexcept {
      started_ = true;
    }
  };

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")

  struct op_rvalref : immovable {
    bool* started_;

    void start() && noexcept {
      *started_ = true;
    }
  };
  STDEXEC_PRAGMA_POP()

  struct op_ref : immovable {
    bool* started_;

    void start() & noexcept {
      *started_ = true;
    }
  };

  struct op_cref : immovable {
    bool* started_;

    void start() const noexcept {
      *started_ = true;
    }
  };

  TEST_CASE("can call start on an operation state", "[cpo][cpo_start]") {
    my_oper op;
    ex::start(op);
    REQUIRE(op.started_);
  }

  TEST_CASE("can call start on an oper with r-value ref type", "[cpo][cpo_start]") {
    static_assert(
      !std::invocable<ex::start_t, op_rvalref&&>, "should not be able to call start on op_rvalref");
  }

  TEST_CASE("can call start on an oper with ref type", "[cpo][cpo_start]") {
    static_assert(std::invocable<ex::start_t, op_ref&>, "cannot call start on op_ref");
    bool started{false};
    op_ref op{{}, &started};
    ex::start(op);
    REQUIRE(started);
  }

  TEST_CASE("can call start on an oper with const ref type", "[cpo][cpo_start]") {
    static_assert(std::invocable<ex::start_t, const op_cref&>, "cannot call start on op_cref");
    bool started{false};
    const op_cref op{{}, &started};
    ex::start(op);
    REQUIRE(started);
  }

  TEST_CASE("tag types can be deduced from ex::start", "[cpo][cpo_start]") {
    static_assert(std::is_same_v<const ex::start_t, decltype(ex::start)>, "type mismatch");
  }
} // namespace
