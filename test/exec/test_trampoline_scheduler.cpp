/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "../../include/exec/trampoline_scheduler.hpp"

#include "../test_common/retry.hpp"

#include <catch2/catch.hpp>
#include <memory>

namespace ex = STDEXEC;

namespace {

  struct try_again { };

  struct fails_alot {
    using sender_concept = ex::sender_t;
    using completion_signatures =
      ex::completion_signatures<ex::set_value_t(), ex::set_error_t(try_again)>;

    template <class Receiver>
    struct operation {
      Receiver rcvr_;
      int counter_;

      void start() & noexcept {
        if (counter_ == 0) {
          ex::set_value(static_cast<Receiver&&>(rcvr_));
        } else {
          ex::set_error(static_cast<Receiver&&>(rcvr_), try_again{});
        }
      }
    };

    template <ex::receiver_of<completion_signatures> Receiver>
    auto connect(Receiver rcvr) const -> operation<Receiver> {
      return {static_cast<Receiver&&>(rcvr), --*counter_};
    }

    std::shared_ptr<int> counter_ = std::make_shared<int>(1'000'000);
  };

  // #if defined(REQUIRE_TERMINATE)
  // // For some reason, when compiling with nvc++, the forked process dies with SIGSEGV
  // // but the error code returned from ::wait reports success, so this test fails.
  // TEST_CASE("running deeply recursing algo blows the stack", "[schedulers][trampoline_scheduler]") {

  //   auto recurse_deeply = retry(fails_alot{});
  //   REQUIRE_TERMINATE([&] { sync_wait(std::move(recurse_deeply)); });
  // }
  // #endif

  TEST_CASE(
    "running deeply recursing algo on trampoline_scheduler doesn't blow the stack",
    "[schedulers][trampoline_scheduler]") {

    exec::trampoline_scheduler sched;
    ex::run_loop loop;

    auto recurse_deeply = retry(ex::on(sched, fails_alot{}));
    ex::sync_wait(std::move(recurse_deeply));
  }

} // namespace
