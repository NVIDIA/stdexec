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

#include "../../include/exec/on.hpp"
#include "../test_common/require_terminate.hpp"
#include "../test_common/retry.hpp"

#include <catch2/catch.hpp>
#include <exception>
#include <memory>

using namespace stdexec;

struct try_again { };

struct fails_alot {
  using is_sender = void;
  using __t = fails_alot;
  using __id = fails_alot;
  using completion_signatures =
    stdexec::completion_signatures<set_value_t(), set_error_t(try_again)>;

  template <class Receiver>
  struct operation {
    Receiver rcvr_;
    int counter_;

    STDEXEC_DEFINE_CUSTOM(void start)(this operation& self, start_t) noexcept {
      if (self.counter_ == 0) {
        stdexec::set_value((Receiver&&) self.rcvr_);
      } else {
        stdexec::set_error((Receiver&&) self.rcvr_, try_again{});
      }
    }
  };

  template <receiver_of<completion_signatures> Receiver>
  STDEXEC_DEFINE_CUSTOM(operation<Receiver> connect)(
    this fails_alot self,
    connect_t,
    Receiver rcvr) {
    return {(Receiver&&) rcvr, --*self.counter_};
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

  using stdexec::__sync_wait::__env;
  exec::trampoline_scheduler sched;
  stdexec::run_loop loop;

  auto recurse_deeply = retry(exec::on(sched, fails_alot{}));
  sync_wait(std::move(recurse_deeply));
}
