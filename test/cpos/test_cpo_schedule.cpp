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

namespace ex = std::execution;
using set_value_t = std::decay_t<decltype(ex::set_value)>;
using set_error_t = std::decay_t<decltype(ex::set_error)>;
using set_stopped_t = std::decay_t<decltype(ex::set_stopped)>;

struct my_sender : ex::completion_signatures<           //
                       set_value_t(),                   //
                       set_error_t(std::exception_ptr), //
                       set_stopped_t()> {

  bool from_scheduler_{false};
};

struct my_scheduler {
  friend my_sender tag_invoke(decltype(ex::schedule), my_scheduler) { return my_sender{{}, true}; }
};

TEST_CASE("can call schedule on an appropriate type", "[cpo][cpo_schedule]") {
  static_assert(std::invocable<decltype(ex::schedule), my_scheduler>, "invalid scheduler type");
  my_scheduler sched;
  auto snd = ex::schedule(sched);
  CHECK(snd.from_scheduler_);
}
