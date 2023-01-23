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

namespace ex = stdexec;

struct my_sender {
  using completion_signatures =
    ex::completion_signatures<             //
      ex::set_value_t(),                   //
      ex::set_error_t(std::exception_ptr), //
      ex::set_stopped_t()>;

  bool from_scheduler_{false};

  friend empty_attrs tag_invoke(ex::get_attrs_t, const my_sender&) noexcept {
    return {};
  }
};

struct my_scheduler {
  friend my_sender tag_invoke(ex::schedule_t, my_scheduler) { return my_sender{true}; }
};

TEST_CASE("can call schedule on an appropriate type", "[cpo][cpo_schedule]") {
  static_assert(std::invocable<ex::schedule_t, my_scheduler>, "invalid scheduler type");
  my_scheduler sched;
  auto snd = ex::schedule(sched);
  CHECK(snd.from_scheduler_);
}

TEST_CASE("tag types can be deduced from ex::schedule", "[cpo][cpo_schedule]") {
  static_assert(std::is_same_v<const ex::schedule_t, decltype(ex::schedule)>, "type mismatch");
}
