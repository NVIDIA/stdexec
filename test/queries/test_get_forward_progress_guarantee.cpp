/*
 * Copyright (c) ETH Zurich
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
#include <test_common/schedulers.hpp>

namespace ex = std::execution;

namespace {
struct uncustomized_scheduler {
  struct operation_state {
    friend void tag_invoke(ex::start_t, operation_state& self) noexcept {}
  };

  struct sender {
    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
    template <typename R>
    friend operation_state tag_invoke(ex::connect_t, sender, R&&) {
      return {};
    }

    template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> CPO>
    friend uncustomized_scheduler tag_invoke(ex::get_completion_scheduler_t<CPO>, sender) noexcept {
      return {};
    }
  };

  friend sender tag_invoke(ex::schedule_t, uncustomized_scheduler) { return {}; }
  friend bool operator==(uncustomized_scheduler, uncustomized_scheduler) noexcept { return true; }
  friend bool operator!=(uncustomized_scheduler, uncustomized_scheduler) noexcept { return false; }
};

template <ex::forward_progress_guarantee fpg>
struct customized_scheduler {
  struct operation_state {
    friend void tag_invoke(ex::start_t, operation_state& self) noexcept {}
  };

  struct sender {
    using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;
    template <typename R>
    friend operation_state tag_invoke(ex::connect_t, sender, R&&) {
      return {};
    }

    template <std::__one_of<ex::set_value_t, ex::set_error_t, ex::set_stopped_t> CPO>
    friend customized_scheduler tag_invoke(ex::get_completion_scheduler_t<CPO>, sender) noexcept {
      return {};
    }
  };

  friend sender tag_invoke(ex::schedule_t, customized_scheduler) { return {}; }
  friend bool operator==(customized_scheduler, customized_scheduler) noexcept { return true; }
  friend bool operator!=(customized_scheduler, customized_scheduler) noexcept { return false; }

  constexpr friend ex::forward_progress_guarantee tag_invoke(
      ex::get_forward_progress_guarantee_t, customized_scheduler) {
    return fpg;
  }
};
}

TEST_CASE("get_forward_progress_guarantee ", "[sched_queries][get_forward_progress_guarantee]") {
  STATIC_REQUIRE(ex::get_forward_progress_guarantee(uncustomized_scheduler{}) ==
                 ex::forward_progress_guarantee::weakly_parallel);
  STATIC_REQUIRE(ex::get_forward_progress_guarantee(
                     customized_scheduler<ex::forward_progress_guarantee::concurrent>{}) ==
                 ex::forward_progress_guarantee::concurrent);
  STATIC_REQUIRE(ex::get_forward_progress_guarantee(
                     customized_scheduler<ex::forward_progress_guarantee::parallel>{}) ==
                 ex::forward_progress_guarantee::parallel);
  STATIC_REQUIRE(ex::get_forward_progress_guarantee(
                     customized_scheduler<ex::forward_progress_guarantee::weakly_parallel>{}) ==
                 ex::forward_progress_guarantee::weakly_parallel);
}
