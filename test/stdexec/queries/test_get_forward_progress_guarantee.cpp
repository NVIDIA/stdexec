/*
 * Copyright (c) 2022 ETH Zurich
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
#include <test_common/schedulers.hpp>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")

namespace ex = STDEXEC;

namespace {
  struct uncustomized_scheduler {
    struct operation_state {
      void start() & noexcept {
      }
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

      template <typename R>
      auto connect(R&&) const -> operation_state {
        return {};
      }

      struct env {
        template <STDEXEC::__completion_tag Tag>
        auto query(ex::get_completion_scheduler_t<Tag>) const noexcept -> uncustomized_scheduler {
          return {};
        }
      };

      [[nodiscard]]
      auto get_env() const noexcept -> env {
        return {};
      }
    };

    [[nodiscard]]
    auto schedule() const noexcept -> sender {
      return {};
    }

    auto operator==(const uncustomized_scheduler&) const noexcept -> bool = default;
  };

  template <ex::forward_progress_guarantee fpg>
  struct customized_scheduler {
    struct operation_state {
      void start() & noexcept {
      }
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        ex::completion_signatures<ex::set_value_t(), ex::set_error_t(std::exception_ptr)>;

      template <typename R>
      auto connect(R&&) const -> operation_state {
        return {};
      }

      struct env {
        template <STDEXEC::__completion_tag Tag>
        auto query(ex::get_completion_scheduler_t<Tag>) const noexcept -> customized_scheduler {
          return {};
        }
      };

      auto get_env() const noexcept -> env {
        return {};
      }
    };

    [[nodiscard]]
    auto schedule() const -> sender {
      return {};
    }

    friend auto operator==(customized_scheduler, customized_scheduler) noexcept -> bool {
      return true;
    }

    friend auto operator!=(customized_scheduler, customized_scheduler) noexcept -> bool {
      return false;
    }

    [[nodiscard]]
    constexpr auto
      query(ex::get_forward_progress_guarantee_t) const noexcept -> ex::forward_progress_guarantee {
      return fpg;
    }
  };

  TEST_CASE("get_forward_progress_guarantee ", "[sched_queries][get_forward_progress_guarantee]") {
    STATIC_REQUIRE(
      ex::get_forward_progress_guarantee(uncustomized_scheduler{})
      == ex::forward_progress_guarantee::weakly_parallel);
    STATIC_REQUIRE(
      ex::get_forward_progress_guarantee(
        customized_scheduler<ex::forward_progress_guarantee::concurrent>{})
      == ex::forward_progress_guarantee::concurrent);
    STATIC_REQUIRE(
      ex::get_forward_progress_guarantee(
        customized_scheduler<ex::forward_progress_guarantee::parallel>{})
      == ex::forward_progress_guarantee::parallel);
    STATIC_REQUIRE(
      ex::get_forward_progress_guarantee(
        customized_scheduler<ex::forward_progress_guarantee::weakly_parallel>{})
      == ex::forward_progress_guarantee::weakly_parallel);
  }

} // namespace

STDEXEC_PRAGMA_POP()
