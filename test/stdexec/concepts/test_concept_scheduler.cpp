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

namespace ex = STDEXEC;

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunneeded-internal-declaration")

namespace {

  template <class Scheduler>
  struct default_env {
    template <typename CPO>
    [[nodiscard]]
    auto query(ex::get_completion_scheduler_t<CPO>, ex::__ignore = {}) const noexcept -> Scheduler {
      return {};
    }
  };

  struct my_scheduler {
    struct my_sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = ex::completion_signatures<
        ex::set_value_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_stopped_t()
      >;

      [[nodiscard]]
      auto get_env() const noexcept -> default_env<my_scheduler> {
        return {};
      }
    };

    template <typename CPO>
    [[nodiscard]]
    auto
      query(ex::get_completion_scheduler_t<CPO>, ex::__ignore = {}) const noexcept -> my_scheduler {
      return {};
    }

    [[nodiscard]]
    auto schedule() const -> my_sender {
      return {};
    }

    friend auto operator==(my_scheduler, my_scheduler) noexcept -> bool {
      return true;
    }

    friend auto operator!=(my_scheduler, my_scheduler) noexcept -> bool {
      return false;
    }
  };

  TEST_CASE("type with schedule CPO models scheduler", "[concepts][scheduler]") {
    REQUIRE(ex::scheduler<my_scheduler>);
    REQUIRE(ex::sender<decltype(ex::schedule(my_scheduler{}))>);
  }

  struct no_schedule_cpo { };

  TEST_CASE("type without schedule CPO doesn't model scheduler", "[concepts][scheduler]") {
    REQUIRE(!ex::scheduler<no_schedule_cpo>);
  }

  struct my_scheduler_except {
    struct my_sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = ex::completion_signatures<
        ex::set_value_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_stopped_t()
      >;

      [[nodiscard]]
      auto get_env() const noexcept -> default_env<my_scheduler_except> {
        return {};
      }
    };

    [[nodiscard]]
    auto schedule() const -> my_sender {
      STDEXEC_THROW(std::logic_error("err"));
      return {};
    }

    friend auto operator==(my_scheduler_except, my_scheduler_except) noexcept -> bool {
      return true;
    }

    friend auto operator!=(my_scheduler_except, my_scheduler_except) noexcept -> bool {
      return false;
    }
  };

  TEST_CASE("type with schedule that throws is a scheduler", "[concepts][scheduler]") {
    REQUIRE(ex::scheduler<my_scheduler_except>);
  }

  struct noeq_sched {
    struct my_sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = ex::completion_signatures<
        ex::set_value_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_stopped_t()
      >;

      [[nodiscard]]
      auto get_env() const noexcept -> default_env<noeq_sched> {
        return {};
      }
    };

    [[nodiscard]]
    auto schedule() const -> my_sender {
      return {};
    }
  };

  TEST_CASE("type w/o equality operations do not model scheduler", "[concepts][scheduler]") {
    REQUIRE(!ex::scheduler<noeq_sched>);
  }
} // namespace

STDEXEC_PRAGMA_POP()