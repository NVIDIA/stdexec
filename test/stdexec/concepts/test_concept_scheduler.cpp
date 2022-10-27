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

namespace ex = stdexec;

struct my_scheduler {
  struct my_sender  {
    using completion_signatures =
      ex::completion_signatures<             //
        ex::set_value_t(),                   //
        ex::set_error_t(std::exception_ptr), //
        ex::set_stopped_t()>;

    template <typename CPO>
    friend my_scheduler tag_invoke(ex::get_completion_scheduler_t<CPO>, my_sender) {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, my_scheduler) { return {}; }

  friend bool operator==(my_scheduler, my_scheduler) noexcept { return true; }
  friend bool operator!=(my_scheduler, my_scheduler) noexcept { return false; }
};

TEST_CASE("type with schedule CPO models scheduler", "[concepts][scheduler]") {
  REQUIRE(ex::scheduler<my_scheduler>);
  REQUIRE(ex::sender<decltype(ex::schedule(my_scheduler{}))>);
}

struct no_schedule_cpo {
  friend void tag_invoke(int, no_schedule_cpo) {}
};

TEST_CASE("type without schedule CPO doesn't model scheduler", "[concepts][scheduler]") {
  REQUIRE(!ex::scheduler<no_schedule_cpo>);
}

struct my_scheduler_except {
  struct my_sender {
    using completion_signatures =
      ex::completion_signatures<             //
        ex::set_value_t(),                   //
        ex::set_error_t(std::exception_ptr), //
        ex::set_stopped_t()>;

    template <typename CPO>
    friend my_scheduler_except tag_invoke(ex::get_completion_scheduler_t<CPO>, my_sender) {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, my_scheduler_except) {
    throw std::logic_error("err");
    return {};
  }

  friend bool operator==(my_scheduler_except, my_scheduler_except) noexcept { return true; }
  friend bool operator!=(my_scheduler_except, my_scheduler_except) noexcept { return false; }
};

TEST_CASE("type with schedule that throws is a scheduler", "[concepts][scheduler]") {
  REQUIRE(ex::scheduler<my_scheduler_except>);
}

struct noeq_sched {
  struct my_sender {
    using completion_signatures =
      ex::completion_signatures<             //
        ex::set_value_t(),                   //
        ex::set_error_t(std::exception_ptr), //
        ex::set_stopped_t()>;

    template <typename CPO>
    friend noeq_sched tag_invoke(ex::get_completion_scheduler_t<CPO>, my_sender) {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, noeq_sched) { return {}; }
};

TEST_CASE("type w/o equality operations do not model scheduler", "[concepts][scheduler]") {
  REQUIRE(!ex::scheduler<noeq_sched>);
}

struct sched_no_completion {
  struct my_sender {
    using completion_signatures =
      ex::completion_signatures<             //
        ex::set_value_t(),                   //
        ex::set_error_t(std::exception_ptr), //
        ex::set_stopped_t()>;

    friend sched_no_completion tag_invoke(
        ex::get_completion_scheduler_t<ex::set_error_t>, my_sender) {
      return {};
    }
  };

  friend my_sender tag_invoke(ex::schedule_t, sched_no_completion) { return {}; }

  friend bool operator==(sched_no_completion, sched_no_completion) noexcept { return true; }
  friend bool operator!=(sched_no_completion, sched_no_completion) noexcept { return false; }
};

TEST_CASE(
    "not a scheduler if the returned sender doesn't have get_completion_scheduler of set_value",
    "[concepts][scheduler]") {
  REQUIRE(!ex::scheduler<noeq_sched>);
}
