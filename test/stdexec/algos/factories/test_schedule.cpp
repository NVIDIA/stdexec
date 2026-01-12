/*
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

#include "exec/static_thread_pool.hpp"
#include "stdexec/__detail/__meta.hpp"
#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/type_helpers.hpp>
#include <utility>

namespace ex = STDEXEC;

namespace {
  template <ex::scheduler Sched = inline_scheduler>
  inline auto _with_scheduler(Sched sched = {}) {
    return ex::write_env(ex::prop{ex::get_scheduler, std::move(sched)});
  }

  TEST_CASE("schedule returns a sender", "[factories][schedule]") {
    using sndr = ex::schedule_result_t<inline_scheduler>;
    static_assert(ex::sender<sndr>);
  }

  TEST_CASE("schedule advertices scheduler and domain", "[factories][schedule]") {
    using target_sched = exec::static_thread_pool::scheduler;
    using target_domain =
      ex::__call_result_t<ex::get_completion_domain_t<ex::set_value_t>, target_sched, ex::env<>>;
    using pool_sndr = ex::schedule_result_t<target_sched>;
    using pool_attr = ex::env_of_t<pool_sndr>;
    using pool_scheduler =
      ex::__call_result_t<ex::get_completion_scheduler_t<ex::set_value_t>, pool_attr>;
    using pool_domain =
      ex::__call_result_t<ex::get_completion_domain_t<ex::set_value_t>, pool_attr>;
    STATIC_REQUIRE(std::is_same_v<pool_scheduler, target_sched>);
    STATIC_REQUIRE(std::is_same_v<pool_domain, target_domain>);

    using inline_env = decltype(ex::write_env(ex::prop{ex::get_scheduler, inline_scheduler{}}));
    using scheduler_with_env =
      ex::__call_result_t<ex::get_completion_scheduler_t<ex::set_value_t>, pool_attr, inline_env>;
    using domain_with_env =
      ex::__call_result_t<ex::get_completion_domain_t<ex::set_value_t>, pool_attr, inline_env>;
    STATIC_REQUIRE(std::is_same_v<scheduler_with_env, target_sched>);
    STATIC_REQUIRE(std::is_same_v<domain_with_env, target_domain>);

    using pool_env = ex::prop<ex::get_scheduler_t, target_sched>;
    using scheduler_with_pool_env =
      ex::__call_result_t<ex::get_completion_scheduler_t<ex::set_value_t>, pool_attr, pool_env>;
    using domain_with_pool_env =
      ex::__call_result_t<ex::get_completion_domain_t<ex::set_value_t>, pool_attr, pool_env>;
    STATIC_REQUIRE(std::is_same_v<scheduler_with_pool_env, target_sched>);
    STATIC_REQUIRE(std::is_same_v<domain_with_pool_env, target_domain>);

    using inline_sndr = ex::schedule_result_t<inline_scheduler>;
    using inline_attr = ex::env_of_t<inline_sndr>;

    using scheduler_with_inline_sender =
      ex::__call_result_t<ex::get_completion_scheduler_t<ex::set_value_t>, inline_attr, pool_env>;
    using domain_with_inline_sender =
      ex::__call_result_t<ex::get_completion_domain_t<ex::set_value_t>, inline_attr, pool_env>;
    STATIC_REQUIRE(std::is_same_v<scheduler_with_inline_sender, target_sched>);
    STATIC_REQUIRE(std::is_same_v<domain_with_inline_sender, target_domain>);
  }
} // namespace
