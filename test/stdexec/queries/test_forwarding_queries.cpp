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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

struct my_derived_forwarding_query_t : ex::forwarding_query_t { };
inline constexpr my_derived_forwarding_query_t my_derived_forwarding_query{};

struct my_non_forwarding_query_t { };
inline constexpr my_non_forwarding_query_t my_non_forwarding_query{};

TEST_CASE("exec.queries are forwarding queries", "[exec.queries][forwarding_queries]") {
  static_assert(ex::forwarding_query(ex::get_allocator));
  static_assert(ex::forwarding_query(ex::get_stop_token));
  static_assert(ex::forwarding_query(ex::get_scheduler));
  static_assert(ex::forwarding_query(ex::get_delegatee_scheduler));
  static_assert(ex::forwarding_query(ex::get_completion_scheduler<ex::set_value_t>));
  static_assert(ex::forwarding_query(ex::get_completion_scheduler<ex::set_error_t>));
  static_assert(ex::forwarding_query(ex::get_completion_scheduler<ex::set_stopped_t>));

  static_assert(ex::forwarding_query(my_derived_forwarding_query));
  static_assert(!ex::forwarding_query(my_non_forwarding_query));
}
