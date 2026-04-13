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

#include <catch2/catch_all.hpp>
#include <stdexec/execution.hpp>

#include <test_common/receivers.hpp>

namespace ex = STDEXEC;

namespace
{
  struct gpu_domain
  {};

  struct thread_pool_domain : ex::default_domain
  {};

  struct parallel_runtime_domain : ex::default_domain
  {};

  template <class... Domains>
  using common_domain_t = ex::__common_domain_t<Domains...>;

  TEST_CASE("finding common domains", "[detail][domain]")
  {
    using common1 = common_domain_t<gpu_domain, gpu_domain>;
    STATIC_REQUIRE(std::same_as<common1, gpu_domain>);

    using common2 = common_domain_t<thread_pool_domain, parallel_runtime_domain>;
    STATIC_REQUIRE(std::same_as<common2, ex::default_domain>);

    using common3 = common_domain_t<gpu_domain, thread_pool_domain>;
    STATIC_REQUIRE(
      std::same_as<common3, ex::indeterminate_domain<gpu_domain, thread_pool_domain>>
      || std::same_as<common3, ex::indeterminate_domain<thread_pool_domain, gpu_domain>>);
  }
}  // namespace
