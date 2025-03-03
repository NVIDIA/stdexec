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
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;

namespace {

  TEST_CASE("read returns empty env", "[factories][read]") {
    auto sndr = ex::read_env(ex::get_allocator);
    using Sndr = decltype(sndr);
    static_assert(ex::sender<Sndr>);
    static_assert(!ex::sender_in<Sndr>);
    static_assert(ex::__is_scheduler_affine<Sndr>);
  }
} // namespace
