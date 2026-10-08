/*
 * Copyright (c) 2026 stdexec contributors
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

#include <test_common/catch2.hpp>
#include <stdexec/execution.hpp>
#include <exec/detail/xorshift.hpp>

#include <cstdint>

TEST_CASE("xorshift does not get stuck with a zero seed", "[xorshift]")
{
  exec::xorshift zero_seed{std::uint64_t{0}};
  exec::xorshift default_seed{};

  bool produced_nonzero = false;
  for (int i = 0; i < 8; ++i)
  {
    auto const sample = zero_seed();
    CHECK(sample == default_seed());
    produced_nonzero = produced_nonzero || sample != 0;
  }
  CHECK(produced_nonzero);
}
