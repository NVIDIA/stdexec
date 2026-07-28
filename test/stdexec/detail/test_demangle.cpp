/*
 * Copyright (c) 2025 NVIDIA Corporation
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

namespace
{
  TEST_CASE("demangling a type", "[detail][demangle]")
  {
    struct Dummy
    {
      void operator()(int const) const {}
    };
    auto sndr = STDEXEC::just(42) | STDEXEC::then(Dummy{});

    static_assert(
      std::same_as<STDEXEC::__demangle_t<decltype(sndr)>,
                   STDEXEC::__basic_sender<
                     STDEXEC::then_t,
                     Dummy,
                     STDEXEC::__basic_sender<STDEXEC::just_t, STDEXEC::__tuple<int>>::type>::type>);
  }
}  // namespace
