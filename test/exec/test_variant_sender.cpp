/*
 * Copyright (c) Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <exec/variant_sender.hpp>

#include <catch2/catch.hpp>

using namespace STDEXEC;
using namespace exec;

namespace {

  template <class... Ts>
  struct overloaded : Ts... {
    using Ts::operator()...;
  };
  template <class... Ts>
  overloaded(Ts...) -> overloaded<Ts...>;

  using just_int_t = decltype(just(0));
  using just_void_t = decltype(just());

  TEST_CASE("variant_sender - default constructible", "[types][variant_sender]") {
    variant_sender<just_void_t, just_int_t> variant{just()};
    CHECK(variant.index() == 0);
  }

  TEST_CASE("variant_sender - using an overloaded then adaptor", "[types][variant_sender]") {
    variant_sender<just_void_t, just_int_t> variant = just();
    int index = -1;
    STATIC_REQUIRE(sender<variant_sender<just_void_t, just_int_t>>);
    sync_wait(variant | then([&index](auto... xs) { index = sizeof...(xs); }));
    CHECK(index == 0);

    variant.emplace<1>(just(42));
    auto [value] = sync_wait(
                     variant
                     | then(
                       overloaded{
                         [&index] {
                           index = 0;
                           return 0;
                         },
                         [&index](int xs) {
                           index = 1;
                           return xs;
                         }}))
                     .value();
    CHECK(index == 1);
    CHECK(value == 42);
  }
} // namespace
