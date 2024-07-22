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
#include <stdexec/__detail/__meta.hpp>

#include <optional>

using namespace std;
using namespace stdexec;

namespace {

  TEST_CASE(
    "transform identity will return the given types (wrapped in __types)",
    "[detail][transform]") {
    using tr = __mtransform<__q<__midentity>>;
    using res = __minvoke<tr, int, char>;
    static_assert(is_same_v<res, __types<int, char>>);
  }

  TEST_CASE(
    "transform can avoid the __types wrapping with __q<__midentity>",
    "[detail][transform]") {
    using tr = __mtransform<__q<__midentity>, __q<__midentity>>;
    using res = __minvoke<tr, int>;
    static_assert(is_same_v<res, int>);
  }

  template <typename T>
  using as_optional = std::optional<T>;

  TEST_CASE("transform can wrap input types", "[detail][transform]") {
    using tr = __mtransform<__q<as_optional>>;
    using res = __minvoke<tr, int, char>;
    static_assert(is_same_v<res, __types<optional<int>, optional<char>>>);
  }

  TEST_CASE(
    "transform continuation can be used to wrap the result in another template",
    "[detail][transform]") {
    using tr = __mtransform<__q<as_optional>, __q<tuple>>;
    using res = __minvoke<tr, int, char>;
    static_assert(is_same_v<res, tuple<optional<int>, optional<char>>>);
  }
} // namespace
