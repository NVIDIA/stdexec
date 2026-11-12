/*
 * Copyright (c) 2026 Ian Petersen
 * Copyright (c) 2026 NVIDIA Corporation
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

#include <stdexec/__detail/__config.hpp>

#if STDEXEC_USE_MODULES()
import std;
import stdexec;
#else
#  include <stdexec/__detail/__tuple.hpp>

#  include <array>
#  include <string>
#  include <type_traits>
#  include <utility>
#endif

using namespace STDEXEC;

namespace
{
  //! Reports, for each argument, whether it was passed as an rvalue
  struct rvalue_args
  {
    template <class... Ts>
    constexpr auto operator()(Ts &&...) const noexcept -> std::array<bool, sizeof...(Ts)>
    {
      return {!std::is_lvalue_reference_v<Ts>...};
    }
  };

  // more than eight elements, so the tuple stores its elements in base classes
  using large_tuple = __tuple<int, int, int, int, int, int, int, int, int &, int &&>;

  TEST_CASE("__apply passes each element of a tuple as std::get would", "[detail][tuple]")
  {
    int i = 0;
    int j = 0;

    SECTION("a moved tuple passes values and rvalue references as rvalues, and lvalue "
            "references as lvalues")
    {
      __tuple<int, int &, int &&> tupl{0, i, std::move(j)};
      REQUIRE(__apply(rvalue_args{}, std::move(tupl)) == std::array{true, false, true});
    }

    SECTION("an lvalue tuple passes every element as an lvalue")
    {
      __tuple<int, int &, int &&> tupl{0, i, std::move(j)};
      REQUIRE(__apply(rvalue_args{}, tupl) == std::array{false, false, false});

      auto const &ctupl = tupl;
      REQUIRE(__apply(rvalue_args{}, ctupl) == std::array{false, false, false});
    }

    SECTION("the same holds for a tuple of more than eight elements")
    {
      large_tuple tupl{
        {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {i}, {std::move(j)}}
      };
      std::array<bool, 10> moved{true, true, true, true, true, true, true, true, false, true};
      REQUIRE(__apply(rvalue_args{}, std::move(tupl)) == moved);
      REQUIRE(__apply(rvalue_args{}, tupl) == std::array<bool, 10>{});
    }

    SECTION("a function taking an element by value moves from an rvalue reference")
    {
      std::string             str = "a string too long for the small-string optimization";
      __tuple<std::string &&> tupl{std::move(str)};
      auto                    taken = __apply([](std::string s) { return s; }, std::move(tupl));
      REQUIRE(taken == "a string too long for the small-string optimization");
      REQUIRE(str.empty());
    }
  }

  TEST_CASE("__get returns each element of a tuple as std::get would", "[detail][tuple]")
  {
    using small_tuple = __tuple<int, int &, int &&>;
    STATIC_REQUIRE(std::is_same_v<decltype(__get<0>(std::declval<small_tuple>())), int &&>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<1>(std::declval<small_tuple>())), int &>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<2>(std::declval<small_tuple>())), int &&>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<2>(std::declval<small_tuple &>())), int &>);
    STATIC_REQUIRE(
      std::is_same_v<decltype(__get<0>(std::declval<small_tuple const &>())), int const &>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<2>(std::declval<small_tuple const &>())), int &>);

    STATIC_REQUIRE(std::is_same_v<decltype(__get<0>(std::declval<large_tuple>())), int &&>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<8>(std::declval<large_tuple>())), int &>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<9>(std::declval<large_tuple>())), int &&>);
    STATIC_REQUIRE(std::is_same_v<decltype(__get<9>(std::declval<large_tuple &>())), int &>);
  }
}  // namespace
