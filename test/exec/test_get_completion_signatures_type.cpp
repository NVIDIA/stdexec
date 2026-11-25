/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <exec/completion_signatures.hpp>
#include <stdexec/execution.hpp>

#include <catch2/catch_all.hpp>

#if !STDEXEC_NO_STDCPP_REFLECTION()
#  include <meta>

namespace ex = STDEXEC;

namespace
{
  using independent_completions = ex::completion_signatures<ex::set_value_t(int)>;
  using dependent_completions   = ex::completion_signatures<ex::set_value_t(double)>;
  using aliased_completions     = dependent_completions;

  struct test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class... Env>
    static consteval auto get_completion_signatures() noexcept
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return independent_completions{};
      }
      else
      {
        return aliased_completions{};
      }
    }
  };

  struct reflection_test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class... Env>
    static consteval std::meta::info get_completion_signatures() noexcept
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return ^^independent_completions;
      }
      else
      {
        return ^^aliased_completions;
      }
    }
  };

  struct throwing_test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class...>
    static consteval auto get_completion_signatures()
    {
      throw std::meta::exception("Unable to compute completion signatures.",
                                 ^^throwing_test_sender);
      return ex::completion_signatures<>{};
    }
  };

  struct move_only_test_sender
  {
    using sender_concept        = ex::sender_tag;
    using completion_signatures = independent_completions;

    move_only_test_sender()                             = default;
    move_only_test_sender(move_only_test_sender&&)      = default;
    move_only_test_sender(move_only_test_sender const&) = delete;
  };

  template <class Sender, class... Env>
  consteval bool get_completion_signatures_type_throws() noexcept
  try
  {
    (void) ::exec::get_completion_signatures_type(^^Sender, {^^Env...});
    return false;
  }
  catch (...)
  {
    return true;
  }

  TEST_CASE("get_completion_signatures_type reflects completion_signatures_of_t",
            "[get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(std::meta::is_same_type(::exec::get_completion_signatures_type(^^test_sender),
                                           ^^independent_completions));
    STATIC_REQUIRE(
      std::meta::is_same_type(::exec::get_completion_signatures_type(^^test_sender, {^^ex::env<>}),
                              ^^ex::completion_signatures_of_t<test_sender, ex::env<>>));
  }

  TEST_CASE("get_completion_signatures_type removes aliases",
            "[get_completion_signatures_type][reflection]")
  {
    constexpr auto expected = std::meta::dealias(^^aliased_completions);
    STATIC_REQUIRE(std::meta::is_type_alias(^^aliased_completions));
    STATIC_REQUIRE(::exec::get_completion_signatures_type(^^test_sender, {^^ex::env<>}) == expected);
    STATIC_REQUIRE(::exec::get_completion_signatures_type(^^reflection_test_sender, {^^ex::env<>})
                   == expected);
  }

  TEST_CASE("get_completion_signatures_type accepts any reflection range",
            "[get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(
      [] consteval
      {
        std::vector<std::meta::info> const env{^^ex::env<>};
        return std::meta::is_same_type(
          ::exec::get_completion_signatures_type(^^reflection_test_sender, env),
          ^^dependent_completions);
      }());
  }

  TEST_CASE("get_completion_signatures_type propagates exceptions",
            "[get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE_FALSE(get_completion_signatures_type_throws<test_sender>());
    STATIC_REQUIRE(get_completion_signatures_type_throws<throwing_test_sender>());
    STATIC_REQUIRE(get_completion_signatures_type_throws<throwing_test_sender, ex::env<>>());
  }

  TEST_CASE("get_completion_signatures_type throws for a type which is not a sender",
            "[get_completion_signatures_type][reflection]")
  {
    // An lvalue of a move-only sender cannot be decay-copied, so it is not a
    // sender and cannot be connected ([exec.connect]).
    STATIC_REQUIRE(ex::sender<move_only_test_sender>);
    STATIC_REQUIRE_FALSE(ex::sender<move_only_test_sender&>);
    STATIC_REQUIRE_FALSE(get_completion_signatures_type_throws<move_only_test_sender>());
    STATIC_REQUIRE(get_completion_signatures_type_throws<move_only_test_sender&>());
  }
}  // namespace
#endif
