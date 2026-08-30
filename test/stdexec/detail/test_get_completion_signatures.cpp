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

#include <catch2/catch_all.hpp>

#include <stdexec/execution.hpp>

#if !STDEXEC_NO_STDCPP_REFLECTION()
#  include <meta>
#endif

namespace ex = STDEXEC;

namespace
{
  using independent_completions = ex::completion_signatures<ex::set_value_t(int)>;
  using dependent_completions   = ex::completion_signatures<ex::set_value_t(double)>;

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
        return dependent_completions{};
      }
    }
  };

#if !STDEXEC_NO_STDCPP_REFLECTION()
  template <class Sender, class... Env>
  consteval bool completion_signature_protocols_agree()
  {
    return std::meta::is_same_type(ex::get_completion_signatures_type<Sender, Env...>(),
                                   ^^decltype(ex::get_completion_signatures<Sender, Env...>()));
  }

  template <class Sender, class... Env>
  consteval bool get_completion_signatures_type_throws() noexcept
  try
  {
    (void) ex::get_completion_signatures_type<Sender, Env...>();
    return false;
  }
  catch (...)
  {
    return true;
  }

  struct throwing_legacy_test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class...>
    static consteval auto get_completion_signatures()
    {
      throw std::meta::exception("Unable to compute completion signatures.",
                                 ^^throwing_legacy_test_sender);
      return ex::completion_signatures<>{};
    }
  };

  struct reflection_test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class... Env>
    static consteval std::meta::info get_completion_signatures_type() noexcept
    {
      if constexpr (sizeof...(Env) == 0)
      {
        return ^^independent_completions;
      }
      else
      {
        return ^^dependent_completions;
      }
    }
  };

  struct transformed_reflection_test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class>
    static consteval std::meta::info get_completion_signatures_type() noexcept
    {
      return ^^dependent_completions;
    }
  };

  struct throwing_reflection_test_sender
  {
    using sender_concept = ex::sender_tag;

    template <class, class...>
    static consteval std::meta::info get_completion_signatures_type()
    {
      throw std::meta::exception("Unable to compute completion signatures.",
                                 ^^throwing_reflection_test_sender);
    }
  };

  struct reflection_test_domain
  {
    template <class Env>
    auto transform_sender(ex::start_t, reflection_test_sender, Env const &) const
      -> transformed_reflection_test_sender
    {
      return {};
    }
  };

  using reflection_test_env = ex::prop<ex::get_domain_t, reflection_test_domain>;

  struct throwing_reflection_test_domain
  {
    template <class Env>
    auto transform_sender(ex::start_t, reflection_test_sender, Env const &) const
      -> throwing_reflection_test_sender
    {
      return {};
    }
  };

  using throwing_reflection_test_env = ex::prop<ex::get_domain_t, throwing_reflection_test_domain>;
#endif

  TEST_CASE("get_completion_signatures queries a sender without an environment",
            "[detail][get_completion_signatures]")
  {
    STATIC_REQUIRE(std::same_as<decltype(ex::get_completion_signatures<test_sender>()),
                                independent_completions>);
#if !STDEXEC_NO_STDCPP_REFLECTION()
    STATIC_REQUIRE(completion_signature_protocols_agree<test_sender>());
#endif
  }

  TEST_CASE("get_completion_signatures queries a sender in an environment",
            "[detail][get_completion_signatures]")
  {
    STATIC_REQUIRE(std::same_as<decltype(ex::get_completion_signatures<test_sender, ex::env<>>()),
                                dependent_completions>);
#if !STDEXEC_NO_STDCPP_REFLECTION()
    STATIC_REQUIRE(completion_signature_protocols_agree<test_sender, ex::env<>>());
#endif
  }

  TEST_CASE("get_completion_signatures supports the legacy function-call interface",
            "[detail][get_completion_signatures]")
  {
    STATIC_REQUIRE(std::same_as<decltype(ex::get_completion_signatures(test_sender{}, ex::env<>{})),
                                dependent_completions>);
#if !STDEXEC_NO_STDCPP_REFLECTION()
    STATIC_REQUIRE(completion_signature_protocols_agree<test_sender, ex::env<>>());
#endif
  }

#if !STDEXEC_NO_STDCPP_REFLECTION()
  TEST_CASE("get_completion_signatures_type reflects the legacy query result",
            "[detail][get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(std::meta::is_same_type(ex::get_completion_signatures_type<test_sender>(),
                                           ^^independent_completions));
    STATIC_REQUIRE(
      std::meta::is_same_type(ex::get_completion_signatures_type<test_sender, ex::env<>>(),
                              ^^dependent_completions));
    STATIC_REQUIRE(completion_signature_protocols_agree<test_sender>());
    STATIC_REQUIRE(completion_signature_protocols_agree<test_sender, ex::env<>>());
  }

  TEST_CASE("get_completion_signatures_type invokes the legacy query",
            "[detail][get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(get_completion_signatures_type_throws<throwing_legacy_test_sender>());
    STATIC_REQUIRE(get_completion_signatures_type_throws<throwing_legacy_test_sender, ex::env<>>());
  }

  TEST_CASE("detect whether get_completion_signatures_type throws",
            "[detail][get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(
      ex::__cmplsigs::__nothrow_get_completion_signatures_type<reflection_test_sender>());
    STATIC_REQUIRE_FALSE(
      ex::__cmplsigs::__nothrow_get_completion_signatures_type<throwing_reflection_test_sender>());
    STATIC_REQUIRE_FALSE(
      ex::__cmplsigs::__nothrow_get_completion_signatures_type<reflection_test_sender,
                                                               throwing_reflection_test_env>());
    STATIC_REQUIRE(
      std::same_as<decltype(ex::get_completion_signatures<throwing_reflection_test_sender>()),
                   ex::completion_signatures<>>);
    STATIC_REQUIRE(
      std::same_as<decltype(ex::get_completion_signatures<reflection_test_sender,
                                                          throwing_reflection_test_env>()),
                   ex::completion_signatures<>>);
  }

  TEST_CASE("get_completion_signatures_type queries a reflection-native sender without an "
            "environment",
            "[detail][get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(
      std::meta::is_same_type(ex::get_completion_signatures_type<reflection_test_sender>(),
                              ^^independent_completions));
    STATIC_REQUIRE(completion_signature_protocols_agree<reflection_test_sender>());
  }

  TEST_CASE("get_completion_signatures_type queries a reflection-native sender in an environment",
            "[detail][get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(std::meta::is_same_type(
      ex::get_completion_signatures_type<reflection_test_sender, ex::env<>>(),
      ^^dependent_completions));
    STATIC_REQUIRE(completion_signature_protocols_agree<reflection_test_sender, ex::env<>>());
  }

  TEST_CASE("get_completion_signatures_type queries the transformed sender",
            "[detail][get_completion_signatures_type][reflection]")
  {
    STATIC_REQUIRE(std::meta::is_same_type(
      ex::get_completion_signatures_type<reflection_test_sender, reflection_test_env>(),
      ^^dependent_completions));
    STATIC_REQUIRE(
      completion_signature_protocols_agree<reflection_test_sender, reflection_test_env>());
  }
#endif
}  // namespace
