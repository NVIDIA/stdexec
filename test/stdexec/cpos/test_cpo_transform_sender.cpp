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

#include <stdexec/execution.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ex = STDEXEC;

namespace
{
  inline constexpr auto return_zero = []() noexcept
  {
    return 0;
  };

  using then_sender_t = decltype(ex::just() | ex::then(return_zero));
  using just_sender_t = decltype(ex::just(42));

  //! A domain that transforms a then_sender_t into a just_sender_t, which it
  //! doesn't transform, so the result is transformed again, by default_domain.
  //! The transformation throws unless it's noexcept.
  template <bool Nothrow>
  struct then_to_just_domain
  {
    template <class Sndr, class Env>
      requires std::same_as<std::remove_cvref_t<Sndr>, then_sender_t>
    auto
    transform_sender(ex::start_t, Sndr &&, Env const &) const noexcept(Nothrow) -> just_sender_t
    {
      if constexpr (!Nothrow)
      {
        STDEXEC_THROW(std::runtime_error("transform_sender"));
      }
      return ex::just(42);
    }
  };

  template <bool Nothrow>
  struct receiver
  {
    using receiver_concept = ex::receiver_tag;

    void set_value(int) noexcept {}

    auto get_env() const noexcept
    {
      return ex::prop(ex::get_domain, then_to_just_domain<Nothrow>{});
    }
  };

  TEST_CASE("transform_sender is noexcept only if every transformation is",
            "[cpo][cpo_transform_sender]")
  {
    SECTION("a potentially-throwing transformation whose result is transformed again")
    {
      auto env = ex::prop(ex::get_domain, then_to_just_domain<false>{});
      STATIC_REQUIRE(
        std::same_as<decltype(ex::transform_sender(std::declval<then_sender_t>(), env)),
                     just_sender_t>);
      STATIC_REQUIRE(!noexcept(ex::transform_sender(std::declval<then_sender_t>(), env)));
      STATIC_REQUIRE(
        !noexcept(ex::connect(std::declval<then_sender_t>(), std::declval<receiver<false>>())));
    }

    SECTION("non-throwing transformations")
    {
      auto env = ex::prop(ex::get_domain, then_to_just_domain<true>{});
      STATIC_REQUIRE(noexcept(ex::transform_sender(std::declval<then_sender_t>(), env)));
      STATIC_REQUIRE(
        noexcept(ex::connect(std::declval<then_sender_t>(), std::declval<receiver<true>>())));
    }
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("an exception thrown by a transformation propagates out of connect",
            "[cpo][cpo_transform_sender]")
  {
    auto sndr = ex::just() | ex::then(return_zero);
    REQUIRE_THROWS_AS(ex::connect(std::move(sndr), receiver<false>{}), std::runtime_error);
  }
#endif
}  // namespace
