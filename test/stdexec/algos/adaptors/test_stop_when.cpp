/*
 * Copyright (c) 2025 Ian Petersen
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

#include <catch2/catch.hpp>
#include <stdexec/execution.hpp>
#include <stdexec/stop_token.hpp>
#include <stdexec/__detail/__stop_when.hpp>
#include <test_common/receivers.hpp>

namespace ex = STDEXEC;

namespace {
  TEST_CASE("stop-when of an unstoppable token is the identity", "[adaptors][stop-when]") {
    auto snd = ex::just(42);
    auto checkIdentity = [](auto&& snd) {
      auto&& result = ex::__stop_when(std::forward<decltype(snd)>(snd), ex::never_stop_token{});

      REQUIRE(&snd == &result);
    };

    checkIdentity(snd);
    checkIdentity(std::as_const(snd));
    checkIdentity(std::move(snd));
  }

  TEST_CASE("stop-when(just(), token) returns a sender", "[adaptors][stop-when]") {
    ex::inplace_stop_source source;
    auto snd = ex::__stop_when(ex::just(), source.get_token());
    STATIC_REQUIRE(ex::sender<decltype(snd)>);
    wait_for_value(snd);
  }

  auto isTokenStoppable() {
    return ex::read_env(ex::get_stop_token)
         | ex::then([](auto token) noexcept { return token.stop_possible(); });
  }

  TEST_CASE(
    "stop-when substitutes its token when the receiver's token is unstoppable",
    "[adaptors][stop-when]") {

    // check that, by default, wait_for_value provides an unstoppable stop token
    wait_for_value(isTokenStoppable(), false);

    // now, check that stop-when mixes in a stoppable token

    ex::inplace_stop_source source;

    REQUIRE(source.get_token().stop_possible());

    wait_for_value(ex::__stop_when(isTokenStoppable(), source.get_token()), true);
  }

  TEST_CASE(
    "stop-when fuses its token with the receiver's when both are stoppable",
    "[adaptors][stop-when]") {
    ex::inplace_stop_source source;
    wait_for_value(
      ex::__stop_when(ex::__stop_when(isTokenStoppable(), source.get_token()), source.get_token()),
      true);
  }

  template <std::invocable Fn>
  ex::sender auto make_stop_callback(Fn&& fn) noexcept {
    return ex::read_env(ex::get_stop_token)
         | ex::then([fn = std::forward<Fn>(fn)](auto token) mutable noexcept {
             using cb_t = decltype(token)::template callback_type<std::remove_cvref_t<Fn>>;
             return std::optional<cb_t>(std::in_place, std::move(token), std::move(fn));
           });
  }

  TEST_CASE("callbacks registered with stop-when's token can be invoked", "[adaptors][stop-when]") {
    int invokeCount = 0;
    auto snd = make_stop_callback([&invokeCount]() noexcept { invokeCount++; });

    {
      ex::inplace_stop_source source;
      wait_for_value(
        snd | ex::then([&](auto&& optCallback) noexcept {
          source.request_stop();
          optCallback.reset();
          return invokeCount;
        }),
        0);
    }

    {
      ex::inplace_stop_source source;

      wait_for_value(
        ex::__stop_when(snd, source.get_token()) | ex::then([&](auto&& optCallback) noexcept {
          source.request_stop();
          optCallback.reset();
          return invokeCount;
        }),
        1);
    }

    {
      ex::inplace_stop_source source1;
      ex::inplace_stop_source source2;

      wait_for_value(
        ex::__stop_when(snd, source2.get_token()) | ex::then([&](auto&& optCallback) noexcept {
          source1.request_stop();
          source2.request_stop();
          optCallback.reset();
          return invokeCount;
        }) | ex::write_env(ex::prop(ex::get_stop_token, source1.get_token())),
        2);
    }
  }
} // namespace
