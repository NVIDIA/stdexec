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
#include <stdexec/execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = stdexec;

namespace {

  TEST_CASE("upon_error returns a sender", "[adaptors][upon_error]") {
    auto snd = ex::upon_error(ex::just_error(std::exception_ptr{}), [](std::exception_ptr) { });
    static_assert(ex::sender<decltype(snd)>);
    (void) snd;
  }

  TEST_CASE("upon_error with environment returns a sender", "[adaptors][upon_error]") {
    auto snd = ex::upon_error(ex::just_error(std::exception_ptr{}), [](std::exception_ptr) { });
    static_assert(ex::sender_in<decltype(snd), ex::env<>>);
    (void) snd;
  }

  TEST_CASE("upon_error simple example", "[adaptors][upon_error]") {
    bool called{false};
    auto snd = ex::upon_error(ex::just_error(std::exception_ptr{}), [&](std::exception_ptr) {
      called = true;
      return 0;
    });
    auto op = ex::connect(std::move(snd), expect_value_receiver{0});
    ex::start(op);
    // The receiver checks that it's called
    // we also check that the function was invoked
    CHECK(called);
  }

  TEST_CASE("upon_error with no-error input sender", "[adaptors][upon_error]") {
    auto snd = ex::upon_error(ex::just(), []() -> double { return 0.0; });
    static_assert(ex::sender<decltype(snd)>);

    using S = decltype(snd);
    static_assert(ex::sender<S>);
    using completion_sigs = decltype(ex::get_completion_signatures(snd, ex::env<>{}));
    static_assert(ex::__mset_eq<ex::__mset<ex::set_value_t()>, completion_sigs>);
  }

  template <typename R>
  struct oper : immovable {
    R recv_;

    void start() & noexcept {
      ex::set_value(static_cast<R&&>(recv_), 0);
    }
  };

  struct Error1 { };

  struct Error2 { };

  struct Error3 { };

  struct Error4 { };

  template <class... AdditionalCompletions>
  struct many_error_sender {
    using sender_concept = stdexec::sender_t;
    using completion_signatures = ex::completion_signatures<
      AdditionalCompletions...,
      ex::set_error_t(Error1),
      ex::set_error_t(Error2),
      ex::set_error_t(Error3)
    >;

    template <typename R>
    friend auto tag_invoke(ex::connect_t, many_error_sender, R&& r) -> oper<R> {
      return {{}, static_cast<R&&>(r)};
    }
  };

  TEST_CASE("upon_error many input error types", "[adaptors][upon_error]") {
    {
      auto s = many_error_sender<>{} | ex::upon_error([](auto e) {
                 if constexpr (std::same_as<decltype(e), Error3>) {
                   return Error4{};
                 } else {
                   return e;
                 }
                 STDEXEC_UNREACHABLE();
               });

      using S = decltype(s);
      static_assert(ex::sender<S>);
      using completion_sigs = decltype(ex::get_completion_signatures(s, ex::env<>{}));
      static_assert(ex::__mset_eq<
                    ex::__mset<
                      ex::set_error_t(std::exception_ptr),
                      ex::set_value_t(Error1),
                      ex::set_value_t(Error2),
                      ex::set_value_t(Error4)
                    >,
                    completion_sigs
      >);
    }

    {
      auto s = many_error_sender<ex::set_value_t(int)>{} | ex::upon_error([](auto) { return 0; });

      using S = decltype(s);
      static_assert(ex::sender<S>);
      using completion_sigs = decltype(ex::get_completion_signatures(s, ex::env<>{}));
      static_assert(ex::__mset_eq<
                    ex::__mset<ex::set_error_t(std::exception_ptr), ex::set_value_t(int)>,
                    completion_sigs
      >);
    }

    {
      auto s = many_error_sender<ex::set_value_t(double)>{}
             | ex::upon_error([](auto) { return 0; });

      using S = decltype(s);
      static_assert(ex::sender<S>);
      using completion_sigs = decltype(ex::get_completion_signatures(s, ex::env<>{}));
      static_assert(ex::__mset_eq<
                    ex::__mset<
                      ex::set_error_t(std::exception_ptr),
                      ex::set_value_t(double),
                      ex::set_value_t(int)
                    >,
                    completion_sigs
      >);
    }
  }
} // namespace
