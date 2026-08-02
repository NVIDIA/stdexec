/*
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "exec/finally.hpp"
#include "test_common/type_helpers.hpp"

#include <test_common/catch2.hpp>

#include <stdexcept>

using namespace STDEXEC;

namespace
{
  constinit int    global_int    = 0;
  constinit double global_double = 0.0;

  struct reference_sender
  {
    using sender_concept = sender_tag;

    template <class, class...>
    static consteval auto get_completion_signatures()
    {
      return completion_signatures<set_value_t(int&)>{};
    }

    template <class Receiver>
    struct operation
    {
      Receiver receiver_;
      int*     value_;

      void start() & noexcept
      {
        set_value(static_cast<Receiver&&>(receiver_), *value_);
      }
    };

    template <class Receiver>
    auto connect(Receiver receiver) && -> operation<Receiver>
    {
      return {static_cast<Receiver&&>(receiver), value_};
    }

    int* value_;
  };

  struct references_sender
  {
    using sender_concept = sender_tag;

    template <class, class...>
    static consteval auto get_completion_signatures()
    {
      return completion_signatures<set_value_t(int&, double&)>{};
    }

    template <class Receiver>
    struct operation
    {
      Receiver receiver_;
      int*     int_value_;
      double*  double_value_;

      void start() & noexcept
      {
        set_value(static_cast<Receiver&&>(receiver_), *int_value_, *double_value_);
      }
    };

    template <class Receiver>
    auto connect(Receiver receiver) && -> operation<Receiver>
    {
      return {static_cast<Receiver&&>(receiver), int_value_, double_value_};
    }

    int*    int_value_;
    double* double_value_;
  };

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  struct throws_on_move
  {
    throws_on_move()                       = default;
    throws_on_move(throws_on_move const &) = delete;
    throws_on_move(throws_on_move&&)
    {
      throw std::runtime_error("Throwing as requested");
    }
  };

  struct throwing_value_sender
  {
    using sender_concept = sender_tag;

    template <class, class...>
    static consteval auto get_completion_signatures()
    {
      return completion_signatures<set_value_t(throws_on_move)>{};
    }

    template <class Receiver>
    struct operation
    {
      Receiver        receiver_;
      throws_on_move* value_;

      void start() & noexcept
      {
        set_value(static_cast<Receiver&&>(receiver_), std::move(*value_));
      }
    };

    template <class Receiver>
    auto connect(Receiver receiver) && -> operation<Receiver>
    {
      return {static_cast<Receiver&&>(receiver), value_};
    }

    throws_on_move* value_;
  };
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  TEST_CASE("finally is a sender", "[adaptors][finally]")
  {
    auto s = exec::finally(just(), just());
    STATIC_REQUIRE(sender<decltype(s)>);
  }

  TEST_CASE("finally with pipe syntax is a sender", "[adaptors][finally]")
  {
    auto s = just() | exec::finally(just());
    STATIC_REQUIRE(sender<decltype(s)>);
  }

  TEST_CASE("finally is a sender in empty env", "[adaptors][finally]")
  {
    auto s = exec::finally(just(), just());
    STATIC_REQUIRE(sender_in<decltype(s), ex::env<>>);
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_value_t()>>);
  }

  TEST_CASE("finally executes the final action", "[adaptors][finally]")
  {
    bool called = false;
    auto s      = exec::finally(just(), just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_value_t()>>);
    sync_wait(s);
    CHECK(called);
  }

  TEST_CASE("finally executes the final action and returns integer", "[adaptors][finally]")
  {
    bool called = false;
    auto s      = exec::finally(just(42), just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_value_t(int)>>);
    auto [i] = *sync_wait(s);
    CHECK(called);
    CHECK(i == 42);
  }

  TEST_CASE("finally preserves a single reference completion", "[adaptors][finally]")
  {
    bool called = false;
    global_int  = 42;
    auto raw    = exec::finally(reference_sender{&global_int},
                             just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(raw), ex::env<>>,
                                  completion_signatures<set_value_t(int&)>>);
    auto s = std::move(raw)
           | then(
               [](auto&& i) noexcept
               {
                 CHECK(&i == &global_int);
                 CHECK(i == 42);
               });
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_value_t()>>);

    sync_wait(s);
    CHECK(called);
  }

  TEST_CASE("finally preserves multiple reference completion arguments", "[adaptors][finally]")
  {
    bool called   = false;
    global_int    = 42;
    global_double = 0.125;
    auto raw      = exec::finally(references_sender{&global_int, &global_double},
                             just() | then([&called]() noexcept { called = true; }));
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(raw), ex::env<>>,
                                  completion_signatures<set_value_t(int&, double&)>>);
    auto s = std::move(raw)
           | then(
               [](auto&& i, auto&& d) noexcept
               {
                 CHECK(&i == &global_int);
                 CHECK(&d == &global_double);
                 CHECK(i == 42);
                 CHECK(d == 0.125);
               });
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_value_t()>>);

    sync_wait(s);
    CHECK(called);
  }

#if !STDEXEC_NO_STDCPP_EXCEPTIONS()
  TEST_CASE("finally executes the final action when storing the initial completion throws",
            "[adaptors][finally]")
  {
    bool           called = false;
    throws_on_move value;

    auto s = exec::finally(throwing_value_sender{&value},
                           just() | then([&called]() noexcept { called = true; }));

    CHECK_THROWS_AS(sync_wait(s), std::runtime_error);
    CHECK(called);
  }

  TEST_CASE("finally does not execute the final action and throws integer", "[adaptors][finally]")
  {
    bool called = false;

    auto s = exec::finally(just(21) | then([](int) -> int { throw 42; }),
                           just() | then([&called]() noexcept(false) { called = true; }));
    STATIC_REQUIRE(
      set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                     completion_signatures<set_error_t(std::exception_ptr), set_value_t(int)>>);
    CHECK_THROWS_AS(sync_wait(s), int);
    CHECK(called);
  }
#endif  // !STDEXEC_NO_STDCPP_EXCEPTIONS()

  TEST_CASE("finally includes the error types of the final action", "[adaptors][finally]")
  {
    auto s = exec::finally(just(), just_error(42));
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_error_t(int)>>);
  }

  TEST_CASE("finally includes the stopped signal of the final action", "[adaptors][finally]")
  {
    auto s = exec::finally(just(), just_stopped());
    STATIC_REQUIRE(set_equivalent<completion_signatures_of_t<decltype(s), ex::env<>>,
                                  completion_signatures<set_stopped_t()>>);
  }
}  // namespace
