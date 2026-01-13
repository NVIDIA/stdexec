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
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-function")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunneeded-internal-declaration")

namespace {
  struct not_a_sender { };

  TEST_CASE("Sender concept rejects non-sender types", "[concepts][sender]") {
    STATIC_REQUIRE(!ex::sender<int>);
    STATIC_REQUIRE(!ex::sender<not_a_sender>);
  }

  struct P2300r7_sender_1 {
    using sender_concept = STDEXEC::sender_t;
  };

  struct P2300r7_sender_2 { };
} // namespace

template <>
inline constexpr bool STDEXEC::enable_sender<P2300r7_sender_2> = true;

namespace {

  TEST_CASE("Sender concept accepts P2300R7-style senders", "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender<P2300r7_sender_1>);
    STATIC_REQUIRE(ex::sender<P2300r7_sender_2>);
  }

#if !STDEXEC_NO_STD_COROUTINES()
  struct an_awaiter {
    auto await_ready() -> bool;
    void await_suspend(STDEXEC::__std::coroutine_handle<>);
    void await_resume();
  };

  struct an_awaitable {
    friend auto operator co_await(an_awaitable) -> an_awaiter {
      return {};
    }
  };

  struct has_as_awaitable {
    template <class Promise>
    auto as_awaitable(Promise&) const -> an_awaitable {
      return {};
    }
  };

  TEST_CASE("Sender concept accepts awaiters and awaitables", "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender<an_awaiter>);
    STATIC_REQUIRE(ex::sender<an_awaitable>);
    STATIC_REQUIRE(ex::sender<has_as_awaitable>);
  }
#endif

  struct oper {
    oper() = default;
    oper(oper&&) = delete;

    void start() & noexcept {
    }
  };

  struct my_sender0 {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()
    >;

    auto connect(empty_recv::recv0&&) const -> oper {
      return {};
    }
  };

  struct void_sender {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;

    template <class Receiver>
    auto connect(Receiver) -> oper {
      return {};
    }
  };

  struct invalid_receiver {
    using receiver_concept = STDEXEC::receiver_t;

    template <class... As>
    void set_value(As&&...) noexcept {
      static_assert(sizeof...(As) == ~size_t(0)); // hard error always
    }
  };

  TEST_CASE("type w/ proper types, is a sender", "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender<my_sender0>);
    STATIC_REQUIRE(ex::sender_in<my_sender0, ex::env<>>);

    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_value_t()>);
    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_error_t(std::exception_ptr)>);
    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_stopped_t()>);
    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_value_t(), ex::env<>>);
    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_error_t(std::exception_ptr), ex::env<>>);
    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_stopped_t(), ex::env<>>);
  }

  TEST_CASE(
    "sender_to concept does not instantiate the receiver's completion functions",
    "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender_to<void_sender, invalid_receiver>);
  }

  TEST_CASE(
    "sender that accepts a void sender models sender_to the given sender",
    "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender_to<my_sender0, empty_recv::recv0>);
  }

  struct my_sender_int {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(int),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()
    >;

    auto connect(empty_recv::recv_int&&) const -> oper {
      return {};
    }
  };

  TEST_CASE("my_sender_int is a sender", "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender<my_sender_int>);
    STATIC_REQUIRE(ex::sender_in<my_sender_int, ex::env<>>);
    STATIC_REQUIRE(ex::sender_of<my_sender_int, ex::set_value_t(int)>);
    STATIC_REQUIRE(ex::sender_of<my_sender_int, ex::set_value_t(int), ex::env<>>);
  }

  TEST_CASE(
    "sender that accepts an int receiver models sender_to the given receiver",
    "[concepts][sender]") {
    STATIC_REQUIRE(ex::sender_to<my_sender_int, empty_recv::recv_int>);
  }

  TEST_CASE(
    "not all combinations of senders & receivers satisfy the sender_to concept",
    "[concepts][sender]") {
    REQUIRE_FALSE(ex::sender_to<my_sender0, empty_recv::recv_int>);
    REQUIRE_FALSE(ex::sender_to<my_sender0, empty_recv::recv0_ec>);
    REQUIRE_FALSE(ex::sender_to<my_sender0, empty_recv::recv_int_ec>);
    REQUIRE_FALSE(ex::sender_to<my_sender_int, empty_recv::recv0>);
    REQUIRE_FALSE(ex::sender_to<my_sender_int, empty_recv::recv0_ec>);
    REQUIRE_FALSE(ex::sender_to<my_sender_int, empty_recv::recv_int_ec>);
  }

  TEST_CASE(
    "can query completion signatures for a typed sender that sends nothing",
    "[concepts][sender]") {
    check_val_types<ex::__mset<pack<>>>(my_sender0{});
    check_err_types<ex::__mset<std::exception_ptr>>(my_sender0{});
    check_sends_stopped<true>(my_sender0{});
    STATIC_REQUIRE(ex::sender_of<my_sender0, ex::set_value_t()>);
  }

  TEST_CASE(
    "can query completion signatures for a typed sender that sends int",
    "[concepts][sender]") {
    check_val_types<ex::__mset<pack<int>>>(my_sender_int{});
    check_err_types<ex::__mset<std::exception_ptr>>(my_sender_int{});
    check_sends_stopped<true>(my_sender_int{});
    STATIC_REQUIRE(ex::sender_of<my_sender_int, ex::set_value_t(int)>);
  }

  struct multival_sender {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(int, double),
      ex::set_value_t(short, long),
      ex::set_error_t(std::exception_ptr)
    >;

    auto connect(empty_recv::recv_int&&) const -> oper {
      return {};
    }
  };

  TEST_CASE(
    "check completion signatures for sender that advertises multiple sets of values",
    "[concepts][sender]") {
    check_val_types<ex::__mset<pack<int, double>, pack<short, long>>>(multival_sender{});
    check_err_types<ex::__mset<std::exception_ptr>>(multival_sender{});
    check_sends_stopped<false>(multival_sender{});
    REQUIRE_FALSE(ex::sender_of<multival_sender, ex::set_value_t(int, double)>);
  }

  struct ec_sender {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(),
      ex::set_error_t(std::exception_ptr),
      ex::set_error_t(int)
    >;

    auto connect(empty_recv::recv_int&&) const -> oper {
      return {};
    }
  };

  TEST_CASE(
    "check completion signatures for sender that also supports error codes",
    "[concepts][sender]") {
    check_val_types<ex::__mset<pack<>>>(ec_sender{});
    check_err_types<ex::__mset<std::exception_ptr, int>>(ec_sender{});
    check_sends_stopped<false>(ec_sender{});
    STATIC_REQUIRE(ex::sender_of<ec_sender, ex::set_value_t()>);
  }

  struct my_r5_sender0 {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()
    >;

    auto connect(empty_recv::recv0&&) const -> oper {
      return {};
    }
  };

  TEST_CASE("r5 sender emits deprecated diagnostics", "[concepts][sender]") {
    void(ex::get_env(my_r5_sender0{}));
    static_assert(ex::sender<my_r5_sender0>);
    static_assert(std::same_as<
                  decltype(ex::get_completion_signatures(my_r5_sender0{}, ex::env<>{})),
                  my_r5_sender0::completion_signatures
    >);
  }

#if !STDEXEC_NVHPC()
  // nvc++ doesn't yet implement subsumption correctly
  struct not_a_sender_tag { };

  struct sender_tag { };

  struct sender_env_tag { };

  struct sender_of_tag { };

  template <class T>
  auto test_subsumption(T&&) -> not_a_sender_tag {
    return {};
  }

  template <ex::sender T>
  auto test_subsumption(T&&) -> sender_tag {
    return {};
  }

  template <ex::sender_in<ex::env<>> T>
  auto test_subsumption(T&&) -> sender_env_tag {
    return {};
  }

  template <ex::sender_of<ex::set_value_t(), ex::env<>> T>
  auto test_subsumption(T&&) -> sender_of_tag {
    return {};
  }

  template <class Expected, class T>
  void has_type(T&&) {
    STATIC_REQUIRE(std::same_as<T, Expected>);
  }

  TEST_CASE(
    "check for subsumption relationships between the sender concepts",
    "[concepts][sender]") {
    ::has_type<not_a_sender_tag>(::test_subsumption(42));
    ::has_type<sender_tag>(::test_subsumption(ex::get_scheduler()));
    ::has_type<sender_env_tag>(::test_subsumption(ex::just(42)));
    ::has_type<sender_of_tag>(::test_subsumption(ex::just()));
  }
#endif
} // namespace

STDEXEC_PRAGMA_POP()
