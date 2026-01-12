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
#include <exec/env.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#include <test_common/receivers.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/tuple.hpp>
#include <test_common/type_helpers.hpp>

#include <thread>

namespace ex = STDEXEC;

using namespace std::chrono_literals;

namespace {

  TEST_CASE("sync_wait simple test", "[consumers][sync_wait]") {
    std::optional<std::tuple<int>> res = ex::sync_wait(ex::just(49));
    CHECK(res.has_value());
    CHECK(std::get<0>(res.value()) == 49);
  }

  TEST_CASE("sync_wait can wait on void values", "[consumers][sync_wait]") {
    std::optional<std::tuple<>> res = ex::sync_wait(ex::just());
    CHECK(res.has_value());
  }

  TEST_CASE("sync_wait can wait on senders sending value packs", "[consumers][sync_wait]") {
    std::optional<std::tuple<int, double>> res = ex::sync_wait(ex::just(3, 0.1415));
    CHECK(res.has_value());
    CHECK(std::get<0>(res.value()) == 3);
    CHECK(std::get<1>(res.value()) == 0.1415);
  }

#if !STDEXEC_NO_STD_EXCEPTIONS()
  TEST_CASE("sync_wait rethrows received exception", "[consumers][sync_wait]") {
    // Generate an exception pointer object
    std::exception_ptr eptr;
    try {
      throw std::logic_error("err");
    } catch (...) {
      eptr = std::current_exception();
    }

    // Ensure that ex::sync_wait will rethrow the error
    try {
      error_scheduler<std::exception_ptr> sched{eptr};
      ex::sync_wait(ex::transfer_just(sched, 19));
      FAIL("exception not thrown?");
    } catch (const std::logic_error& e) {
      CHECK(std::string{e.what()} == "err");
    } catch (...) {
      FAIL("invalid exception received");
    }
  }

  TEST_CASE("sync_wait handling error_code errors", "[consumers][sync_wait]") {
    try {
      error_scheduler<std::error_code> sched{
        std::make_error_code(std::errc::argument_out_of_domain)};
      ex::sender auto snd = ex::transfer_just(sched, 19);
      static_assert(std::invocable<ex::sync_wait_t, decltype(snd)>);
      ex::sync_wait(std::move(snd)); // doesn't work
      FAIL("expecting exception to be thrown");
    } catch (const std::system_error& e) {
      CHECK(e.code() == std::errc::argument_out_of_domain);
    } catch (...) {
      FAIL("expecting std::system_error exception to be thrown");
    }
  }

  TEST_CASE("sync_wait handling non-exception errors", "[consumers][sync_wait]") {
    try {
      error_scheduler<std::string> sched{std::string{"err"}};
      ex::sender auto snd = ex::transfer_just(sched, 19);
      static_assert(std::invocable<ex::sync_wait_t, decltype(snd)>);
      ex::sync_wait(std::move(snd)); // doesn't work
      FAIL("expecting exception to be thrown");
    } catch (const std::string& e) {
      CHECK(e == "err");
    } catch (...) {
      FAIL("expecting std::string exception to be thrown");
    }
  }
#endif // !STDEXEC_NO_STD_EXCEPTIONS()

  TEST_CASE("sync_wait returns empty optional on cancellation", "[consumers][sync_wait]") {
    stopped_scheduler sched;
    std::optional<std::tuple<int>> res = ex::sync_wait(ex::transfer_just(sched, 19));
    CHECK_FALSE(res.has_value());
  }

  template <class T>
  auto always(T t) {
    return [t](auto&&...) mutable {
      return std::move(t);
    };
  }

  TEST_CASE("sync_wait doesn't accept multi-variant senders", "[consumers][sync_wait]") {
    ex::sender auto snd = fallible_just{13} | ex::let_error(always(ex::just(std::string{"err"})));
    check_val_types<ex::__mset<pack<int>, pack<std::string>>>(std::move(snd));
    // static_assert(!std::invocable<ex::sync_wait_t, decltype(snd)>);
  }

  TEST_CASE(
    "sync_wait_with_variant accepts multi-variant senders",
    "[consumers][sync_wait_with_variant]") {
    ex::sender auto snd = fallible_just{13} | ex::let_error(always(ex::just(std::string{"err"})));
    check_val_types<ex::__mset<pack<int>, pack<std::string>>>(std::move(snd));
    static_assert(std::invocable<ex::sync_wait_with_variant_t, decltype(snd)>);

    std::optional<std::tuple<std::variant<std::tuple<int>, std::tuple<std::string>>>> res =
      ex::sync_wait_with_variant(std::move(snd));

    CHECK(res.has_value());
    CHECK_TUPLE(std::get<0>(std::get<0>(res.value())) == std::make_tuple(13));
  }

  TEST_CASE(
    "sync_wait_with_variant accepts single-value senders",
    "[consumers][sync_wait_with_variant]") {
    ex::sender auto snd = ex::just(13);
    check_val_types<ex::__mset<pack<int>>>(snd);
    static_assert(std::invocable<ex::sync_wait_with_variant_t, decltype(snd)>);

    std::optional<std::tuple<std::variant<std::tuple<int>>>> res = ex::sync_wait_with_variant(snd);

    CHECK(res.has_value());
    CHECK_TUPLE(std::get<0>(std::get<0>(res.value())) == std::make_tuple(13));
  }

  TEST_CASE("sync_wait works if signaled from a different thread", "[consumers][sync_wait]") {
    std::atomic<bool> thread_started{false};
    std::atomic<bool> thread_stopped{false};
    impulse_scheduler sched;

    // Thread that calls `ex::sync_wait`
    auto waiting_thread = std::thread{[&] {
      thread_started.store(true);

      // Wait for a result that is triggered by the impulse scheduler
      std::optional<std::tuple<int>> res = ex::sync_wait(ex::transfer_just(sched, 49));
      CHECK(res.has_value());
      CHECK(std::get<0>(res.value()) == 49);

      thread_stopped.store(true);
    }};
    // Wait for the thread to start (poor-man's sync)
    for (int i = 0; i < 10'000 && !thread_started.load(); i++)
      std::this_thread::sleep_for(100us);

    // The thread should be waiting on the impulse
    CHECK_FALSE(thread_stopped.load());
    sched.start_next();

    // Now, the thread should exit
    waiting_thread.join();
    CHECK(thread_stopped.load());
  }

  TEST_CASE(
    "sync_wait can wait on operations happening on different threads",
    "[consumers][sync_wait]") {
    auto square = [](int x) {
      return x * x;
    };

    exec::static_thread_pool pool{3};
    ex::scheduler auto sched = pool.get_scheduler();
    ex::sender auto snd = ex::when_all(
      ex::transfer_just(sched, 2) | ex::then(square),
      ex::transfer_just(sched, 3) | ex::then(square),
      ex::transfer_just(sched, 5) | ex::then(square));
    std::optional<std::tuple<int, int, int>> res = ex::sync_wait(std::move(snd));
    CHECK(res.has_value());
    CHECK(std::get<0>(res.value()) == 4);
    CHECK(std::get<1>(res.value()) == 9);
    CHECK(std::get<2>(res.value()) == 25);
  }

  // This domain is used to customize the behavior of sync_wait and sync_wait_with_variant
  // for senders with particular completion signatures.
  struct sync_wait_test_domain {
    using single_result_t = std::optional<std::tuple<std::string>>;
    using multi_result_t = std::optional<std::variant<std::tuple<int>, std::tuple<std::string>>>;

    template <class Sender>
      requires std::same_as<
        ex::value_types_of_t<Sender, ex::env<>, std::type_identity_t, std::type_identity_t>,
        std::string
      >
    static auto apply_sender(ex::sync_wait_t, Sender&&) -> single_result_t {
      return {std::string{"ciao"}};
    }

    template <class Sender>
      requires ex::__mset_eq<
        ex::value_types_of_t<Sender, ex::env<>, std::type_identity_t, ex::__mmake_set>,
        ex::__mset<std::string, int>
      >
    static auto apply_sender(ex::sync_wait_with_variant_t, Sender&&) -> multi_result_t {
      return {std::string{"ciao_multi"}};
    }
  };

  TEST_CASE("sync_wait can be customized", "[consumers][sync_wait]") {
    basic_inline_scheduler<sync_wait_test_domain> sched;

    // The customization will return a different value
    auto snd = ex::starts_on(sched, ex::just(std::string{"hello"}));
    auto res = ex::sync_wait(std::move(snd));
    STATIC_REQUIRE(std::same_as<decltype(res), sync_wait_test_domain::single_result_t>);
    CHECK(res.has_value());
    CHECK(std::get<0>(res.value()) == "ciao");
  }

  // TODO(ericniebler)
#if 0
  TEST_CASE("sync_wait_with_variant can be customized", "[consumers][sync_wait_with_variant]") {
    basic_inline_scheduler<sync_wait_test_domain> sched;

    // The customization will return a different value
    auto snd = ex::starts_on(
      sched, 
      fallible_just(std::string{"hello_multi"}) | ex::let_error(always(ex::just(42))));
    auto res = ex::sync_wait_with_variant(std::move(snd));
    STATIC_REQUIRE(std::same_as<decltype(res), sync_wait_test_domain::multi_result_t>);
    CHECK(res.has_value());
    CHECK(std::get<0>(std::get<1>(res.value())) == std::string{"ciao_multi"});
  }
#endif

  template <class... Ts>
  using decayed_tuple = std::tuple<std::decay_t<Ts>...>;

  TEST_CASE(
    "sync_wait spec's return type defined in terms of value_types_of_t",
    "[consumers][sync_wait]") {
    static_assert(
      std::is_same_v<
        std::tuple<>,
        ex::value_types_of_t<decltype(ex::just()), ex::env<>, decayed_tuple, std::type_identity_t>
      >);
  }

} // namespace
