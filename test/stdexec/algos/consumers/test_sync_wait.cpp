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
#include <test_common/senders.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <exec/static_thread_pool.hpp>

#include <thread>
#include <chrono>

namespace ex = stdexec;
using std::optional;
using std::tuple;
using stdexec::sync_wait;
using stdexec::sync_wait_with_variant;

using namespace std::chrono_literals;

TEST_CASE("sync_wait simple test", "[consumers][sync_wait]") {
  optional<tuple<int>> res = sync_wait(ex::just(49));
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == 49);
}

TEST_CASE("sync_wait can wait on void values", "[consumers][sync_wait]") {
  optional<tuple<>> res = sync_wait(ex::just());
  CHECK(res.has_value());
}

TEST_CASE("sync_wait can wait on senders sending value packs", "[consumers][sync_wait]") {
  optional<tuple<int, double>> res = sync_wait(ex::just(3, 0.1415));
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == 3);
  CHECK(std::get<1>(res.value()) == 0.1415);
}

TEST_CASE("sync_wait rethrows received exception", "[consumers][sync_wait]") {
  // Generate an exception pointer object
  std::exception_ptr eptr;
  try {
    throw std::logic_error("err");
  } catch (...) {
    eptr = std::current_exception();
  }

  // Ensure that sync_wait will rethrow the error
  try {
    error_scheduler<std::exception_ptr> sched{eptr};
    sync_wait(ex::transfer_just(sched, 19));
    FAIL("exception not thrown?");
  } catch (const std::logic_error& e) {
    CHECK(std::string{e.what()} == "err");
  } catch (...) {
    FAIL("invalid exception received");
  }
}

TEST_CASE("sync_wait handling error_code errors", "[consumers][sync_wait]") {
  try {
    error_scheduler<std::error_code> sched{std::make_error_code(std::errc::argument_out_of_domain)};
    ex::sender auto snd = ex::transfer_just(sched, 19);
    static_assert(std::invocable<decltype(sync_wait), decltype(snd)>);
    sync_wait(std::move(snd)); // doesn't work
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
    static_assert(std::invocable<decltype(sync_wait), decltype(snd)>);
    sync_wait(std::move(snd)); // doesn't work
    FAIL("expecting exception to be thrown");
  } catch (const std::string& e) {
    CHECK(e == "err");
  } catch (...) {
    FAIL("expecting std::string exception to be thrown");
  }
}

TEST_CASE("sync_wait returns empty optional on cancellation", "[consumers][sync_wait]") {
  stopped_scheduler sched;
  optional<tuple<int>> res = sync_wait(ex::transfer_just(sched, 19));
  CHECK_FALSE(res.has_value());
}

template <class T>
auto always(T t) {
  return [t](auto&&...) mutable { return std::move(t); };
}

TEST_CASE("sync_wait doesn't accept multi-variant senders", "[consumers][sync_wait]") {
  ex::sender auto snd =
      fallible_just{13} //
      | ex::let_error(always(ex::just(std::string{"err"})));
  check_val_types<type_array<type_array<int>, type_array<std::string>>>(snd);
  static_assert(!std::invocable<decltype(sync_wait), decltype(snd)>);
}

TEST_CASE("sync_wait_with_variant accepts multi-variant senders", "[consumers][sync_wait_with_variant]") {
  ex::sender auto snd =
      fallible_just{13}
      | ex::let_error(always(ex::just(std::string{"err"})));
  check_val_types<type_array<type_array<int>, type_array<std::string>>>(snd);
  static_assert(std::invocable<decltype(sync_wait_with_variant), decltype(snd)>);

  std::optional<std::tuple<std::variant<std::tuple<int>, std::tuple<std::string>>>> res =
    sync_wait_with_variant(snd);

  CHECK(res.has_value());
  CHECK(std::get<0>(std::get<0>(res.value())) == std::make_tuple(13));
}

TEST_CASE("sync_wait_with_variant accepts single-value senders", "[consumers][sync_wait_with_variant]") {
  ex::sender auto snd = ex::just(13);
  check_val_types<type_array<type_array<int>>>(snd);
  static_assert(std::invocable<decltype(sync_wait_with_variant), decltype(snd)>);

  std::optional<std::tuple<std::variant<std::tuple<int>>>> res =
    sync_wait_with_variant(snd);

  CHECK(res.has_value());
  CHECK(std::get<0>(std::get<0>(res.value())) == std::make_tuple(13));
}

TEST_CASE("sync_wait works if signaled from a different thread", "[consumers][sync_wait]") {
  std::atomic<bool> thread_started{false};
  std::atomic<bool> thread_stopped{false};
  impulse_scheduler sched;

  // Thread that calls `sync_wait`
  auto waiting_thread = std::thread{[&] {
    thread_started.store(true);

    // Wait for a result that is triggered by the impulse scheduler
    optional<tuple<int>> res = sync_wait(ex::transfer_just(sched, 49));
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
    "sync_wait can wait on operations happening on different threads", "[consumers][sync_wait]") {
  auto square = [](int x) { return x * x; };

  exec::static_thread_pool pool{3};
  ex::scheduler auto sched = pool.get_scheduler();
  ex::sender auto snd = ex::when_all(                 //
      ex::transfer_just(sched, 2) | ex::then(square), //
      ex::transfer_just(sched, 3) | ex::then(square), //
      ex::transfer_just(sched, 5) | ex::then(square)  //
  );
  optional<tuple<int, int, int>> res = sync_wait(std::move(snd));
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == 4);
  CHECK(std::get<1>(res.value()) == 9);
  CHECK(std::get<2>(res.value()) == 25);
}

using my_string_sender_t = decltype(ex::transfer_just(inline_scheduler{}, std::string{}));

optional<tuple<std::string>> tag_invoke(
    decltype(sync_wait), inline_scheduler sched, my_string_sender_t&& s) {
  std::string res;
  auto op = ex::connect(std::move(s), expect_value_receiver_ex{res});
  ex::start(op);
  CHECK(res == "hello");
  // change the string
  res = "hallo";
  return {res};
}

struct my_other_string_sender_t {
  std::string str_;

  using completion_signatures = ex::completion_signatures_of_t<decltype(ex::just(std::string{}))>;

  template <class Recv>
  friend auto tag_invoke(ex::connect_t, my_other_string_sender_t&& self, Recv&& recv) {
    return ex::connect(ex::just(std::move(self.str_)), std::forward<Recv>(recv));
  }
  template <class Recv>
  friend auto tag_invoke(ex::connect_t, const my_other_string_sender_t& self, Recv&& recv) {
    return ex::connect(ex::just(self.str_), std::forward<Recv>(recv));
  }

  friend empty_attrs tag_invoke(ex::get_attrs_t, const my_other_string_sender_t&) noexcept {
    return {};
  }
};

optional<tuple<std::string>> tag_invoke(decltype(sync_wait), my_other_string_sender_t s) {
  CHECK(s.str_ == "hello");
  return {std::string{"ciao"}};
}

TEST_CASE("sync_wait can be customized with scheduler", "[consumers][sync_wait]") {
  // The customization will return a different value
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"hello"});
  optional<tuple<std::string>> res = sync_wait(std::move(snd));
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == "hallo");
}

TEST_CASE("sync_wait can be customized without scheduler", "[consumers][sync_wait]") {
  // The customization will return a different value
  my_other_string_sender_t snd{std::string{"hello"}};
  optional<tuple<std::string>> res = sync_wait(std::move(snd));
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == "ciao");
}

using multi_value_impl_t = decltype(fallible_just{std::string{}} | ex::let_error(always(ex::just(0))));
struct my_multi_value_sender_t {
  std::string str_;
  using completion_signatures = ex::completion_signatures_of_t<multi_value_impl_t>;

  template <class Recv>
  friend auto tag_invoke(ex::connect_t, my_multi_value_sender_t&& self, Recv&& recv) {
    return ex::connect(ex::just(std::move(self.str_)), std::forward<Recv>(recv));
  }
  template <class Recv>
  friend auto tag_invoke(ex::connect_t, const my_multi_value_sender_t& self, Recv&& recv) {
    return ex::connect(ex::just(self.str_), std::forward<Recv>(recv));
  }

  friend empty_attrs tag_invoke(ex::get_attrs_t, const my_multi_value_sender_t&) noexcept {
    return {};
  }
};

using my_transfered_multi_value_sender_t = decltype(ex::transfer(my_multi_value_sender_t{}, inline_scheduler{}));
optional<std::tuple<std::variant<std::tuple<std::string>, std::tuple<int>>>> tag_invoke(
    decltype(sync_wait_with_variant), inline_scheduler sched, my_transfered_multi_value_sender_t&& s) {
  std::string res;
  auto op = ex::connect(std::move(s), expect_value_receiver_ex{res});
  ex::start(op);
  CHECK(res == "hello_multi");
  // change the string
  res = "hallo_multi";
  return {res};
}

optional<std::tuple<std::variant<std::tuple<std::string>, std::tuple<int>>>> tag_invoke(decltype(sync_wait_with_variant), my_multi_value_sender_t s) {
  CHECK(s.str_ == "hello_multi");
  return {std::string{"ciao_multi"}};
}

TEST_CASE("sync_wait_with_variant can be customized with scheduler", "[consumers][sync_wait_with_variant]") {
  // The customization will return a different value
  auto snd = ex::transfer(my_multi_value_sender_t{"hello_multi"}, inline_scheduler{});
  auto snd2 = ex::transfer_just(inline_scheduler{}, std::string{"hello"});
  optional<std::tuple<std::variant<std::tuple<std::string>, std::tuple<int>>>> res = sync_wait_with_variant(std::move(snd));
  CHECK(res.has_value());
  CHECK(std::get<0>(std::get<0>(res.value())) == std::make_tuple(std::string{"hallo_multi"}));
}

TEST_CASE("sync_wait_with_variant can be customized without scheduler", "[consumers][sync_wait_with_variant]") {
  // The customization will return a different value
  my_multi_value_sender_t snd{std::string{"hello_multi"}};
  optional<std::tuple<std::variant<std::tuple<std::string>, std::tuple<int>>>> res = sync_wait_with_variant(std::move(snd));
  CHECK(res.has_value());
  CHECK(std::get<0>(std::get<0>(res.value())) == std::make_tuple(std::string{"ciao_multi"}));
}
