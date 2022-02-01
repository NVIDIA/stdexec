/*
 * Copyright (c) Lucian Radu Teodorescu
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
#include <execution.hpp>
#include <test_common/schedulers.hpp>
#include <test_common/senders.hpp>
#include <test_common/receivers.hpp>
#include <test_common/type_helpers.hpp>
#include <examples/schedulers/static_thread_pool.hpp>

#include <thread>
#include <chrono>

namespace ex = std::execution;
using std::optional;
using std::tuple;
using std::this_thread::sync_wait;

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

TEST_CASE("TODO: sync_wait handling non-exception errors", "[consumers][sync_wait]") {
  // TODO: the specification isn't clear what to do with non-exception errors
  error_scheduler<std::string> sched{std::string{"err"}};
  ex::sender auto snd = ex::transfer_just(sched, 19);
  static_assert(std::invocable<decltype(sync_wait), decltype(snd)>);
  // sync_wait(std::move(snd)); // doesn't work
}

TEST_CASE("sync_wait returns empty optional on cancellation", "[consumers][sync_wait]") {
  stopped_scheduler sched;
  optional<tuple<int>> res = sync_wait(ex::transfer_just(sched, 19));
  CHECK_FALSE(res.has_value());
}

TEST_CASE("sync_wait doesn't accept multi-variant senders", "[consumers][sync_wait]") {
  ex::sender auto snd =
      fallible_just{13} //
      | ex::let_error([](std::exception_ptr) { return ex::just(std::string{"err"}); });
  check_val_types<type_array<type_array<int>, type_array<std::string>>>(snd);
  static_assert(!std::invocable<decltype(sync_wait), decltype(snd)>);
}

TEST_CASE("TODO: sync_wait works if signaled from a different thread", "[consumers][sync_wait]") {
  bool thread_started{false};
  bool thread_stopped{false};
  impulse_scheduler sched;

  // Thread that calls `sync_wait`
  auto waiting_thread = std::thread{[&] {
    thread_started = true;

    // Wait for a result that is triggered by the impulse scheduler
    // TODO: find out why this hangs:
    //optional<tuple<int>> res = sync_wait(ex::transfer_just(sched, 49));
    optional<tuple<int>> res = sync_wait(ex::on(sched, ex::just(49)));
    CHECK(res.has_value());
    CHECK(std::get<0>(res.value()) == 49);

    thread_stopped = true;
  }};
  // Wait for the thread to start (poor-man's sync)
  for (int i = 0; i < 10'000 && !thread_started; i++)
    std::this_thread::sleep_for(100us);

  // The thread should be waiting on the impulse
  CHECK_FALSE(thread_stopped);
  sched.start_next();

  // Now, the thread should exit
  waiting_thread.join();
  CHECK(thread_stopped);
}

TEST_CASE(
    "sync_wait can wait on operations happening on different threads", "[consumers][sync_wait]") {
  auto square = [](int x) { return x * x; };

  example::static_thread_pool pool{3};
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
using just_string_sender_t = decltype(ex::just(std::string{}));

optional<tuple<std::string>> tag_invoke(
    decltype(sync_wait), inline_scheduler sched, my_string_sender_t&& s) {
  std::string res;
  auto op = ex::connect(std::move(s), expect_value_receiver_ex{&res});
  ex::start(op);
  CHECK(res == "hello");
  // change the string
  res = "hallo";
  return {res};
}

optional<tuple<std::string>> tag_invoke(decltype(sync_wait), just_string_sender_t s) {
  std::string res;
  auto op = ex::connect(std::move(s), expect_value_receiver_ex{&res});
  ex::start(op);
  CHECK(res == "hello");
  // change the string
  res = "ciao";
  return {res};
}

TEST_CASE("sync_wait can be customized with scheduler", "[consumers][sync_wait]") {
  // The customization will return a different value
  auto snd = ex::transfer_just(inline_scheduler{}, std::string{"hello"});
  optional<tuple<std::string>> res = sync_wait(std::move(snd));
  CHECK(res.has_value());
  CHECK(std::get<0>(res.value()) == "hallo");
}

TEST_CASE("TODO: sync_wait can be customized without scheduler", "[consumers][sync_wait]") {
  // The customization will return a different value
  auto snd = ex::just(std::string{"hello"});
  optional<tuple<std::string>> res = sync_wait(std::move(snd));
  CHECK(res.has_value());
  // TODO: customization doesn't work
  // CHECK(std::get<0>(res.value()) == "ciao");
  // invalid check:
  CHECK(std::get<0>(res.value()) == "hello");
}
