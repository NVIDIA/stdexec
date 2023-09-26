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
#include <test_common/type_helpers.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/env.hpp>

#include <chrono>

namespace ex = stdexec;

using namespace std::chrono_literals;

TEST_CASE("start_detached simple test", "[consumers][start_detached]") {
  bool called{false};
  ex::start_detached(ex::just() | ex::then([&] { called = true; }));
  CHECK(called);
}

TEST_CASE("start_detached works with void senders", "[consumers][start_detached]") {
  ex::start_detached(ex::just());
}

TEST_CASE("start_detached works with multi-value senders", "[consumers][start_detached]") {
  ex::start_detached(ex::just(3, 0.1415));
}

// Trying to test `start_detached` with error flows will result in calling `std::terminate()`.
// We don't want that

TEST_CASE(
  "start_detached works with sender ending with `set_stopped`",
  "[consumers][start_detached]") {
  ex::start_detached(ex::just_stopped());
}

TEST_CASE(
  "start_detached works with senders that do not complete immediately",
  "[consumers][start_detached]") {
  impulse_scheduler sched;
  bool called{false};
  // Start the sender
  ex::start_detached(ex::transfer_just(sched) | ex::then([&] { called = true; }));
  // The `then` function is not yet called
  CHECK_FALSE(called);
  // After an impulse to the scheduler, the function would complete
  sched.start_next();
  CHECK(called);
}

TEST_CASE("start_detached works when changing threads", "[consumers][start_detached]") {
  exec::static_thread_pool pool{2};
  std::atomic<bool> called{false};
  {
    // lunch some work on the thread pool
    ex::sender auto snd = ex::transfer_just(pool.get_scheduler()) //
                        | ex::then([&] { called.store(true); });
    ex::start_detached(std::move(snd));
  }
  // wait for the work to be executed, with timeout
  // perform a poor-man's sync
  // NOTE: it's a shame that the `join` method in static_thread_pool is not public
  for (int i = 0; i < 1000 && !called.load(); i++)
    std::this_thread::sleep_for(1ms);
  // the work should be executed
  REQUIRE(called);
}

struct custom_sender {
  using is_sender = void;
  bool* called;

  template <class Receiver>
  STDEXEC_DEFINE_CUSTOM(auto connect)(this custom_sender, ex::connect_t, Receiver&& rcvr) {
    return ex::connect(ex::schedule(inline_scheduler{}), (Receiver&&) rcvr);
  }

  template <class Env>
  STDEXEC_DEFINE_CUSTOM(auto get_completion_signatures)(this custom_sender, ex::get_completion_signatures_t, Env) noexcept
    -> ex::completion_signatures<ex::set_value_t()> {
    return {};
  }

  friend void tag_invoke(ex::start_detached_t, custom_sender sndr) {
    *sndr.called = true;
  }

  STDEXEC_DEFINE_CUSTOM(empty_env get_env)(this const custom_sender&, ex::get_env_t) noexcept {
    return {};
  }
};

struct custom_scheduler {
  struct sender : ex::schedule_result_t<inline_scheduler> {
    struct env {
      template <class Tag>
      friend custom_scheduler tag_invoke(ex::get_completion_scheduler_t<Tag>, const env&) noexcept {
        return {};
      }
    };

    STDEXEC_DEFINE_CUSTOM(env get_env)(this const sender&, ex::get_env_t) noexcept {
      return {};
    }
  };

  struct domain {
    template <class Sender, class Env>
    friend void tag_invoke(ex::start_detached_t, domain, Sender, Env) {
      // drop the sender on the floor
    }

    // BUGBUG legacy
    operator custom_scheduler() const {
      return {};
    }
  };

  friend domain tag_invoke(ex::get_domain_t, custom_scheduler) noexcept {
    return {};
  }

  STDEXEC_DEFINE_CUSTOM(sender schedule)(this custom_scheduler, ex::schedule_t) noexcept {
    return {};
  }

  bool operator==(const custom_scheduler&) const = default;
};

TEST_CASE("start_detached can be customized on sender", "[consumers][start_detached]") {
  bool called = false;
  ex::start_detached(custom_sender{&called});
  CHECK(called);
}

// NOT TO SPEC
TEST_CASE("start_detached can be customized on scheduler", "[consumers][start_detached]") {
  bool called = false;
  ex::start_detached(
    ex::just() | ex::then([&] { called = true; }),
    exec::make_env(exec::with(ex::get_scheduler, custom_scheduler{})));
  CHECK_FALSE(called);
}
