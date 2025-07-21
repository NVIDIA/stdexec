/*
 * Copyright (c) 2022 Lucian Radu Teodorescu
 * Copyright (c) 2022 NVIDIA Corporation
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
#include <exec/async_scope.hpp>

namespace ex = stdexec;
using namespace ex::tags;

namespace {
  template <class Rcvr>
  struct get_env_rcvr {
    using receiver_concept = ex::receiver_t;
    Rcvr rcvr;

    template <class... Values>
    void set_value(Values&&...) noexcept {
      auto env = ex::get_env(rcvr);
      ex::set_value(std::move(rcvr), std::move(env));
    }

    template <class Error>
    void set_error(Error&& err) noexcept {
      ex::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() noexcept {
      ex::set_stopped(std::move(rcvr));
    }

    [[nodiscard]]
    auto get_env() const noexcept {
      return ex::get_env(rcvr);
    }
  };

  template <class Sndr>
  struct get_env_sender {
    using sender_concept = ex::sender_t;
    Sndr sndr;

    template <ex::__decays_to<get_env_sender> Self, ex::receiver Rcvr>
    static auto connect(Self&& self, Rcvr rcvr) {
      return ex::connect(
        static_cast<Self&&>(self).sndr, get_env_rcvr<Rcvr>{static_cast<Rcvr&&>(rcvr)});
    }

    template <ex::__decays_to<get_env_sender> Self, class Env>
    static auto get_completion_signatures(Self&&, const Env&) {
      return ex::__try_make_completion_signatures<
        ex::__copy_cvref_t<Self, Sndr>,
        Env,
        ex::completion_signatures<set_value_t(Env)>,
        ex::__mconst<ex::completion_signatures<>>
      >{};
    }
  };

  struct probe_env_t {
    template <ex::sender Sndr>
    auto operator()(Sndr&& sndr) const {
      return get_env_sender<Sndr>{static_cast<Sndr&&>(sndr)};
    }

    auto operator()() const {
      return ex::__binder_back<probe_env_t>{{}, {}, {}};
    }
  };

  static const auto probe_env = probe_env_t{};

  static const auto env = ex::prop{ex::get_scheduler, inline_scheduler{}};

  TEST_CASE("Can pass stdexec::on sender to start_detached", "[adaptors][stdexec::on]") {
    ex::start_detached(ex::on(inline_scheduler{}, ex::just()), env);
  }

  TEST_CASE("Can pass stdexec::on sender to split", "[adaptors][stdexec::on]") {
    auto snd = ex::split(ex::on(inline_scheduler{}, ex::just()), env);
    (void) snd;
  }

  TEST_CASE("Can pass stdexec::on sender to ensure_started", "[adaptors][stdexec::on]") {
    auto snd = ex::ensure_started(ex::on(inline_scheduler{}, ex::just()), env);
    (void) snd;
  }

  TEST_CASE("Can pass stdexec::on sender to async_scope::spawn", "[adaptors][stdexec::on]") {
    exec::async_scope scope;
    impulse_scheduler sched;
    scope.spawn(ex::on(sched, ex::just()), env);
    sched.start_next();
    ex::sync_wait(scope.on_empty());
  }

  TEST_CASE("Can pass stdexec::on sender to async_scope::spawn_future", "[adaptors][stdexec::on]") {
    exec::async_scope scope;
    impulse_scheduler sched;
    auto fut = scope.spawn_future(ex::on(sched, ex::just(42)), env);
    sched.start_next();
    auto [i] = ex::sync_wait(std::move(fut)).value();
    CHECK(i == 42);
    ex::sync_wait(scope.on_empty());
  }

  TEST_CASE(
    "stdexec::on updates the current scheduler in the receiver",
    "[adaptors][stdexec::on]") {
    auto snd = ex::get_scheduler() | ex::on(inline_scheduler{}, probe_env())
             | ex::then([]<class Env>(Env) noexcept {
                 using Sched = ex::__call_result_t<ex::get_scheduler_t, Env>;
                 static_assert(ex::same_as<Sched, inline_scheduler>);
               })
             | probe_env() | ex::then([]<class Env>(Env) noexcept {
                 using Sched = ex::__call_result_t<ex::get_scheduler_t, Env>;
                 static_assert(ex::same_as<Sched, ex::run_loop::__scheduler>);
               });

    ex::sync_wait(snd);
  }
} // namespace
