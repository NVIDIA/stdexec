#define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING 1
/*
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

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <iostream>

namespace ex = stdexec;

struct receiver_t {
  using receiver_concept = ex::receiver_t;

  void set_value() && noexcept {
  }

  template <class... As>
  void set_value(As...) && noexcept {
  }

  template <class Error>
  void set_error(Error) && noexcept {
  }

  void set_stopped() && noexcept {
  }
};

template <class Domain>
void check_if_pool_domain() {
  using pool_domain = exec::_pool_::_static_thread_pool::domain;
  static_assert(std::is_same_v<Domain, pool_domain>);
}

template <class Domain>
void check_if_inline_domain() {
  static_assert(std::is_same_v<Domain, ex::default_domain>);
}

template <class Sender, class Receiver>
void check_if_starts_inline_and_completes_on_pool(Sender, Receiver) {
  using receiver_env_t = ex::env_of_t<Receiver>;

  check_if_pool_domain<
    ex::__detail::__completing_domain_t<ex::set_value_t, Sender, receiver_env_t>
  >();
  check_if_inline_domain<ex::__detail::__starting_domain_t<receiver_env_t>>();

  // auto op_state = ex::connect(std::move(sender), std::move(receiver));
  // op_state.start();
}

#if 0
template <class T>
void print(T) {
  std::cout << __PRETTY_FUNCTION__ << "\n";
}
#endif

template <class T, class Env = ex::env<>>
struct expect_value_receiver_ex {
  T dest_;
  Env env_{};

  using receiver_concept = stdexec::receiver_t;

  explicit expect_value_receiver_ex(T dest)
    : dest_(dest) {
  }

  expect_value_receiver_ex(Env env, T dest)
    : dest_(dest)
    , env_(std::move(env)) {
  }

  void set_value(T val) noexcept {
    dest_ = val;
  }

  template <class... Ts>
  void set_value(Ts...) noexcept {
    std::cerr << "set_value called with wrong value types on expect_value_receiver_ex\n";
  }

  void set_stopped() noexcept {
    std::cerr << "set_stopped called on expect_value_receiver_ex\n";
  }

  void set_error(std::exception_ptr) noexcept {
    std::cerr << "set_error called on expect_value_receiver_ex\n";
  }

  auto get_env() const noexcept -> Env {
    return env_;
  }
};

template <ex::scheduler Sched = ex::inline_scheduler>
inline auto _with_scheduler(Sched sched = {}) {
  return ex::write_env(ex::prop{ex::get_scheduler, std::move(sched)});
}

int main() {
  exec::static_thread_pool pool(3);
  auto sched = pool.get_scheduler();

  // check_if_starts_inline_and_completes_on_pool(ex::schedule(sched), receiver_t{});
  // check_if_starts_inline_and_completes_on_pool(ex::continues_on(ex::just(), sched), receiver_t{});
  // check_if_starts_inline_and_completes_on_pool(ex::let_value(ex::continues_on(ex::just(), sched), []() { return ex::just(); }), receiver_t{});

  // fails
  // check_if_starts_inline_and_completes_on_pool(ex::starts_on(sched, ex::just()), receiver_t{});

#if 0
  print(
    ex::get_completion_domain<ex::set_value_t>(
      ex::get_env(
        ex::starts_on(sched, ex::just())
      ),
      ex::get_env(receiver_t{})
    )
  );

  std::cout << "main: " << std::this_thread::get_id() << "\n";
  auto snd = ex::starts_on(sched, ex::just())
           | ex::bulk(ex::par_unseq, 2, [](int i) {
    std::cout << "   " << i << ": " << std::this_thread::get_id() << "\n";
  });
  ex::sync_wait(snd);
#else

  [[maybe_unused]]
  bool called{false};
  // launch some work on the thread pool
  ex::sender auto snd =
    ex::starts_on(sched, ex::just())
    //  ex::just() | ex::continues_on(sched) | ex::let_value([]() { return ex::just(); })
    | ex::continues_on(ex::inline_scheduler{});
  ex::sync_wait(std::move(snd));
#endif

  return 0;
}
