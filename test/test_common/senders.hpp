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

#pragma once

#include <catch2/catch.hpp>
#include <test_common/type_helpers.hpp>
#include <stdexec/execution.hpp>

namespace ex = stdexec;

template <class... Values>
struct fallible_just {
  std::tuple<Values...> values_;
  using is_sender = void;
  using completion_signatures =
    ex::completion_signatures<ex::set_value_t(Values...), ex::set_error_t(std::exception_ptr)>;

  template <class Receiver>
  struct operation : immovable {
    std::tuple<Values...> values_;
    Receiver rcvr_;

    friend void tag_invoke(ex::start_t, operation& self) noexcept try {
      std::apply(
        [&](Values&... ts) { ex::set_value(std::move(self.rcvr_), std::move(ts)...); },
        self.values_);
    } catch (...) {

      ex::set_error(std::move(self.rcvr_), std::current_exception());
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, fallible_just&& self, Receiver&& rcvr)
    -> operation<std::decay_t<Receiver>> {
    return {{}, std::move(self.values_), std::forward<Receiver>(rcvr)};
  }

  friend empty_env tag_invoke(ex::get_env_t, const fallible_just&) noexcept {
    return {};
  }
};

template <class... Values>
fallible_just(Values...) -> fallible_just<Values...>;

struct value_env {
  int value;
};

template <class Attrs, class... Values>
struct just_with_env {
  std::remove_cvref_t<Attrs> env_;
  std::tuple<Values...> values_;
  using is_sender = void;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(Values...)>;

  template <class Receiver>
  struct operation : immovable {
    std::tuple<Values...> values_;
    Receiver rcvr_;

    friend void tag_invoke(ex::start_t, operation& self) noexcept {
      std::apply(
        [&](Values&... ts) { ex::set_value(std::move(self.rcvr_), std::move(ts)...); },
        self.values_);
    }
  };

  template <class Receiver>
  friend auto tag_invoke(ex::connect_t, just_with_env&& self, Receiver&& rcvr)
    -> operation<std::decay_t<Receiver>> {
    return {{}, std::move(self.values_), std::forward<Receiver>(rcvr)};
  }

  friend Attrs tag_invoke(ex::get_env_t, const just_with_env& self) noexcept {
    return self.env_;
  }
};

struct completes_if {
  using __t = completes_if;
  using __id = completes_if;
  using is_sender = void;
  using completion_signatures = ex::completion_signatures<ex::set_value_t(), ex::set_stopped_t()>;

  bool condition_;

  template <class Receiver>
  struct operation {

    bool condition_;
    Receiver rcvr_;

    // without this synchronization, the thread sanitzier shows a race for construction and
    // destruction of on_stop_
    enum class state_t {
      construction,
      emplaced,
      stopped
    };
    std::atomic<state_t> state_{state_t::construction};

    struct on_stopped {
      operation& self_;

      void operator()() noexcept {
        state_t expected = self_.state_.load(std::memory_order_relaxed);
        while (!self_.state_.compare_exchange_weak(
          expected, state_t::stopped, std::memory_order_acq_rel))
          ;
        if (expected == state_t::emplaced) {
          ex::set_stopped(std::move(self_.rcvr_));
        }
      }
    };

    using callback_t =
      typename ex::stop_token_of_t<ex::env_of_t<Receiver>&>::template callback_type<on_stopped>;
    std::optional<callback_t> on_stop_{};

    friend void tag_invoke(ex::start_t, operation& self) noexcept {
      if (self.condition_) {
        ex::set_value(std::move(self.rcvr_));
      } else {
        self.on_stop_.emplace(ex::get_stop_token(ex::get_env(self.rcvr_)), on_stopped{self});
        state_t expected = state_t::construction;
        if (!self.state_.compare_exchange_strong(
              expected, state_t::emplaced, std::memory_order_acq_rel)) {
          ex::set_stopped(std::move(self.rcvr_));
        }
      }
    }
  };

  template <ex::__decays_to<completes_if> Self, class Receiver>
  friend auto tag_invoke(ex::connect_t, Self&& self, Receiver&& rcvr) noexcept
    -> operation<std::decay_t<Receiver>> {
    return {self.condition_, std::forward<Receiver>(rcvr)};
  }

  friend empty_env tag_invoke(ex::get_env_t, const completes_if&) noexcept {
    return {};
  }
};
