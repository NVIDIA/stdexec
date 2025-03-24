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

namespace {

  template <class... Sigs>
  struct a_sender_of {
    using sender_concept = ex::sender_t;
    using completion_signatures = ex::completion_signatures<Sigs...>;

    struct operation {
      void start() & noexcept {
      }
    };

    template <class Receiver>
    friend auto tag_invoke(ex::connect_t, a_sender_of, Receiver&&) noexcept {
      return operation();
    }
  };

  template <class... Values>
  struct fallible_just {
    using sender_concept = stdexec::sender_t;
    using completion_signatures =
      ex::completion_signatures<ex::set_value_t(Values...), ex::set_error_t(std::exception_ptr)>;

    explicit fallible_just(Values... values)
      : values_(std::move(values)...) {
    }

    template <class Receiver>
    struct operation : immovable {
      std::tuple<Values...> values_;
      Receiver rcvr_;

      void start() & noexcept {
        try {
          std::apply(
            [&](Values&... ts) { ex::set_value(std::move(rcvr_), std::move(ts)...); }, values_);
        } catch (...) {
          ex::set_error(std::move(rcvr_), std::current_exception());
        }
      }
    };

    template <class Receiver>
    friend auto tag_invoke(ex::connect_t, fallible_just&& self, Receiver&& rcvr)
      -> operation<std::decay_t<Receiver>> {
      return {{}, std::move(self.values_), std::forward<Receiver>(rcvr)};
    }

    std::tuple<Values...> values_;
  };

  template <class... Values>
  fallible_just(Values...) -> fallible_just<Values...>;

  inline constexpr struct value_query_t : ex::forwarding_query_t {
    template <class Env>
    [[nodiscard]]
    auto operator()(Env const & env) const noexcept -> decltype(env.query(*this)) {
      return env.query(*this);
    }
  } value_query;

  struct value_env {
    int value{};

    [[nodiscard]]
    auto query(value_query_t) const noexcept -> int {
      return value;
    }
  };

  template <class Env, class... Values>
  struct just_with_env {
    std::remove_cvref_t<Env> env_;
    std::tuple<Values...> values_;
    using sender_concept = stdexec::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(Values...)>;

    template <class Receiver>
    struct operation : immovable {
      std::tuple<Values...> values_;
      Receiver rcvr_;

      void start() & noexcept {
        std::apply(
          [&](Values&... ts) { ex::set_value(std::move(rcvr_), std::move(ts)...); }, values_);
      }
    };

    template <class Receiver>
    friend auto tag_invoke(ex::connect_t, just_with_env&& self, Receiver&& rcvr)
      -> operation<std::decay_t<Receiver>> {
      return {{}, std::move(self.values_), std::forward<Receiver>(rcvr)};
    }

    auto get_env() const noexcept -> Env {
      return env_;
    }
  };

  struct completes_if {
    using __t = completes_if;
    using __id = completes_if;
    using sender_concept = stdexec::sender_t;
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

      void start() & noexcept {
        if (condition_) {
          ex::set_value(std::move(rcvr_));
        } else {
          on_stop_.emplace(ex::get_stop_token(ex::get_env(rcvr_)), on_stopped{*this});
          state_t expected = state_t::construction;
          if (!state_.compare_exchange_strong(
                expected, state_t::emplaced, std::memory_order_acq_rel)) {
            ex::set_stopped(std::move(rcvr_));
          }
        }
      }
    };

    template <ex::__decays_to<completes_if> Self, class Receiver>
    friend auto tag_invoke(ex::connect_t, Self&& self, Receiver&& rcvr) noexcept
      -> operation<std::decay_t<Receiver>> {
      return {self.condition_, std::forward<Receiver>(rcvr)};
    }
  };

  struct non_default_constructible {
    int x;

    non_default_constructible(int x)
      : x(x) {
    }

    friend auto
      operator==(non_default_constructible const & lhs, non_default_constructible const & rhs)
        -> bool {
      return lhs.x == rhs.x;
    }
  };
} // namespace
