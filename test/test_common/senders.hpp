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
#include <memory>
#include <stdexec/execution.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

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
    auto connect(Receiver) const noexcept {
      return operation{};
    }
  };

  template <class... Values>
  struct fallible_just {
    using sender_concept = STDEXEC::sender_t;

    explicit fallible_just(Values... values)
      : values_(std::move(values)...) {
    }
    fallible_just(fallible_just&&) noexcept = default;

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
    auto connect(Receiver rcvr) && -> operation<Receiver> {
      return {{}, std::move(values_), std::forward<Receiver>(rcvr)};
    }

    struct attrs {
      [[nodiscard]]
      auto query(ex::get_completion_behavior_t<ex::set_value_t>) const noexcept {
        return ex::completion_behavior::inline_completion;
      }
      [[nodiscard]]
      auto query(ex::get_completion_behavior_t<ex::set_error_t>) const noexcept {
        return ex::completion_behavior::inline_completion;
      }
    };

    [[nodiscard]]
    auto get_env() const noexcept -> attrs {
      return {};
    }

    template <std::same_as<fallible_just> Self>
    static consteval auto get_completion_signatures() noexcept {
      return ex::completion_signatures<
        ex::set_value_t(Values...),
        ex::set_error_t(std::exception_ptr)
      >{};
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
    using sender_concept = STDEXEC::sender_t;
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
    auto connect(Receiver rcvr) && -> operation<Receiver> {
      return {{}, std::move(values_), std::forward<Receiver>(rcvr)};
    }

    auto get_env() const noexcept -> Env {
      return env_;
    }
  };

  struct completes_if {
    using sender_concept = STDEXEC::sender_t;
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
          while (!self_.state_
                    .compare_exchange_weak(expected, state_t::stopped, std::memory_order_acq_rel))
            ;
          if (expected == state_t::emplaced) {
            ex::set_stopped(std::move(self_.rcvr_));
          }
        }
      };

      using callback_t =
        ex::stop_token_of_t<ex::env_of_t<Receiver>&>::template callback_type<on_stopped>;
      std::optional<callback_t> on_stop_{};

      void start() & noexcept {
        if (condition_) {
          ex::set_value(std::move(rcvr_));
        } else {
          on_stop_.emplace(ex::get_stop_token(ex::get_env(rcvr_)), on_stopped{*this});
          state_t expected = state_t::construction;
          if (!state_
                 .compare_exchange_strong(expected, state_t::emplaced, std::memory_order_acq_rel)) {
            ex::set_stopped(std::move(rcvr_));
          }
        }
      }
    };

    template <ex::__decays_to<completes_if> Self, class Receiver>
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr) noexcept
      -> operation<Receiver> {
      return {self.condition_, std::forward<Receiver>(rcvr)};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)
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

  template <class Tag, class... Args>
  struct succeed_n_sender {
    using sender_concept = STDEXEC::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t(), Tag(Args...)>;

    explicit succeed_n_sender(int count, Tag, Args... args)
      : args_(std::move(args)...)
      , counter_(std::make_shared<std::atomic<int>>(count)) {
    }

    template <class Receiver>
    struct operation {
      void start() noexcept {
        if (--*counter_ == -1) {
          std::apply(
            [&](Args&... args) -> void { Tag{}(std::move(rcvr_), std::move(args)...); }, args_);
        } else {
          ex::set_value(std::move(rcvr_));
        }
      }

      std::tuple<Args...> args_;
      std::shared_ptr<std::atomic<int>> counter_;
      Receiver rcvr_;
    };

    template <class Receiver>
    auto connect(Receiver rcvr) const -> operation<Receiver> {
      return {args_, counter_, std::forward<Receiver>(rcvr)};
    }

   private:
    std::tuple<Args...> args_;
    std::shared_ptr<std::atomic<int>> counter_;
  };

  // A sender that sends by reference
  template <class Type>
  struct just_ref {
    using sender_concept = ex::sender_t;

    explicit just_ref(Type& value)
      : value_(value) {
    }

    template <class>
    static consteval auto get_completion_signatures() noexcept {
      return ex::completion_signatures<ex::set_value_t(Type&)>{};
    }

    template <class Receiver>
    auto connect(Receiver rcvr) const noexcept {
      return opstate<Receiver>{value_, static_cast<Receiver&&>(rcvr)};
    }

   private:
    template <class Receiver>
    struct opstate {
      using operation_state_concept = ex::operation_state_t;

      explicit opstate(Type& value, Receiver rcvr)
        : value_(value)
        , rcvr_(std::move(rcvr)) {
      }

      void start() noexcept {
        ex::set_value(static_cast<Receiver&&>(rcvr_), value_);
      }

      Type& value_;
      Receiver rcvr_;
    };

    Type& value_;
  };

} // namespace
