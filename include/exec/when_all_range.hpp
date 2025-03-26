/*
 * Copyright (c) 2024 Maikel Nadolski
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

#include <stdexec/execution.hpp>

#include <exec/env.hpp>

#include <atomic>
#include <compare>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <variant>

namespace exec {
  template <class T1, class T2>
  class manual_alternative {
   public:
    manual_alternative(manual_alternative&& other) {
      emplace<0>(std::move(other.storage_.first));
    }

    template <class... Args>
    explicit constexpr manual_alternative(std::in_place_t, Args&&... args) {
      emplace<0>(std::forward<Args>(args)...);
    }

    template <std::size_t Idx, class... Args>
    constexpr void emplace(Args&&... args) {
      if constexpr (Idx == 0) {
        new (&storage_.first) T1(std::forward<Args>(args)...);
      } else {
        new (&storage_.second) T2(std::forward<Args>(args)...);
      }
    }

    template <std::size_t Idx, class Fn, class... Args>
    constexpr void emplace_from(Fn&& fun, Args&&... args) {
      if constexpr (Idx == 0) {
        new (&storage_.first) T1(std::forward<Fn>(fun)(std::forward<Args>(args)...));
      } else {
        new (&storage_.second) T2(std::forward<Fn>(fun)(std::forward<Args>(args)...));
      }
    }

    template <std::size_t Idx>
    constexpr void destroy() noexcept {
      if constexpr (Idx == 0) {
        storage_.first.~T1();
      } else {
        storage_.second.~T2();
      }
    }

    template <std::size_t Idx, class Self>
    constexpr auto&& get(this Self&& self) noexcept {
      if constexpr (Idx == 0) {
        return std::forward<Self>(self).storage_.first;
      } else {
        return std::forward<Self>(self).storage_.second;
      }
    }

   private:
    union storage_type {
      constexpr storage_type() {
      }

      constexpr ~storage_type() noexcept {
      }

      T1 first;
      T2 second;
    };

    storage_type storage_;
  };

  template <auto>
  class intrusive_queue;

  template <class Tp, Tp* Tp::* Next>
  class intrusive_queue<Next> {
   public:
    intrusive_queue() noexcept = default;

    void push(Tp& object) noexcept {
      auto* item = std::addressof(object);
      item->*Next = nullptr;
      if (tail_ == nullptr) {
        head_ = item;
      } else {
        tail_->*Next = item;
      }
      tail_ = item;
      size_ += 1;
    }

    [[nodiscard]]
    Tp* pop() noexcept {
      if (head_ == nullptr) {
        return nullptr;
      }
      auto* item = head_;
      head_ = head_->*Next;
      if (head_ == nullptr) {
        tail_ = nullptr;
      }
      size_ -= 1;
      return item;
    }

    [[nodiscard]]
    std::size_t size() const noexcept {
      return size_;
    }

   private:
    Tp* head_{nullptr};
    Tp* tail_{nullptr};
    std::size_t size_{0};
  };

  namespace when_all_range_ {
    struct unit { };

    template <class Result>
    struct local_operation_result {
      local_operation_result* next_{nullptr};

      using storage_type = std::conditional_t<std::is_void_v<Result>, unit, std::optional<Result>>;

      [[no_unique_address]]
      storage_type result_{};
    };

    template <class Result>
    struct operation_results_base {
      intrusive_queue<&local_operation_result<Result>::next_> results_{};
    };

    template <>
    struct operation_results_base<void> { };

    template <class Result, class ErrorVariant, class Receiver>
    struct operation_base : operation_results_base<Result> {
      struct stop_callback_t {
        operation_base& op_;

        void operator()() noexcept {
          op_.notify_stopped();
        }
      };

      using stop_token = stdexec::stop_token_of_t<stdexec::env_of_t<Receiver>>;
      using stop_callback_type = stdexec::stop_callback_for_t<stop_token, stop_callback_t>;

      ErrorVariant error_{};
      Receiver receiver_;
      stdexec::inplace_stop_source stop_source_{};
      std::atomic<std::ptrdiff_t> count_{0};
      std::atomic<int> disposition_{0};
      stdexec::__manual_lifetime<stop_callback_type> stop_callback_{};

      explicit operation_base(Receiver receiver) noexcept
        : receiver_(std::move(receiver)) {
      }

      void do_start() noexcept {
        stop_callback_.__construct(
          stdexec::get_stop_token(stdexec::get_env(receiver_)), stop_callback_t{*this});
      }

      void notify() noexcept {
        if (count_.fetch_sub(1) == 1) {
          stop_callback_.__destroy();
          switch (disposition_) {
          case 0: {
            if constexpr (std::is_void_v<Result>) {
              stdexec::set_value(std::move(receiver_));
            } else {
              std::vector<Result> result;
              result.reserve(this->results_.size());
              while (auto* item = this->results_.pop()) {
                assert(item->result_);
                result.push_back(*std::exchange(item->result_, std::nullopt));
              }
              stdexec::set_value(std::move(receiver_), std::move(result));
            }
            break;
          }
          case 1: {
            std::visit(
              [&]<class Err>(Err&& err) noexcept {
                if constexpr (!std::same_as<Err, unit>) {
                  stdexec::set_error(std::move(receiver_), std::move(err));
                }
              },
              std::move(error_));
            break;
          }
          case 2:
            stdexec::set_stopped(std::move(receiver_));
            break;
          }
        }
      }

      template <class Error>
      void notify_error(Error&& err) noexcept {
        int expected_disposition = 0;
        if (disposition_.compare_exchange_strong(expected_disposition, 1)) {
          try {
            error_.template emplace<std::remove_cvref_t<Error>>(std::forward<Error>(err));
          } catch (...) {
            if constexpr (!std::is_nothrow_constructible_v<std::remove_cvref_t<Error>, Error>) {
              error_.template emplace<std::exception_ptr>(std::current_exception());
            }
          }
          stop_source_.request_stop();
        }
        this->notify();
      }

      void notify_stopped() noexcept {
        int expected_disposition = 0;
        if (disposition_.compare_exchange_strong(expected_disposition, 2)) {
          stop_source_.request_stop();
        }
        this->notify();
      }
    };

    template <class Result, class ErrorVariant, class Receiver>
    struct local_operation_base : local_operation_result<Result> {

      explicit local_operation_base(operation_base<Result, ErrorVariant, Receiver>& parent) noexcept
        : local_operation_result<Result>{}
        , parent_{parent} {
        if constexpr (!std::is_void_v<Result>) {
          parent_.results_.push(*this);
        }
      }

      local_operation_base(const local_operation_base&) = delete;
      local_operation_base& operator=(const local_operation_base&) = delete;
      local_operation_base(local_operation_base&&) = delete;
      local_operation_base& operator=(local_operation_base&&) = delete;

      operation_base<Result, ErrorVariant, Receiver>& parent_;
    };

    template <class Receiver>
    using local_env_t = exec::make_env_t<
      stdexec::env_of_t<Receiver>,
      exec::with_t<stdexec::get_stop_token_t, stdexec::inplace_stop_token>>;

    template <class Result, class ErrorVariant, class Receiver>
    struct local_receiver {
      using receiver_concept = stdexec::receiver_t;

      auto get_env() const noexcept -> local_env_t<Receiver> {
        return exec::make_env(
          stdexec::get_env(local_op_->parent_.receiver_),
          exec::with(stdexec::get_stop_token, local_op_->parent_.stop_source_.get_token()));
      }

      template <class... Args>
      void set_value(Args&&... result) && noexcept {
        if constexpr (sizeof...(Args) > 0) {
          local_op_->result_.emplace(std::forward<Args>(result)...);
        }
        local_op_->parent_.notify();
      }

      template <class Error>
      void set_error(Error&& error) && noexcept {
        local_op_->parent_.notify_error(std::forward<Error>(error));
      }

      void set_stopped() && noexcept {
        local_op_->parent_.notify_stopped();
      }

      local_operation_base<Result, ErrorVariant, Receiver>* local_op_;
    };

    template <class... Ts>
    using nullable_std_variant = std::variant<unit, Ts...>;

    template <class... Ts>
    using nullable_std_variant_for = stdexec::__minvoke<
      stdexec::__munique<stdexec::__q<nullable_std_variant>>,
      std::exception_ptr,
      std::remove_cvref_t<Ts>...>;

    template <class Sender, class Environment>
    struct traits {
      using Result = stdexec::__single_sender_value_t<Sender, Environment>;
      using ErrorVariant = stdexec::error_types_of_t<Sender, Environment, nullable_std_variant_for>;
    };

    template <class Sender, class Receiver>
    struct local_operation
      : local_operation_base<
          typename traits<Sender, stdexec::env_of_t<Receiver>>::Result,
          typename traits<Sender, stdexec::env_of_t<Receiver>>::ErrorVariant,
          Receiver> {
      using Result = typename traits<Sender, stdexec::env_of_t<Receiver>>::Result;
      using ErrorVariant = typename traits<Sender, stdexec::env_of_t<Receiver>>::ErrorVariant;

      local_operation(operation_base<Result, ErrorVariant, Receiver>& parent, Sender&& sndr) noexcept
        : local_operation_base<Result, ErrorVariant, Receiver>(parent)
        , child_op_(stdexec::connect(
            std::forward<Sender>(sndr),
            local_receiver<Result, ErrorVariant, Receiver>(this))) {
      }

      void start() noexcept {
        stdexec::start(child_op_);
      }

      stdexec::connect_result_t<Sender, local_receiver<Result, ErrorVariant, Receiver>> child_op_;
    };

    template <class Range, class Receiver>
    class operation
      : public operation_base<
          typename traits<std::ranges::range_reference_t<Range>, stdexec::env_of_t<Receiver>>::Result,
          typename traits<std::ranges::range_reference_t<Range>, stdexec::env_of_t<Receiver>>::
            ErrorVariant,
          Receiver> {
     public:
      using operation_state_concept = stdexec::operation_state_t;

      using Sender = std::ranges::range_value_t<Range>;

      using Result =
        typename traits<std::ranges::range_reference_t<Range>, stdexec::env_of_t<Receiver>>::Result;

      using ErrorVariant =
        typename traits<std::ranges::range_reference_t<Range>, stdexec::env_of_t<Receiver>>::
          ErrorVariant;

      explicit operation(Range range, Receiver receiver)
        : operation_base<Result, ErrorVariant, Receiver>(std::move(receiver)) {
        if constexpr (std::ranges::sized_range<Range>) {
          children_.reserve(std::ranges::size(range));
        }
        try {
          for (auto&& sndr: range) {
            children_.emplace_back(std::in_place, std::forward<decltype(sndr)>(sndr));
          }
        } catch (...) {
          for (auto& variant: children_) {
            variant.template destroy<0>();
          }
          throw;
        }
        std::size_t counter = 0;
        for (auto& variant: children_) {
          try {
            auto sndr = std::move(variant).template get<0>();
            variant.template emplace_from<1>([&] {
              return local_operation<Sender, Receiver>(*this, std::move(sndr));
            }); // NOLINT
          } catch (...) {
            std::size_t i = 0;
            for (; i < counter; ++i) {
              children_[i].template destroy<1>();
            }
            for (; i < children_.size(); ++i) {
              children_[i].template destroy<0>();
            }
            throw;
          }
          counter += 1;
        }
        this->count_.store(static_cast<std::ptrdiff_t>(counter));
      }

      void start() noexcept {
        this->do_start();
        for (auto& child: children_) {
          stdexec::start(child.template get<1>());
        }
      }

      std::vector<manual_alternative<Sender, local_operation<Sender, Receiver>>> children_;
    };

    template <class... Tps>
    using to_std_vector =
      stdexec::completion_signatures<stdexec::set_value_t(std::vector<std::remove_cvref_t<Tps>>)...>;

    template <class Range>
    class sender {
     public:
      using sender_concept = stdexec::sender_t;

      explicit sender(Range range) noexcept(std::is_nothrow_move_constructible_v<Range>)
        : range_(std::move(range)) {
      }

      template <class Self, class Receiver>
        requires stdexec::
          __single_value_sender<std::ranges::range_reference_t<Range>, stdexec::env_of_t<Receiver>>
        auto connect(this Self&& self, Receiver receiver) -> operation<Range, Receiver> {
        return operation<Range, Receiver>(std::forward<Self>(self).range_, std::move(receiver));
      }

      template <class Env>
        requires stdexec::__single_value_sender<std::ranges::range_reference_t<Range>, Env>
      auto get_completion_signatures(Env&&) const noexcept
        -> stdexec::transform_completion_signatures_of<
          std::ranges::range_value_t<Range>,
          Env,
          stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>> {
        return {};
      }

     private:
      Range range_;
    };
  } // namespace when_all_range_

  struct when_all_range_t {
    template <std::ranges::range Range>
    auto operator()(Range range) const
      noexcept(noexcept(when_all_range_::sender<Range>{std::move(range)}))
        -> when_all_range_::sender<Range> {
      return when_all_range_::sender<Range>{std::move(range)};
    }
  };

  inline constexpr when_all_range_t when_all_range{};

} // namespace exec