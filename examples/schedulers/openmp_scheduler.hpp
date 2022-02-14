/*
 * Copyright (c) NVIDIA
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

#include <execution.hpp>
#include <type_traits>
#include <exception>

namespace example {

  namespace detail::openmp_scheduler::bulk {
    template <class R, class ShapeT, class F>
    struct receiver_t
      : std::execution::receiver_adaptor<receiver_t<R, ShapeT, F>, R>
    {
      ShapeT shape_;
      F function_;

      receiver_t(R receiver, ShapeT shape, F function)
          : std::execution::receiver_adaptor<receiver_t<R, ShapeT, F>, R>(
                std::forward<R>(receiver)),
            shape_(shape), function_(function) {}

      template <class... Ts>
      void set_value(Ts &&...values) && noexcept try {
        #pragma omp parallel for
        for (ShapeT i = {}; i < shape_; i++) {
          function_(i, std::forward<Ts>(values)...);
        }

        std::execution::set_value(std::move(this->base()));
      } catch(...) {
        std::execution::set_error(
            std::move(this->base()), std::current_exception());
      }

      void set_error(std::exception_ptr) && noexcept {
        std::execution::set_error(
            std::move(this->base()), std::current_exception());
      }

      void set_stopped() && noexcept {
        std::execution::set_stopped(std::move(this->base()));
      }

      friend auto tag_invoke(std::execution::get_env_t, const receiver_t &self)
          -> std::execution::env_of_t<R> {
        return std::execution::get_env(self.base());
      }
    };

    template <class S, class Shape, class F>
    struct sender_t : std::execution::sender_adaptor<sender_t<S, Shape, F>, S>
    {
      Shape shape_;
      F function_;

      template <class Receiver>
      auto connect(Receiver &&r) && noexcept
      {
        return std::execution::connect(
            std::move(this->base()),
            receiver_t<Receiver, Shape, F>{std::forward<Receiver>(r), shape_,
                                           function_});
      }

      template <class _CPO>
      friend auto tag_invoke(std::execution::get_completion_scheduler_t<_CPO>,
                             const sender_t &self) noexcept
      {
        return std::execution::get_completion_scheduler<_CPO>(self.base());
      }

      template <std::__decays_to<sender_t> Self, class _Env>
      friend auto tag_invoke(std::execution::get_completion_signatures_t,
                             Self &&, _Env)
          -> std::execution::completion_signatures_of_t<
              std::__member_t<Self, S>, _Env>;

      explicit sender_t(S sender, Shape shape, F function)
        : std::execution::sender_adaptor<sender_t<S, Shape, F>, S>(std::forward<S>(sender))
        , shape_(shape)
        , function_(function)
      {}
    };
  } // namespace detail::bulk

  // A simple scheduler that executes its continuation inline, on the
  // thread of the caller of start().
  struct openmp_scheduler {
    template <class R_>
      struct __op {
        using R = std::__t<R_>;
        [[no_unique_address]] R rec_;
        friend void tag_invoke(std::execution::start_t, __op& op) noexcept try {
          std::execution::set_value((R&&) op.rec_);
        } catch(...) {
          std::execution::set_error((R&&) op.rec_, std::current_exception());
        }
      };

    struct __sender {
      using completion_signatures =
        std::execution::completion_signatures<
          std::execution::set_value_t(),
          std::execution::set_error_t(std::exception_ptr)>;

      template <std::execution::receiver R>
        friend auto tag_invoke(std::execution::connect_t, __sender, R&& rec)
          noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
          -> __op<std::__x<std::remove_cvref_t<R>>> {
          return {(R&&) rec};
        }

      friend openmp_scheduler
      tag_invoke(std::execution::get_completion_scheduler_t<std::execution::set_value_t>, __sender) noexcept {
        return {};
      }
    };

    friend __sender tag_invoke(std::execution::schedule_t, openmp_scheduler) noexcept {
      return {};
    }

    template <class S, std::integral Shape, class F>
    friend auto
    tag_invoke(std::execution::bulk_t,
               const openmp_scheduler &,
               S &&self,
               Shape shape,
               F f)
      noexcept {
      return detail::openmp_scheduler::bulk::sender_t<S, Shape, F>{
          std::forward<S>(self), std::forward<Shape>(shape), f};
    }

    friend std::execution::forward_progress_guarantee tag_invoke(
        std::execution::get_forward_progress_guarantee_t,
        const openmp_scheduler&) noexcept {
      return std::execution::forward_progress_guarantee::weakly_parallel;
    }

    bool operator==(const openmp_scheduler&) const noexcept = default;
  };
}
