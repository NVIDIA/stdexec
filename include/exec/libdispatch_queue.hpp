/*
 * Copyright (c) 2024 Rishabh Dwivedi <rishabhdwivedi17@gmail.com>
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

#if __has_include(<dispatch/dispatch.h>)

// TODO: This is needed for libdispatch to compile with GCC. Need to look for
// workaround.
#  ifndef __has_feature
#    define __has_feature(x) false
#  endif
#  ifndef __has_extension
#    define __has_extension(x) false
#  endif

#  include "../stdexec/execution.hpp"
#  include <dispatch/dispatch.h>

namespace exec {
  struct libdispatch_queue;

  namespace __libdispatch {
    struct task_base {
      void (*execute)(task_base *) noexcept;
    };

    template <class Receiver>
    struct operation;

    template <class Sender, std::integral Shape, class Fun>
    struct bulk_sender;

    template <class CvSender, class Receiver, class Shape, class Fun, bool MayThrow>
    struct bulk_shared_state;

    template <class Fun, class Shape, class... Args>
      requires STDEXEC::__callable<Fun, Shape, STDEXEC::__decay_t<Args> &...>
    using bulk_non_throwing_t = STDEXEC::__mbool<
      STDEXEC::__nothrow_callable<Fun, Shape, STDEXEC::__decay_t<Args> &...>
      && STDEXEC::__nothrow_decay_copyable<Args...>
    >;

    template <class CvSender, class Receiver, class Shape, class Fun, bool MayThrow>
    struct bulk_receiver;

    template <class CvSender, class Receiver, std::integral Shape, class Fun>
    struct bulk_op_state;

    struct transform_bulk {
      template <class Data, class Sender>
      auto operator()(STDEXEC::bulk_t, Data &&data, Sender &&sndr) {
        auto [pol, shape, fun] = static_cast<Data &&>(data);
        // TODO: handle non-par execution policies
        return bulk_sender<STDEXEC::__decay_t<Sender>, decltype(shape), decltype(fun)>{
          queue_, static_cast<Sender &&>(sndr), shape, std::move(fun)};
      }

      libdispatch_queue &queue_;
    };
  } // namespace __libdispatch

  struct CANNOT_DISPATCH_THE_BULK_ALGORITHM_TO_THE_LIBDISPATCH_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_LIBDISPATCH_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_LIBDISPATCH_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  struct libdispatch_scheduler {
    bool operator==(libdispatch_scheduler const &) const = default;

    struct domain {
      // transform the generic bulk sender into a parallel libdispatch bulk sender
      template <STDEXEC::sender_expr_for<STDEXEC::bulk_t> Sender, class Env>
      auto transform_sender(STDEXEC::set_value_t, Sender &&sndr, const Env &env) const noexcept {
        if constexpr (STDEXEC::__completes_on<Sender, libdispatch_scheduler, Env>) {
          auto sched =
            STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(sndr), env);
          static_assert(std::is_same_v<decltype(sched), libdispatch_scheduler>);
          return STDEXEC::__apply(
            __libdispatch::transform_bulk{*sched.queue_}, std::forward<Sender>(sndr));
        } else {
          return STDEXEC::__not_a_sender<
            STDEXEC::_WHAT_(CANNOT_DISPATCH_THE_BULK_ALGORITHM_TO_THE_LIBDISPATCH_SCHEDULER),
            STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_LIBDISPATCH_SCHEDULER_IN_THE_ENVIRONMENT),
            STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, STDEXEC::bulk_t),
            STDEXEC::_TO_FIX_THIS_ERROR_(
              ADD_A_CONTINUES_ON_TRANSITION_TO_THE_LIBDISPATCH_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
            STDEXEC::_WITH_PRETTY_SENDER_<Sender>,
            STDEXEC::_WITH_ENVIRONMENT_(Env)
          >();
        }
      }
    };

    struct sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures =
        STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>;

      template <STDEXEC::receiver Receiver>
      auto connect(Receiver rcvr) const -> __libdispatch::operation<Receiver> {
        return __libdispatch::operation<Receiver>(queue, std::move(rcvr));
      }

      struct attrs {
        template <typename CPO>
        libdispatch_scheduler query(STDEXEC::get_completion_scheduler_t<CPO>) const noexcept {
          return libdispatch_scheduler{queue};
        }

        template <typename CPO>
        domain query(STDEXEC::get_completion_domain_t<CPO>) const noexcept {
          return {};
        }

        libdispatch_queue *queue;
      };

      auto get_env() const noexcept -> attrs {
        return attrs{queue};
      }

      libdispatch_queue *queue;
    };

    auto schedule() const noexcept -> sender {
      return sender{queue_};
    }

    auto query(STDEXEC::get_domain_t) const noexcept -> domain {
      return {};
    }

    template <typename CPO>
    libdispatch_scheduler query(STDEXEC::get_completion_scheduler_t<CPO>) const noexcept {
      return *this;
    }

    template <typename CPO>
    domain query(STDEXEC::get_completion_domain_t<CPO>) const noexcept {
      return {};
    }

    auto query(STDEXEC::get_forward_progress_guarantee_t) const noexcept
      -> STDEXEC::forward_progress_guarantee {
      return STDEXEC::forward_progress_guarantee::parallel;
    }

    libdispatch_queue *queue_;
  };

  struct libdispatch_queue {
    bool operator==(libdispatch_queue const &) const = default;

    void submit(__libdispatch::task_base *f) {
      auto queue = dispatch_get_global_queue(priority, 0);
      dispatch_async_f(queue, f, reinterpret_cast<void (*)(void *) noexcept>(f->execute));
    }

    auto get_scheduler() {
      return libdispatch_scheduler{this};
    }

    int priority{DISPATCH_QUEUE_PRIORITY_DEFAULT};
  };

  namespace __libdispatch {
    template <typename Receiver>
    struct operation : task_base {
      libdispatch_queue &queue;
      Receiver receiver;

      operation(libdispatch_queue *queue_arg, Receiver rcvr)
        : queue(*queue_arg)
        , receiver(std::move(rcvr)) {
        this->execute = [](task_base *t) noexcept {
          auto &op = *static_cast<operation *>(t);
          auto stoken = STDEXEC::get_stop_token(STDEXEC::get_env(op.receiver));
          if constexpr (STDEXEC::unstoppable_token<decltype(stoken)>) {
            STDEXEC::set_value(std::move(op.receiver));
          } else if (stoken.stop_requested()) {
            STDEXEC::set_stopped(std::move(op.receiver));
          } else {
            STDEXEC::set_value(std::move(op.receiver));
          }
        };
      }

      void enqueue(task_base *op) const {
        queue.submit(op);
      }

      void start() & noexcept {
        enqueue(this);
      }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // What follows is the implementation for parallel bulk execution on
    // libdispatch queue.
    template <class Sender, std::integral Shape, class Fun>
    struct bulk_sender {
      using sender_concept = STDEXEC::sender_t;

      template <class CvSender, class... Env>
      using with_error_invoke_t = STDEXEC::__if_c<
        STDEXEC::__value_types_t<
          STDEXEC::__completion_signatures_of_t<CvSender, Env...>,
          STDEXEC::__mbind_front_q<bulk_non_throwing_t, Fun, Shape>,
          STDEXEC::__q<STDEXEC::__mand>
        >::value,
        STDEXEC::completion_signatures<>,
        STDEXEC::__eptr_completion
      >;

      template <class... Tys>
      using set_value_t =
        STDEXEC::completion_signatures<STDEXEC::set_value_t(STDEXEC::__decay_t<Tys>...)>;

      template <class Self, class... Env>
      using _completions_t = STDEXEC::transform_completion_signatures<
        STDEXEC::__completion_signatures_of_t<STDEXEC::__copy_cvref_t<Self, Sender>, Env...>,
        with_error_invoke_t<STDEXEC::__copy_cvref_t<Self, Sender>, Env...>,
        set_value_t
      >;

      template <class Self, class Receiver>
      using bulk_op_state_t =
        bulk_op_state<STDEXEC::__copy_cvref_t<Self, Sender>, Receiver, Shape, Fun>;

      template <STDEXEC::__decays_to<bulk_sender> Self, STDEXEC::receiver Receiver>
        requires STDEXEC::receiver_of<Receiver, _completions_t<Self, STDEXEC::env_of_t<Receiver>>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self &&self, Receiver rcvr)
        noexcept(STDEXEC::__nothrow_constructible_from<
                 bulk_op_state_t<Self, Receiver>,
                 libdispatch_queue &,
                 Shape,
                 Fun,
                 Sender,
                 Receiver
        >) -> bulk_op_state_t<Self, Receiver> {
        return bulk_op_state_t<Self, Receiver>{
          self.queue_,
          self.shape_,
          self.fun_,
          std::forward<Self>(self).sndr_,
          std::forward<Receiver>(rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <STDEXEC::__decays_to<bulk_sender> Self, class... Env>
      static consteval auto get_completion_signatures() -> _completions_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> STDEXEC::env_of_t<const Sender &> {
        return STDEXEC::get_env(sndr_);
      }

      libdispatch_queue &queue_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;
    };

    template <class CvSender, class Receiver, class Shape, class Fun, bool MayThrow>
    struct bulk_shared_state {
      struct bulk_task : task_base {
        bulk_shared_state *sh_state_;
        Shape task_id_;

        explicit bulk_task(bulk_shared_state *sh_state_arg, Shape task_id_arg)
          : sh_state_(sh_state_arg)
          , task_id_(task_id_arg) {
          this->execute = [](task_base *t) noexcept {
            auto &sh_state = *static_cast<bulk_task *>(t)->sh_state_;
            auto task_id = static_cast<bulk_task *>(t)->task_id_;
            auto total_tasks = static_cast<std::uint32_t>(sh_state.num_tasks());

            auto computation = [&sh_state, task_id](auto &...args) {
              sh_state.fun_(task_id, args...);
            };

            auto completion = [&](auto &...args) {
              STDEXEC::set_value(std::move(sh_state.rcvr_), std::move(args)...);
            };

            if constexpr (MayThrow) {
              STDEXEC_TRY {
                sh_state.apply(computation);
              }
              STDEXEC_CATCH_ALL {
                std::uint32_t expected = total_tasks;

                if (sh_state.task_with_exception_.compare_exchange_strong(
                      expected,
                      static_cast<std::uint32_t>(task_id),
                      STDEXEC::__std::memory_order_relaxed,
                      STDEXEC::__std::memory_order_relaxed)) {
                  sh_state.exception_ = std::current_exception();
                }
              }

              const bool is_last_task = sh_state.finished_tasks_.fetch_add(1) == (total_tasks - 1);

              if (is_last_task) {
                if (sh_state.exception_) {
                  STDEXEC::set_error(std::move(sh_state.rcvr_), std::move(sh_state.exception_));
                } else {
                  sh_state.apply(completion);
                }
              }
            } else {
              sh_state.apply(computation);

              bool const is_last_task = sh_state.finished_tasks_.fetch_add(1) == (total_tasks - 1);

              if (is_last_task) {
                sh_state.apply(completion);
              }
            }
          };
        }
      };

      using variant_t = STDEXEC::__value_types_of_t<
        CvSender,
        STDEXEC::env_of_t<Receiver>,
        STDEXEC::__q<STDEXEC::__decayed_std_tuple>,
        STDEXEC::__q<STDEXEC::__std_variant>
      >;

      bulk_shared_state(Receiver rcvr, Shape shape, Fun fun)
        : rcvr_{std::move(rcvr)}
        , shape_{shape}
        , fun_{fun}
        , task_with_exception_{static_cast<std::uint32_t>(num_tasks())} {
      }

      Shape num_tasks() const {
        return shape_;
      }

      template <class F>
      void apply(F f) {
        std::visit(
          [&](auto &tupl) -> void { std::apply([&](auto &...args) -> void { f(args...); }, tupl); },
          data_);
      }

      variant_t data_;
      Receiver rcvr_;
      Shape shape_;
      Fun fun_;

      STDEXEC::__std::atomic<std::uint32_t> finished_tasks_{0};
      STDEXEC::__std::atomic<std::uint32_t> task_with_exception_{0};
      std::exception_ptr exception_;
      std::vector<bulk_task> tasks_;
    };

    template <class CvSender, class Receiver, class Shape, class Fun, bool MayThrow>
    struct bulk_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      using shared_state = bulk_shared_state<CvSender, Receiver, Shape, Fun, MayThrow>;

      void enqueue() noexcept {
        using bulk_task = shared_state::bulk_task;
        shared_state_.tasks_.reserve(static_cast<std::size_t>(shared_state_.shape_));
        for (Shape i{}; i != shared_state_.shape_; ++i) {
          shared_state_.tasks_.push_back(bulk_task(&shared_state_, i));
          queue_.submit(&(shared_state_.tasks_.back()));
        }
      }

      template <class... As>
      void set_value(As &&...as) noexcept {
        using tuple_t = STDEXEC::__decayed_std_tuple<As...>;

        if constexpr (MayThrow) {
          STDEXEC_TRY {
            shared_state_.data_.template emplace<tuple_t>(std::move(as)...);
          }
          STDEXEC_CATCH_ALL {
            STDEXEC::set_error(std::move(shared_state_.rcvr_), std::current_exception());
          }
        } else {
          shared_state_.data_.template emplace<tuple_t>(std::move(as)...);
        }

        if (shared_state_.shape_) {
          enqueue();
        } else {
          shared_state_.apply([&](auto &...args) {
            STDEXEC::set_value(std::move(shared_state_.rcvr_), std::move(args)...);
          });
        }
      }

      template <class Error>
      void set_error(Error &&error) noexcept {
        STDEXEC::set_error(std::move(shared_state_.rcvr_), static_cast<Error &&>(error));
      }

      void set_stopped() noexcept {
        STDEXEC::set_stopped(std::move(shared_state_.rcvr_));
      }

      auto get_env() const noexcept -> STDEXEC::env_of_t<Receiver> {
        return STDEXEC::get_env(shared_state_.rcvr_);
      }

      shared_state &shared_state_;
      libdispatch_queue &queue_;
    };

    template <class CvSender, class Receiver, std::integral Shape, class Fun>
    struct bulk_op_state {
      static constexpr bool may_throw = !STDEXEC::__value_types_of_t<
        CvSender,
        STDEXEC::env_of_t<Receiver>,
        STDEXEC::__mbind_front_q<bulk_non_throwing_t, Fun, Shape>,
        STDEXEC::__q<STDEXEC::__mand>
      >::value;

      using bulk_rcvr = bulk_receiver<CvSender, Receiver, Shape, Fun, may_throw>;
      using shared_state = bulk_shared_state<CvSender, Receiver, Shape, Fun, may_throw>;
      using inner_op_state = STDEXEC::connect_result_t<CvSender, bulk_rcvr>;

      bulk_op_state(libdispatch_queue &queue, Shape shape, Fun fun, CvSender &&sndr, Receiver rcvr)
        : shared_state_(std::move(rcvr), shape, fun)
        , inner_op_{STDEXEC::connect(std::move(sndr), bulk_rcvr{shared_state_, queue})} {
      }

      void start() & noexcept {
        STDEXEC::start(inner_op_);
      }

      shared_state shared_state_;
      inner_op_state inner_op_;
    };
  } // namespace __libdispatch
} // namespace exec

#endif // __has_include(<dispatch/dispatch.h>)
