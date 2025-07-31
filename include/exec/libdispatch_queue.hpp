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

  namespace __libdispatch_details {
    using namespace stdexec::tags;

    template <class>
    struct not_a_sender {
      using sender_concept = stdexec::sender_t;
    };

    struct task_base {
      void (*execute)(task_base *) noexcept;
    };

    template <typename ReceiverId>
    struct operation;
  } // namespace __libdispatch_details

  namespace __libdispatch_bulk {
    using namespace stdexec::tags;

    template <class SenderId, std::integral Shape, class Fun>
    struct bulk_sender {
      using Sender = stdexec::__t<SenderId>;
      struct __t;
    };

    template <stdexec::sender Sender, std::integral Shape, class Fun>
    using bulk_sender_t =
      stdexec::__t<bulk_sender<stdexec::__id<stdexec::__decay_t<Sender>>, Shape, Fun>>;

    template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
    struct bulk_shared_state;

    template <class Fun, class Shape, class... Args>
      requires stdexec::__callable<Fun, Shape, Args &...>
    using bulk_non_throwing = stdexec::__mbool<
      stdexec::__nothrow_callable<Fun, Shape, Args &...>
      && noexcept(stdexec::__decayed_std_tuple<Args...>(std::declval<Args>()...))
    >;

    template <class CvrefSenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
    struct bulk_receiver {
      using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      struct __t;
    };

    template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
    using bulk_receiver_t = stdexec::__t<
      bulk_receiver<stdexec::__cvref_id<CvrefSender>, stdexec::__id<Receiver>, Shape, Fun, MayThrow>
    >;

    template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
    struct bulk_op_state {
      using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
      using Receiver = stdexec::__t<ReceiverId>;
      struct __t;
    };

    template <class Sender, class Receiver, std::integral Shape, class Fun>
    using bulk_op_state_t = stdexec::__t<bulk_op_state<
      stdexec::__id<stdexec::__decay_t<Sender>>,
      stdexec::__id<stdexec::__decay_t<Receiver>>,
      Shape,
      Fun
    >>;

    struct transform_bulk {
      template <class Data, class Sender>
      auto operator()(stdexec::bulk_t, Data &&data, Sender &&sndr) {
        auto [pol, shape, fun] = std::forward<Data>(data);
        // TODO: handle non-par execution policies
        return bulk_sender_t<Sender, decltype(shape), decltype(fun)>{
          queue_, std::forward<Sender>(sndr), shape, std::move(fun)};
      }

      libdispatch_queue &queue_;
    };
  } // namespace __libdispatch_bulk

  struct libdispatch_scheduler {
    using __t = libdispatch_scheduler;
    using __id = libdispatch_scheduler;
    bool operator==(libdispatch_scheduler const &) const = default;

    struct domain {
      // For eager customization
      template <stdexec::sender_expr_for<stdexec::bulk_t> Sender>
      auto transform_sender(Sender &&sndr) const noexcept {
        if constexpr (stdexec::__completes_on<Sender, libdispatch_scheduler>) {
          auto sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(
            stdexec::get_env(sndr));
          return stdexec::__sexpr_apply(
            std::forward<Sender>(sndr), __libdispatch_bulk::transform_bulk{*sched.queue_});
        } else {
          static_assert(
            stdexec::__completes_on<Sender, libdispatch_scheduler>,
            "No libdispatch_queue instance can be found in the sender's "
            "attributes on which to schedule bulk work.");
          return __libdispatch_details::not_a_sender<stdexec::__name_of<Sender>>();
        }
      }

      // transform the generic bulk sender into a parallel libdispatch bulk sender
      template <stdexec::sender_expr_for<stdexec::bulk_t> Sender, class Env>
      auto transform_sender(Sender &&sndr, const Env &env) const noexcept {
        if constexpr (stdexec::__starts_on<Sender, libdispatch_scheduler, Env>) {
          auto sched = stdexec::get_scheduler(env);
          return stdexec::__sexpr_apply(
            std::forward<Sender>(sndr), __libdispatch_bulk::transform_bulk{*sched.queue_});
        } else {
          static_assert(
            stdexec::__starts_on<Sender, libdispatch_scheduler, Env>,
            "No libdispatch_queue instance can be found in the receiver's "
            "environment on which to schedule bulk work.");
          return __libdispatch_details::not_a_sender<stdexec::__name_of<Sender>>();
        }
      }
    };

    struct sender {
      using __t = sender;
      using __id = sender;
      using sender_concept = stdexec::sender_t;
      using completion_signatures =
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

      template <stdexec::receiver Receiver>
      auto connect(Receiver r) const -> __libdispatch_details::operation<stdexec::__id<Receiver>> {
        return __libdispatch_details::operation<stdexec::__id<Receiver>>(queue, std::move(r));
      }

      struct env {
        libdispatch_queue *queue;

        template <typename CPO>
        libdispatch_scheduler query(stdexec::get_completion_scheduler_t<CPO>) const noexcept {
          return libdispatch_scheduler{queue};
        }
      };

      STDEXEC_MEMFN_FRIEND(get_env);

      env get_env() const noexcept {
        return env{queue};
      }

      libdispatch_queue *queue;
    };

    auto schedule() const noexcept -> sender {
      return sender{queue_};
    }

    auto query(stdexec::get_domain_t) const noexcept -> domain {
      return {};
    }

    auto query(stdexec::get_forward_progress_guarantee_t) const noexcept
      -> stdexec::forward_progress_guarantee {
      return stdexec::forward_progress_guarantee::parallel;
    }

    libdispatch_queue *queue_;
  };

  struct libdispatch_queue {
    bool operator==(libdispatch_queue const &) const = default;

    void submit(__libdispatch_details::task_base *f) {
      auto queue = dispatch_get_global_queue(priority, 0);
      dispatch_async_f(queue, f, reinterpret_cast<void (*)(void *) noexcept>(f->execute));
    }

    auto get_scheduler() {
      return libdispatch_scheduler{this};
    }

    int priority{DISPATCH_QUEUE_PRIORITY_DEFAULT};
  };

  template <typename ReceiverId>
  struct __libdispatch_details::operation : public __libdispatch_details::task_base {
    using Receiver = stdexec::__t<ReceiverId>;
    libdispatch_queue &queue;
    Receiver receiver;

    operation(libdispatch_queue *queue_arg, Receiver r)
      : queue(*queue_arg)
      , receiver(std::move(r)) {
      this->execute = [](__libdispatch_details::task_base *t) noexcept {
        auto &op = *static_cast<operation *>(t);
        auto stoken = stdexec::get_stop_token(stdexec::get_env(op.receiver));
        if constexpr (stdexec::unstoppable_token<decltype(stoken)>) {
          stdexec::set_value(std::move(op.receiver));
        } else if (stoken.stop_requested()) {
          stdexec::set_stopped(std::move(op.receiver));
        } else {
          stdexec::set_value(std::move(op.receiver));
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
  template <class SenderId, std::integral Shape, class Fun>
  struct __libdispatch_bulk::bulk_sender<SenderId, Shape, Fun>::__t {
    using __id = bulk_sender;
    using sender_concept = stdexec::sender_t;

    libdispatch_queue &queue_;
    Sender sndr_;
    Shape shape_;
    Fun fun_;

    template <class Sender, class... Env>
    using with_error_invoke_t = stdexec::__if_c<
      stdexec::__v<stdexec::__value_types_t<
        stdexec::__completion_signatures_of_t<Sender, Env...>,
        stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
        stdexec::__q<stdexec::__mand>
      >>,
      stdexec::completion_signatures<>,
      stdexec::__eptr_completion
    >;

    template <class... Tys>
    using set_value_t =
      stdexec::completion_signatures<stdexec::set_value_t(stdexec::__decay_t<Tys>...)>;

    template <class Self, class... Env>
    using __completions_t = stdexec::transform_completion_signatures<
      stdexec::__completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sender>, Env...>,
      with_error_invoke_t<stdexec::__copy_cvref_t<Self, Sender>, Env...>,
      set_value_t
    >;

    template <class Self, class Receiver>
    using bulk_op_state_t = stdexec::__t<
      bulk_op_state<stdexec::__cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun>
    >;

    template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
      requires stdexec::receiver_of<Receiver, __completions_t<Self, stdexec::env_of_t<Receiver>>>
    static bulk_op_state_t<Self, Receiver>
      connect(Self &&self, Receiver rcvr) noexcept(stdexec::__nothrow_constructible_from<
                                                   bulk_op_state_t<Self, Receiver>,
                                                   libdispatch_queue &,
                                                   Shape,
                                                   Fun,
                                                   Sender,
                                                   Receiver
      >) {
      return bulk_op_state_t<Self, Receiver>{
        self.queue_,
        self.shape_,
        self.fun_,
        (std::forward<Self>(self)).sndr_,
        (std::forward<Receiver>(rcvr))};
    }

    template <stdexec::__decays_to<__t> Self, class... Env>
    static auto get_completion_signatures(Self &&, Env &&...) -> __completions_t<Self, Env...> {
      return {};
    }

    auto get_env() const noexcept -> stdexec::env_of_t<const Sender &> {
      return stdexec::get_env(sndr_);
    }
  };

  template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
  struct __libdispatch_bulk::bulk_shared_state {
    struct bulk_task : __libdispatch_details::task_base {
      bulk_shared_state *sh_state_;
      Shape task_id_;

      bulk_task(bulk_shared_state *sh_state_arg, Shape task_id_arg)
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
            stdexec::set_value(std::move(sh_state.rcvr_), std::move(args)...);
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
                    std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                sh_state.exception_ = std::current_exception();
              }
            }

            const bool is_last_task = sh_state.finished_tasks_.fetch_add(1) == (total_tasks - 1);

            if (is_last_task) {
              if (sh_state.exception_) {
                stdexec::set_error(std::move(sh_state.rcvr_), std::move(sh_state.exception_));
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

    using variant_t = stdexec::__value_types_of_t<
      CvrefSender,
      stdexec::env_of_t<Receiver>,
      stdexec::__q<stdexec::__decayed_std_tuple>,
      stdexec::__q<stdexec::__std_variant>
    >;

    variant_t data_;
    Receiver rcvr_;
    Shape shape_;
    Fun fun_;

    std::atomic<std::uint32_t> finished_tasks_{0};
    std::atomic<std::uint32_t> task_with_exception_{0};
    std::exception_ptr exception_;
    std::vector<bulk_task> tasks_;

    Shape num_tasks() const {
      return shape_;
    }

    template <class F>
    void apply(F f) {
      std::visit(
        [&](auto &tupl) -> void { std::apply([&](auto &...args) -> void { f(args...); }, tupl); },
        data_);
    }

    bulk_shared_state(Receiver rcvr, Shape shape, Fun fun)
      : rcvr_{std::move(rcvr)}
      , shape_{shape}
      , fun_{fun}
      , task_with_exception_{static_cast<std::uint32_t>(num_tasks())} {
    }
  };

  template <class CvrefSenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
  struct __libdispatch_bulk::bulk_receiver<CvrefSenderId, ReceiverId, Shape, Fun, MayThrow>::__t {
    using __id = bulk_receiver;
    using receiver_concept = stdexec::receiver_t;

    using shared_state = bulk_shared_state<CvrefSender, Receiver, Shape, Fun, MayThrow>;

    shared_state &shared_state_;
    libdispatch_queue &queue_;

    void enqueue() noexcept {
      using bulk_task = typename shared_state::bulk_task;
      shared_state_.tasks_.reserve(static_cast<std::size_t>(shared_state_.shape_));
      for (Shape i{}; i != shared_state_.shape_; ++i) {
        shared_state_.tasks_.push_back(bulk_task(&shared_state_, i));
        queue_.submit(&(shared_state_.tasks_.back()));
      }
    }

    template <class... As>
    void set_value(As &&...as) noexcept {
      using tuple_t = stdexec::__decayed_std_tuple<As...>;

      if constexpr (MayThrow) {
        STDEXEC_TRY {
          shared_state_.data_.template emplace<tuple_t>(std::move(as)...);
        }
        STDEXEC_CATCH_ALL {
          stdexec::set_error(std::move(shared_state_.rcvr_), std::current_exception());
        }
      } else {
        shared_state_.data_.template emplace<tuple_t>(std::move(as)...);
      }

      if (shared_state_.shape_) {
        enqueue();
      } else {
        shared_state_.apply([&](auto &...args) {
          stdexec::set_value(std::move(shared_state_.rcvr_), std::move(args)...);
        });
      }
    }

    template <class Error>
    void set_error(Error &&error) noexcept {
      stdexec::set_error(std::move(shared_state_.rcvr_), static_cast<Error &&>(error));
    }

    void set_stopped() noexcept {
      stdexec::set_stopped(std::move(shared_state_.rcvr_));
    }

    auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
      return stdexec::get_env(shared_state_.rcvr_);
    }
  };

  template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
  struct __libdispatch_bulk::bulk_op_state<CvrefSenderId, ReceiverId, Shape, Fun>::__t {
    using __id = bulk_op_state;

    static constexpr bool may_throw = !stdexec::__v<stdexec::__value_types_of_t<
      CvrefSender,
      stdexec::env_of_t<Receiver>,
      stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
      stdexec::__q<stdexec::__mand>
    >>;

    using bulk_rcvr = bulk_receiver_t<CvrefSender, Receiver, Shape, Fun, may_throw>;
    using shared_state = bulk_shared_state<CvrefSender, Receiver, Shape, Fun, may_throw>;
    using inner_op_state = stdexec::connect_result_t<CvrefSender, bulk_rcvr>;

    shared_state shared_state_;

    inner_op_state inner_op_;

    void start() & noexcept {
      stdexec::start(inner_op_);
    }

    __t(libdispatch_queue &queue, Shape shape, Fun fun, CvrefSender &&sndr, Receiver rcvr)
      : shared_state_(std::move(rcvr), shape, fun)
      , inner_op_{stdexec::connect(std::move(sndr), bulk_rcvr{shared_state_, queue})} {
    }
  };

} // namespace exec

#endif // __has_include(<dispatch/dispatch.h>)
