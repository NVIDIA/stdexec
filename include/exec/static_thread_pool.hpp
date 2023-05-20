/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates.
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "../stdexec/execution.hpp"
#include "../stdexec/__detail/__config.hpp"
#include "../stdexec/__detail/__intrusive_queue.hpp"
#include "../stdexec/__detail/__meta.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace exec {
  using stdexec::__intrusive_queue;

  struct task_base {
    task_base* next;
    void (*__execute)(task_base*, std::uint32_t tid) noexcept;
  };

  template <typename ReceiverID>
  class operation;

  class static_thread_pool {
    template <typename ReceiverId>
    friend class operation;
   public:
    static_thread_pool();
    static_thread_pool(std::uint32_t threadCount);
    ~static_thread_pool();

    struct scheduler {
      using __t = scheduler;
      using __id = scheduler;
      bool operator==(const scheduler&) const = default;

     private:
      template <typename ReceiverId>
      friend class operation;

      class sender {
       public:
        using __t = sender;
        using __id = sender;
        using is_sender = void;
        using completion_signatures =
          stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t()>;
       private:
        template <typename Receiver>
        auto make_operation_(Receiver r) const -> operation<stdexec::__id<Receiver>> {
          return operation<stdexec::__id<Receiver>>{pool_, (Receiver&&) r};
        }

        template <stdexec::receiver Receiver>
        friend auto tag_invoke(stdexec::connect_t, sender s, Receiver r)
          -> operation<stdexec::__id<Receiver>> {
          return s.make_operation_((Receiver&&) r);
        }

        struct env {
          static_thread_pool& pool_;

          template <class CPO>
          friend static_thread_pool::scheduler
            tag_invoke(stdexec::get_completion_scheduler_t<CPO>, const env& self) noexcept {
            return self.make_scheduler_();
          }

          static_thread_pool::scheduler make_scheduler_() const {
            return static_thread_pool::scheduler{pool_};
          }
        };

        friend env tag_invoke(stdexec::get_env_t, const sender& self) noexcept {
          return env{self.pool_};
        }

        friend struct static_thread_pool::scheduler;

        explicit sender(static_thread_pool& pool) noexcept
          : pool_(pool) {
        }

        static_thread_pool& pool_;
      };

      sender make_sender_() const {
        return sender{*pool_};
      }

      template <class Fun, class Shape, class... Args>
        requires stdexec::__callable<Fun, Shape, Args&...>
      using bulk_non_throwing = //
        stdexec::__mbool<
          // If function invocation doesn't throw
          stdexec::__nothrow_callable<Fun, Shape, Args&...> &&
          // and emplacing a tuple doesn't throw
          noexcept(stdexec::__decayed_tuple<Args...>(std::declval<Args>()...))
          // there's no need to advertise completion with `exception_ptr`
          >;

      template <class SenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
      struct bulk_shared_state {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct bulk_task : task_base {
          bulk_shared_state* sh_state_;

          bulk_task(bulk_shared_state* sh_state)
            : sh_state_(sh_state) {
            this->__execute = [](task_base* t, const std::uint32_t tid) noexcept {
              auto& sh_state = *static_cast<bulk_task*>(t)->sh_state_;
              auto total_threads = sh_state.num_agents_required();

              auto computation = [&](auto&... args) {
                auto [begin, end] = even_share(sh_state.shape_, tid, total_threads);
                for (Shape i = begin; i < end; ++i) {
                  sh_state.fn_(i, args...);
                }
              };

              auto completion = [&](auto&... args) {
                stdexec::set_value((Receiver&&) sh_state.receiver_, std::move(args)...);
              };

              if constexpr (MayThrow) {
                try {
                  sh_state.apply(computation);
                } catch (...) {
                  std::uint32_t expected = total_threads;

                  if (sh_state.thread_with_exception_.compare_exchange_strong(
                        expected, tid, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    sh_state.exception_ = std::current_exception();
                  }
                }

                const bool is_last_thread = sh_state.finished_threads_.fetch_add(1)
                                         == (total_threads - 1);

                if (is_last_thread) {
                  if (sh_state.exception_) {
                    stdexec::set_error(
                      (Receiver&&) sh_state.receiver_, std::move(sh_state.exception_));
                  } else {
                    sh_state.apply(completion);
                  }
                }
              } else {
                sh_state.apply(computation);

                const bool is_last_thread = sh_state.finished_threads_.fetch_add(1)
                                         == (total_threads - 1);

                if (is_last_thread) {
                  sh_state.apply(completion);
                }
              }
            };
          }
        };

        using variant_t = //
          stdexec::__value_types_of_t<
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__q<stdexec::__decayed_tuple>,
            stdexec::__q<stdexec::__variant>>;

        variant_t data_;
        static_thread_pool& pool_;
        Receiver receiver_;
        Shape shape_;
        Fun fn_;

        std::atomic<std::uint32_t> finished_threads_{0};
        std::atomic<std::uint32_t> thread_with_exception_{0};
        std::exception_ptr exception_;
        std::vector<bulk_task> tasks_;

        // Splits `n` into `size` chunks distributing `n % size` evenly between ranks.
        // Returns `[begin, end)` range in `n` for a given `rank`.
        // Example:
        // ```cpp
        // //         n_items  thread  n_threads
        // even_share(     11,      0,         3); // -> [0,  4) -> 4 items
        // even_share(     11,      1,         3); // -> [4,  8) -> 4 items
        // even_share(     11,      2,         3); // -> [8, 11) -> 3 items
        // ```
        static std::pair<Shape, Shape>
          even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept {
          const auto avg_per_thread = n / size;
          const auto n_big_share = avg_per_thread + 1;
          const auto big_shares = n % size;
          const auto is_big_share = rank < big_shares;
          const auto begin = is_big_share
                             ? n_big_share * rank
                             : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
          const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

          return std::make_pair(begin, end);
        }

        std::uint32_t num_agents_required() const {
          return std::min(shape_, static_cast<Shape>(pool_.available_parallelism()));
        }

        template <class F>
        void apply(F f) {
          std::visit(
            [&](auto& tupl) -> void {
              std::apply([&](auto&... args) -> void { f(args...); }, tupl);
            },
            data_);
        }

        bulk_shared_state(static_thread_pool& pool, Receiver receiver, Shape shape, Fun fn)
          : pool_{pool}
          , receiver_{(Receiver&&) receiver}
          , shape_{shape}
          , fn_{fn}
          , thread_with_exception_{num_agents_required()}
          , tasks_{num_agents_required(), {this}} {
        }
      };

      template <class SenderId, class ReceiverId, class Shape, class Fn, bool MayThrow>
      struct bulk_receiver {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        using shared_state = bulk_shared_state<SenderId, ReceiverId, Shape, Fn, MayThrow>;

        shared_state& shared_state_;

        void enqueue() noexcept {
          shared_state_.pool_.bulk_enqueue(
            shared_state_.tasks_.data(), shared_state_.num_agents_required());
        }

        template <class... As>
        friend void tag_invoke(
          stdexec::same_as<stdexec::set_value_t> auto,
          bulk_receiver&& self,
          As&&... as) noexcept {
          using tuple_t = stdexec::__decayed_tuple<As...>;

          shared_state& state = self.shared_state_;

          if constexpr (MayThrow) {
            try {
              state.data_.template emplace<tuple_t>((As&&) as...);
            } catch (...) {
              stdexec::set_error(std::move(state.receiver_), std::current_exception());
            }
          } else {
            state.data_.template emplace<tuple_t>((As&&) as...);
          }

          if (state.shape_) {
            self.enqueue();
          } else {
            state.apply([&](auto&... args) {
              stdexec::set_value(std::move(state.receiver_), std::move(args)...);
            });
          }
        }

        template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
        friend void tag_invoke(Tag tag, bulk_receiver&& self, As&&... as) noexcept {
          shared_state& state = self.shared_state_;
          tag((Receiver&&) state.receiver_, (As&&) as...);
        }

        friend auto tag_invoke(stdexec::get_env_t, const bulk_receiver& self) noexcept
          -> stdexec::env_of_t<Receiver> {
          return stdexec::get_env(self.shared_state_.receiver_);
        }
      };

      template <class SenderId, class ReceiverId, std::integral Shape, class Fun>
      struct bulk_op_state {
        using Sender = stdexec::__t<SenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        static constexpr bool may_throw = //
          !stdexec::__v<stdexec::__value_types_of_t<
            Sender,
            stdexec::env_of_t<Receiver>,
            stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
            stdexec::__q<stdexec::__mand>>>;

        using bulk_rcvr = bulk_receiver<SenderId, ReceiverId, Shape, Fun, may_throw>;
        using shared_state = bulk_shared_state<SenderId, ReceiverId, Shape, Fun, may_throw>;
        using inner_op_state = stdexec::connect_result_t<Sender, bulk_rcvr>;

        shared_state shared_state_;

        inner_op_state inner_op_;

        friend void tag_invoke(stdexec::start_t, bulk_op_state& op) noexcept {
          stdexec::start(op.inner_op_);
        }

        bulk_op_state(
          static_thread_pool& pool,
          Shape shape,
          Fun fn,
          Sender&& sender,
          Receiver receiver)
          : shared_state_(pool, (Receiver&&) receiver, shape, fn)
          , inner_op_{stdexec::connect((Sender&&) sender, bulk_rcvr{shared_state_})} {
        }
      };

      template <class SenderId, std::integral Shape, class FunId>
      struct bulk_sender {
        using Sender = stdexec::__t<SenderId>;
        using Fun = stdexec::__t<FunId>;
        using is_sender = void;

        static_thread_pool& pool_;
        Sender sndr_;
        Shape shape_;
        Fun fun_;

        template <class Fun, class Sender, class Env>
        using with_error_invoke_t = //
          stdexec::__if_c<
            stdexec::__v<stdexec::__value_types_of_t<
              Sender,
              Env,
              stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
              stdexec::__q<stdexec::__mand>>>,
            stdexec::completion_signatures<>,
            stdexec::__with_exception_ptr>;

        template <class... Tys>
        using set_value_t =
          stdexec::completion_signatures< stdexec::set_value_t(stdexec::__decay_t<Tys>...)>;

        template <class Self, class Env>
        using completion_signatures = //
          stdexec::__try_make_completion_signatures<
            stdexec::__copy_cvref_t<Self, Sender>,
            Env,
            with_error_invoke_t<Fun, stdexec::__copy_cvref_t<Self, Sender>, Env>,
            stdexec::__q<set_value_t>>;

        template <class Self, class Receiver>
        using bulk_op_state_t = //
          bulk_op_state<
            stdexec::__x<stdexec::__copy_cvref_t<Self, Sender>>,
            stdexec::__x<stdexec::__decay_t<Receiver>>,
            Shape,
            Fun>;

        template <stdexec::__decays_to<bulk_sender> Self, stdexec::receiver Receiver>
          requires stdexec::
            receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
          friend bulk_op_state_t<Self, Receiver>                       //
          tag_invoke(stdexec::connect_t, Self&& self, Receiver&& rcvr) //
          noexcept(stdexec::__nothrow_constructible_from<
                   bulk_op_state_t<Self, Receiver>,
                   static_thread_pool&,
                   Shape,
                   Fun,
                   Sender,
                   Receiver>) {
          return bulk_op_state_t<Self, Receiver>{
            self.pool_, self.shape_, self.fun_, ((Self&&) self).sndr_, (Receiver&&) rcvr};
        }

        template <stdexec::__decays_to<bulk_sender> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env&&)
          -> stdexec::dependent_completion_signatures<Env>;

        template <stdexec::__decays_to<bulk_sender> Self, class Env>
        friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env&&)
          -> completion_signatures<Self, Env>
          requires true;

        friend auto tag_invoke(stdexec::get_env_t, const bulk_sender& self) noexcept
          -> stdexec::env_of_t<const Sender&> {
          return stdexec::get_env(self.sndr_);
        }
      };

      friend sender tag_invoke(stdexec::schedule_t, const scheduler& s) noexcept {
        return s.make_sender_();
      }

      template <stdexec::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_t = //
        bulk_sender<
          stdexec::__x<stdexec::__decay_t<Sender>>,
          Shape,
          stdexec::__x<stdexec::__decay_t<Fun>>>;

      template <stdexec::sender S, std::integral Shape, class Fn>
      friend bulk_sender_t<S, Shape, Fn>
        tag_invoke(stdexec::bulk_t, const scheduler& sch, S&& sndr, Shape shape, Fn fun) noexcept {
        return bulk_sender_t<S, Shape, Fn>{*sch.pool_, (S&&) sndr, shape, (Fn&&) fun};
      }

      friend stdexec::forward_progress_guarantee
        tag_invoke(stdexec::get_forward_progress_guarantee_t, const static_thread_pool&) noexcept {
        return stdexec::forward_progress_guarantee::parallel;
      }

      friend class static_thread_pool;

      explicit scheduler(static_thread_pool& pool) noexcept
        : pool_(&pool) {
      }

      static_thread_pool* pool_;
    };

    scheduler get_scheduler() noexcept {
      return scheduler{*this};
    }

    void request_stop() noexcept;

    std::uint32_t available_parallelism() const {
      return threadCount_;
    }

   private:
    class thread_state {
     public:
      task_base* try_pop();
      task_base* pop();
      bool try_push(task_base* task);
      void push(task_base* task);
      void request_stop();

     private:
      std::mutex mut_;
      std::condition_variable cv_;
      __intrusive_queue<&task_base::next> queue_;
      bool stopRequested_ = false;
    };

    void run(std::uint32_t index) noexcept;
    void join() noexcept;

    void enqueue(task_base* task) noexcept;

    template <std::derived_from<task_base> TaskT>
    void bulk_enqueue(TaskT* task, std::uint32_t n_threads) noexcept;

    std::uint32_t threadCount_;
    std::vector<std::thread> threads_;
    std::vector<thread_state> threadStates_;
    std::atomic<std::uint32_t> nextThread_;
  };

  template <typename ReceiverId>
  class operation : task_base {
    using Receiver = stdexec::__t<ReceiverId>;
    friend static_thread_pool::scheduler::sender;

    static_thread_pool& pool_;
    Receiver receiver_;

    explicit operation(static_thread_pool& pool, Receiver&& r)
      : pool_(pool)
      , receiver_((Receiver&&) r) {
      this->__execute = [](task_base* t, const std::uint32_t /* tid */) noexcept {
        auto& op = *static_cast<operation*>(t);
        auto stoken = stdexec::get_stop_token(stdexec::get_env(op.receiver_));
        if constexpr (std::unstoppable_token<decltype(stoken)>) {
          stdexec::set_value((Receiver&&) op.receiver_);
        } else if (stoken.stop_requested()) {
          stdexec::set_stopped((Receiver&&) op.receiver_);
        } else {
          stdexec::set_value((Receiver&&) op.receiver_);
        }
      };
    }

    void enqueue_(task_base* op) const {
      pool_.enqueue(op);
    }

    friend void tag_invoke(stdexec::start_t, operation& op) noexcept {
      op.enqueue_(&op);
    }
  };

  inline static_thread_pool::static_thread_pool()
    : static_thread_pool(std::thread::hardware_concurrency()) {
  }

  inline static_thread_pool::static_thread_pool(std::uint32_t threadCount)
    : threadCount_(threadCount)
    , threadStates_(threadCount)
    , nextThread_(0) {
    STDEXEC_ASSERT(threadCount > 0);

    threads_.reserve(threadCount);

    try {
      for (std::uint32_t i = 0; i < threadCount; ++i) {
        threads_.emplace_back([this, i] { run(i); });
      }
    } catch (...) {
      request_stop();
      join();
      throw;
    }
  }

  inline static_thread_pool::~static_thread_pool() {
    request_stop();
    join();
  }

  inline void static_thread_pool::request_stop() noexcept {
    for (auto& state: threadStates_) {
      state.request_stop();
    }
  }

  inline void static_thread_pool::run(const std::uint32_t threadIndex) noexcept {
    STDEXEC_ASSERT(threadIndex < threadCount_);
    while (true) {
      task_base* task = nullptr;
      std::uint32_t queueIndex = threadIndex;

      // Starting with this thread's queue, try to de-queue a task
      // from each thread's queue. try_pop() is non-blocking.
      do {
        task = threadStates_[queueIndex].try_pop();
      } while (!task && (++queueIndex %= threadCount_) != threadIndex);

      STDEXEC_ASSERT(task || queueIndex == threadIndex);
      // Make a blocking call to de-queue a task if we don't already have one.
      if (!task && !(task = threadStates_[queueIndex].pop()))
        return; // pop() only returns null when request_stop() was called.

      task->__execute(task, queueIndex);
    }
  }

  inline void static_thread_pool::join() noexcept {
    for (auto& t: threads_) {
      t.join();
    }
    threads_.clear();
  }

  inline void static_thread_pool::enqueue(task_base* task) noexcept {
    const std::uint32_t threadCount = static_cast<std::uint32_t>(threads_.size());
    const std::uint32_t startIndex =
      nextThread_.fetch_add(1, std::memory_order_relaxed) % threadCount;

    // First try to enqueue to one of the threads without blocking.
    for (std::uint32_t i = 0; i < threadCount; ++i) {
      const auto index =
        (startIndex + i) < threadCount ? (startIndex + i) : (startIndex + i - threadCount);
      if (threadStates_[index].try_push(task)) {
        return;
      }
    }

    // Otherwise, do a blocking enqueue on the selected thread.
    threadStates_[startIndex].push(task);
  }

  template <std::derived_from<task_base> TaskT>
  inline void static_thread_pool::bulk_enqueue(TaskT* task, std::uint32_t n_threads) noexcept {
    for (std::size_t i = 0; i < n_threads; ++i) {
      threadStates_[i % available_parallelism()].push(task + i);
    }
  }

  inline task_base* static_thread_pool::thread_state::try_pop() {
    std::unique_lock lk{mut_, std::try_to_lock};
    if (!lk || queue_.empty()) {
      return nullptr;
    }
    return queue_.pop_front();
  }

  inline task_base* static_thread_pool::thread_state::pop() {
    std::unique_lock lk{mut_};
    while (queue_.empty()) {
      if (stopRequested_) {
        return nullptr;
      }
      cv_.wait(lk);
    }
    return queue_.pop_front();
  }

  inline bool static_thread_pool::thread_state::try_push(task_base* task) {
    std::unique_lock lk{mut_, std::try_to_lock};
    if (!lk) {
      return false;
    }
    const bool wasEmpty = queue_.empty();
    queue_.push_back(task);
    if (wasEmpty) {
      cv_.notify_one();
    }
    return true;
  }

  inline void static_thread_pool::thread_state::push(task_base* task) {
    std::lock_guard lk{mut_};
    const bool wasEmpty = queue_.empty();
    queue_.push_back(task);
    if (wasEmpty) {
      cv_.notify_one();
    }
  }

  inline void static_thread_pool::thread_state::request_stop() {
    std::lock_guard lk{mut_};
    stopRequested_ = true;
    cv_.notify_one();
  }
} // namespace exec
