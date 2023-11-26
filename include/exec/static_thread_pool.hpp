/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates.
 * Copyright (c) 2021-2022 NVIDIA Corporation
 * Copyright (c) 2023 Maikel Nadolski
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
#include "./__detail/__atomic_intrusive_queue.hpp"
#include "./__detail/__bwos_lifo_queue.hpp"
#include "./__detail/__manual_lifetime.hpp"
#include "./__detail/__xorshift.hpp"

#include "./sequence_senders.hpp"
#include "./sequence/iterate.hpp"

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace exec {
  using stdexec::__intrusive_queue;

  // Splits `n` into `size` chunks distributing `n % size` evenly between ranks.
  // Returns `[begin, end)` range in `n` for a given `rank`.
  // Example:
  // ```cpp
  // //         n_items  thread  n_threads
  // even_share(     11,      0,         3); // -> [0,  4) -> 4 items
  // even_share(     11,      1,         3); // -> [4,  8) -> 4 items
  // even_share(     11,      2,         3); // -> [8, 11) -> 3 items
  // ```
  template <class Shape>
  std::pair<Shape, Shape> even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept {
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

#if STDEXEC_HAS_STD_RANGES()
  namespace schedule_all_ {
    template <class Range>
    struct sequence {
      class __t;
    };
  }
#endif

  template <class>
  struct not_a_sender {
    using sender_concept = stdexec::sender_t;
  };

  struct task_base {
    task_base* next;
    void (*__execute)(task_base*, std::uint32_t tid) noexcept;
  };

  struct bwos_params {
    std::size_t numBlocks{8};
    std::size_t blockSize{1024};
  };

  struct remote_queue {
    remote_queue* next_{};
    std::unique_ptr<__atomic_intrusive_queue<&task_base::next>[]> queues_{};
    std::thread::id id_{std::this_thread::get_id()};
  };

  struct remote_queue_list {
   private:
    std::atomic<remote_queue*> head_;
    remote_queue* tail_;
    std::size_t nthreads_;
    remote_queue this_remotes_;

   public:
    explicit remote_queue_list(std::size_t nthreads) noexcept
      : head_{&this_remotes_}
      , tail_{&this_remotes_}
      , nthreads_(nthreads) {
      this_remotes_.queues_ = std::make_unique<__atomic_intrusive_queue<&task_base::next>[]>(
        nthreads_);
    }

    ~remote_queue_list() noexcept {
      remote_queue* head = head_.load(std::memory_order_acquire);
      while (head != tail_) {
        remote_queue* tmp = std::exchange(head, head->next_);
        delete tmp;
      }
    }

    __intrusive_queue<&task_base::next> pop_all_reversed(std::size_t tid) noexcept {
      remote_queue* head = head_.load(std::memory_order_acquire);
      __intrusive_queue<&task_base::next> tasks{};
      while (head != nullptr) {
        tasks.append(head->queues_[tid].pop_all_reversed());
        head = head->next_;
      }
      return tasks;
    }

    remote_queue* get() {
      thread_local std::thread::id this_id = std::this_thread::get_id();
      remote_queue* head = head_.load(std::memory_order_acquire);
      remote_queue* queue = head;
      while (queue != tail_) {
        if (queue->id_ == this_id) {
          return queue;
        }
        queue = queue->next_;
      }
      remote_queue* new_head = new remote_queue{head};
      new_head->queues_ = std::make_unique<__atomic_intrusive_queue<&task_base::next>[]>(nthreads_);
      while (!head_.compare_exchange_weak(head, new_head, std::memory_order_acq_rel)) {
        new_head->next_ = head;
      }
      return new_head;
    }
  };

  class static_thread_pool {
    template <class ReceiverId>
    class operation;

    struct schedule_tag {
      // TODO: code to reconstitute a static_thread_pool schedule sender
    };

    template <class SenderId, std::integral Shape, class FunId>
    struct bulk_sender;

    template <stdexec::sender Sender, std::integral Shape, class Fun>
    using bulk_sender_t = //
      bulk_sender<
        stdexec::__x<stdexec::__decay_t<Sender>>,
        Shape,
        stdexec::__x<stdexec::__decay_t<Fun>>>;

#if STDEXEC_MSVC()
    // MSVCBUG https://developercommunity.visualstudio.com/t/Alias-template-with-pack-expansion-in-no/10437850

    template <class... Args>
    struct __bulk_non_throwing {
      using __t = stdexec::__decayed_tuple<Args...>;
      static constexpr bool __v = noexcept(__t(std::declval<Args>()...));
    };
#endif

    template <class Fun, class Shape, class... Args>
      requires stdexec::__callable<Fun, Shape, Args&...>
    using bulk_non_throwing = //
      stdexec::__mbool<
        // If function invocation doesn't throw
        stdexec::__nothrow_callable<Fun, Shape, Args&...> &&
    // and emplacing a tuple doesn't throw
#if STDEXEC_MSVC()
        __bulk_non_throwing<Args...>::__v
#else
        noexcept(stdexec::__decayed_tuple<Args...>(std::declval<Args>()...))
#endif
        // there's no need to advertise completion with `exception_ptr`
        >;

    template <class SenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
    struct bulk_shared_state;

    template <class SenderId, class ReceiverId, class Shape, class Fn, bool MayThrow>
    struct bulk_receiver;

    template <class SenderId, class ReceiverId, std::integral Shape, class Fun>
    struct bulk_op_state;

    struct transform_bulk {
      template <class Data, class Sender>
      auto operator()(stdexec::bulk_t, Data&& data, Sender&& sndr) {
        auto [shape, fun] = (Data&&) data;
        return bulk_sender_t<Sender, decltype(shape), decltype(fun)>{
          pool_, (Sender&&) sndr, shape, std::move(fun)};
      }

      static_thread_pool& pool_;
    };

#if STDEXEC_HAS_STD_RANGES()
    struct transform_iterate {
      template <class Range>
      stdexec::__t<schedule_all_::sequence<Range>> operator()(exec::iterate_t, Range&& range) {
        return {static_cast<Range&&>(range), pool_};
      }

      static_thread_pool& pool_;
    };
#endif

   public:
    struct domain {
      // For eager customization
      template <stdexec::sender_expr_for<stdexec::bulk_t> Sender>
      auto transform_sender(Sender&& sndr) const noexcept {
        if constexpr (stdexec::__completes_on<Sender, static_thread_pool::scheduler>) {
          auto sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(
            stdexec::get_env(sndr));
          return stdexec::__sexpr_apply((Sender&&) sndr, transform_bulk{*sched.pool_});
        } else {
          static_assert(
            stdexec::__completes_on<Sender, static_thread_pool::scheduler>,
            "No static_thread_pool instance can be found in the sender's environment "
            "on which to schedule bulk work.");
          return not_a_sender<stdexec::__name_of<Sender>>();
        }
        STDEXEC_UNREACHABLE();
      }

      // transform the generic bulk sender into a parallel thread-pool bulk sender
      template <stdexec::sender_expr_for<stdexec::bulk_t> Sender, class Env>
      auto transform_sender(Sender&& sndr, const Env& env) const noexcept {
        if constexpr (stdexec::__completes_on<Sender, static_thread_pool::scheduler>) {
          auto sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(
            stdexec::get_env(sndr));
          return stdexec::__sexpr_apply((Sender&&) sndr, transform_bulk{*sched.pool_});
        } else if constexpr (stdexec::__starts_on<Sender, static_thread_pool::scheduler, Env>) {
          auto sched = stdexec::get_scheduler(env);
          return stdexec::__sexpr_apply((Sender&&) sndr, transform_bulk{*sched.pool_});
        } else {
          static_assert( //
            stdexec::__starts_on<Sender, static_thread_pool::scheduler, Env>
              || stdexec::__completes_on<Sender, static_thread_pool::scheduler>,
            "No static_thread_pool instance can be found in the sender's or receiver's "
            "environment on which to schedule bulk work.");
          return not_a_sender<stdexec::__name_of<Sender>>();
        }
        STDEXEC_UNREACHABLE();
      }

#if STDEXEC_HAS_STD_RANGES()
      template <stdexec::sender_expr_for<exec::iterate_t> Sender>
      auto transform_sender(Sender&& sndr) const noexcept {
        auto sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(
          stdexec::get_env(sndr));
        return stdexec::__sexpr_apply((Sender&&) sndr, transform_iterate{*sched.pool_});
      }

      template <stdexec::sender_expr_for<exec::iterate_t> Sender, class Env>
        requires stdexec::__callable<stdexec::get_scheduler_t, Env>
      auto transform_sender(Sender&& sndr, const Env& env) const noexcept {
        auto sched = stdexec::get_scheduler(env);
        return stdexec::__sexpr_apply((Sender&&) sndr, transform_iterate{*sched.pool_});
      }
#endif
    };

   public:
    static_thread_pool();
    static_thread_pool(std::uint32_t threadCount, bwos_params params = {});
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
        using sender_concept = stdexec::sender_t;
        using completion_signatures =
          stdexec::completion_signatures< stdexec::set_value_t(), stdexec::set_stopped_t()>;
       private:
        template <typename Receiver>
        auto make_operation_(Receiver r) const -> operation<stdexec::__id<Receiver>> {
          return operation<stdexec::__id<Receiver>>{pool_, queue_, (Receiver&&) r};
        }

        template <stdexec::receiver Receiver>
        friend auto tag_invoke(stdexec::connect_t, sender s, Receiver r)
          -> operation<stdexec::__id<Receiver>> {
          return s.make_operation_((Receiver&&) r);
        }

        struct env {
          static_thread_pool& pool_;
          remote_queue* queue_;

          template <class CPO>
          friend static_thread_pool::scheduler
            tag_invoke(stdexec::get_completion_scheduler_t<CPO>, const env& self) noexcept {
            return self.make_scheduler_();
          }

          static_thread_pool::scheduler make_scheduler_() const {
            return static_thread_pool::scheduler{pool_, *queue_};
          }
        };

        friend env tag_invoke(stdexec::get_env_t, const sender& self) noexcept {
          return env{self.pool_, self.queue_};
        }

        friend struct static_thread_pool::scheduler;

        explicit sender(static_thread_pool& pool, remote_queue* queue) noexcept
          : pool_(pool)
          , queue_(queue) {
        }

        static_thread_pool& pool_;
        remote_queue* queue_;
      };

      sender make_sender_() const {
        return sender{*pool_, queue_};
      }

      friend sender tag_invoke(stdexec::schedule_t, const scheduler& s) noexcept {
        return s.make_sender_();
      }

      friend stdexec::forward_progress_guarantee
        tag_invoke(stdexec::get_forward_progress_guarantee_t, const static_thread_pool&) noexcept {
        return stdexec::forward_progress_guarantee::parallel;
      }

      friend domain tag_invoke(stdexec::get_domain_t, scheduler) noexcept {
        return {};
      }

      friend class static_thread_pool;

      explicit scheduler(static_thread_pool& pool) noexcept
        : pool_(&pool)
        , queue_{pool.get_remote_queue()} {
      }

      explicit scheduler(static_thread_pool& pool, remote_queue& queue) noexcept
        : pool_(&pool)
        , queue_{&queue} {
      }

      static_thread_pool* pool_;
      remote_queue* queue_;
    };

    scheduler get_scheduler() noexcept {
      return scheduler{*this};
    }

    remote_queue* get_remote_queue() noexcept {
      return remotes_.get();
    }

    void request_stop() noexcept;

    std::uint32_t available_parallelism() const {
      return threadCount_;
    }

    bwos_params params() const {
      return params_;
    }

    void enqueue(task_base* task) noexcept;
    void enqueue(remote_queue& queue, task_base* task) noexcept;

    template <std::derived_from<task_base> TaskT>
    void bulk_enqueue(TaskT* task, std::uint32_t n_threads) noexcept;
    void bulk_enqueue(remote_queue& queue, __intrusive_queue<&task_base::next> tasks) noexcept;

   private:
    class workstealing_victim {
     public:
      explicit workstealing_victim(
        bwos::lifo_queue<task_base*>* queue,
        std::uint32_t index) noexcept
        : queue_(queue)
        , index_(index) {
      }

      task_base* try_steal() noexcept {
        return queue_->steal_front();
      }

      std::uint32_t index() const noexcept {
        return index_;
      }

     private:
      bwos::lifo_queue<task_base*>* queue_;
      std::uint32_t index_;
    };

    class thread_state {
     public:
      struct pop_result {
        task_base* task;
        std::uint32_t queueIndex;
      };

      explicit thread_state(
        static_thread_pool* pool,
        std::uint32_t index,
        bwos_params params) noexcept
        : local_queue_(params.numBlocks, params.blockSize)
        , state_(state::running)
        , index_(index)
        , pool_(pool) {
        std::random_device rd;
        rng_.seed(rd);
      }

      pop_result pop();
      void push_local(task_base* task);
      void push_local(__intrusive_queue<&task_base::next>&& tasks);

      bool notify();
      void request_stop();

      void victims(std::vector<workstealing_victim>& victims) {
        victims_ = victims;
        // TODO sort by numa distance
        std::sort(victims_.begin(), victims_.end(), [i0 = index_](const auto& a, const auto& b) {
          auto distA = std::abs(static_cast<int>(a.index()) - static_cast<int>(i0));
          auto distB = std::abs(static_cast<int>(b.index()) - static_cast<int>(i0));
          return distA < distB;
        });
        // remove self from victims
        victims_.erase(victims_.begin());
      }

      std::uint32_t index() const noexcept {
        return index_;
      }

      workstealing_victim as_victim() noexcept {
        return workstealing_victim{&local_queue_, index_};
      }

     private:
      enum state {
        running,
        stealing,
        sleeping,
        notified
      };

      pop_result try_pop();
      pop_result try_remote();
      pop_result try_steal();

      void notify_one_sleeping();
      void set_stealing();
      void clear_stealing();

      bwos::lifo_queue<task_base*> local_queue_;
      __intrusive_queue<&task_base::next> pending_queue_{};
      std::mutex mut_{};
      std::condition_variable cv_{};
      bool stopRequested_{false};
      std::vector<workstealing_victim> victims_{};
      std::atomic<state> state_;
      std::uint32_t index_{};
      static_thread_pool* pool_;
      xorshift rng_{};
    };

    void run(std::uint32_t index) noexcept;
    void join() noexcept;

    alignas(64) std::atomic<std::uint32_t> nextThread_;
    alignas(64) std::atomic<std::uint32_t> numThiefs_{};
    alignas(64) remote_queue_list remotes_;
    std::uint32_t threadCount_;
    std::uint32_t maxSteals_{(threadCount_ + 1) << 1};
    bwos_params params_;
    std::vector<std::thread> threads_;
    std::vector<std::optional<thread_state>> threadStates_;
  };

  inline static_thread_pool::static_thread_pool()
    : static_thread_pool(std::thread::hardware_concurrency()) {
  }

  inline static_thread_pool::static_thread_pool(std::uint32_t threadCount, bwos_params params)
    : nextThread_(0)
    , remotes_(threadCount)
    , threadCount_(threadCount)
    , params_(params)
    , threadStates_(threadCount) {
    STDEXEC_ASSERT(threadCount > 0);

    for (std::uint32_t index = 0; index < threadCount; ++index) {
      threadStates_[index].emplace(this, index, params);
    }
    std::vector<workstealing_victim> victims{};
    for (auto& state: threadStates_) {
      victims.emplace_back(state->as_victim());
    }
    for (auto& state: threadStates_) {
      state->victims(victims);
    }
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
      state->request_stop();
    }
  }

  inline void static_thread_pool::run(const std::uint32_t threadIndex) noexcept {
    STDEXEC_ASSERT(threadIndex < threadCount_);
    while (true) {
      // Make a blocking call to de-queue a task if we don't already have one.
      auto [task, queueIndex] = threadStates_[threadIndex]->pop();
      if (!task) {
        return; // pop() only returns null when request_stop() was called.
      }
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
    this->enqueue(*get_remote_queue(), task);
  }

  inline void static_thread_pool::enqueue(remote_queue& queue, task_base* task) noexcept {
    static thread_local std::thread::id this_id = std::this_thread::get_id();
    std::size_t idx = 0;
    for (std::thread& t: threads_) {
      if (t.get_id() == this_id) {
        threadStates_[idx]->push_local(task);
        return;
      }
      ++idx;
    }
    if (this_id == queue.id_) {
      const std::uint32_t threadCount = static_cast<std::uint32_t>(threads_.size());
      const std::uint32_t startIndex =
        nextThread_.fetch_add(1, std::memory_order_relaxed) % threadCount;
      queue.queues_[startIndex].push_front(task);
      threadStates_[startIndex]->notify();
      return;
    } else {
      enqueue(task);
    }
  }

  template <std::derived_from<task_base> TaskT>
  void static_thread_pool::bulk_enqueue(TaskT* task, std::uint32_t n_threads) noexcept {
    auto& queue = *get_remote_queue();
    for (std::size_t i = 0; i < n_threads; ++i) {
      std::uint32_t index = i % available_parallelism();
      queue.queues_[index].push_front(task + i);
      threadStates_[index]->notify();
    }
  }

  inline void static_thread_pool::bulk_enqueue(
    remote_queue& queue,
    __intrusive_queue<&task_base::next> tasks) noexcept {
    static thread_local std::thread::id this_id = std::this_thread::get_id();
    std::size_t nTasks = 0;
    std::size_t idx = 0;
    for ([[maybe_unused]] auto t: tasks) {
      ++nTasks;
    }
    for (std::thread& t: threads_) {
      if (t.get_id() == this_id) {
        threadStates_[idx]->push_local(std::move(tasks));
        return;
      }
      ++idx;
    }
    remote_queue* correct_queue = this_id == queue.id_ ? &queue : get_remote_queue();
    std::uint32_t nThreads = available_parallelism();
    for (std::uint32_t i = 0; i < nThreads; ++i) {
      auto [i0, iEnd] = even_share(nTasks, i, available_parallelism());
      if (i0 == iEnd) {
        continue;
      }
      __intrusive_queue<&task_base::next> tmp{};
      for (std::size_t j = i0; j < iEnd; ++j) {
        tmp.push_back(tasks.pop_front());
      }
      correct_queue->queues_[i].prepend(std::move(tmp));
      threadStates_[i]->notify();
    }
  }

  inline void move_pending_to_local(
    __intrusive_queue<&task_base::next>& pending_queue,
    bwos::lifo_queue<task_base*>& local_queue) {
    auto last = local_queue.push_back(pending_queue.begin(), pending_queue.end());
    __intrusive_queue<&task_base::next> tmp{};
    tmp.splice(tmp.begin(), pending_queue, pending_queue.begin(), last);
    tmp.clear();
  }

  inline static_thread_pool::thread_state::pop_result
    static_thread_pool::thread_state::try_remote() {
    pop_result result{nullptr, index_};
    __intrusive_queue<& task_base::next> remotes = pool_->remotes_.pop_all_reversed(index_);
    pending_queue_.append(std::move(remotes));
    if (!pending_queue_.empty()) {
      move_pending_to_local(pending_queue_, local_queue_);
      result.task = local_queue_.pop_back();
    }
    return result;
  }

  inline static_thread_pool::thread_state::pop_result static_thread_pool::thread_state::try_pop() {
    pop_result result{nullptr, index_};
    result.task = local_queue_.pop_back();
    if (result.task) [[likely]] {
      return result;
    }
    return try_remote();
  }

  inline static_thread_pool::thread_state::pop_result
    static_thread_pool::thread_state::try_steal() {
    if (victims_.empty()) {
      return {nullptr, index_};
    }
    std::uniform_int_distribution<std::size_t> dist(0, victims_.size() - 1);
    std::size_t victimIndex = dist(rng_);
    auto& v = victims_[victimIndex];
    return {v.try_steal(), v.index()};
  }

  inline void static_thread_pool::thread_state::push_local(task_base* task) {
    if (!local_queue_.push_back(task)) {
      pending_queue_.push_back(task);
    }
  }

  inline void
    static_thread_pool::thread_state::push_local(__intrusive_queue<&task_base::next>&& tasks) {
    pending_queue_.prepend(std::move(tasks));
  }

  inline void static_thread_pool::thread_state::set_stealing() {
    pool_->numThiefs_.fetch_add(1, std::memory_order_relaxed);
  }

  inline void static_thread_pool::thread_state::clear_stealing() {
    if (pool_->numThiefs_.fetch_sub(1, std::memory_order_relaxed) == 1) {
      notify_one_sleeping();
    }
  }

  inline void static_thread_pool::thread_state::notify_one_sleeping() {
    std::uniform_int_distribution<std::uint32_t> dist(0, pool_->threadCount_ - 1);
    std::uint32_t startIndex = dist(rng_);
    for (std::uint32_t i = 0; i < pool_->threadCount_; ++i) {
      std::uint32_t index = (startIndex + i) % pool_->threadCount_;
      if (index == index_) {
        continue;
      }
      if (pool_->threadStates_[index]->notify()) {
        return;
      }
    }
  }

  inline static_thread_pool::thread_state::pop_result static_thread_pool::thread_state::pop() {
    pop_result result = try_pop();
    while (!result.task) {
      set_stealing();
      for (std::size_t i = 0; i < pool_->maxSteals_; ++i) {
        result = try_steal();
        if (result.task) {
          clear_stealing();
          return result;
        }
      }
      std::this_thread::yield();
      clear_stealing();

      std::unique_lock lock{mut_};
      if (stopRequested_) {
        return result;
      }
      state expected = state::running;
      if (state_.compare_exchange_weak(expected, state::sleeping, std::memory_order_relaxed)) {
        result = try_remote();
        if (result.task) {
          return result;
        }
        cv_.wait(lock);
      }
      lock.unlock();
      state_.store(state::running, std::memory_order_relaxed);
      result = try_pop();
    }
    return result;
  }

  inline bool static_thread_pool::thread_state::notify() {
    if (state_.exchange(state::notified, std::memory_order_relaxed) == state::sleeping) {
      {
        std::lock_guard lock{mut_};
      }
      cv_.notify_one();
      return true;
    }
    return false;
  }

  inline void static_thread_pool::thread_state::request_stop() {
    {
      std::lock_guard lock{mut_};
      stopRequested_ = true;
    }
    cv_.notify_one();
  }

  template <typename ReceiverId>
  class static_thread_pool::operation : public task_base {
    using Receiver = stdexec::__t<ReceiverId>;
    friend static_thread_pool::scheduler::sender;

    static_thread_pool& pool_;
    remote_queue* queue_;
    Receiver receiver_;

    explicit operation(static_thread_pool& pool, remote_queue* queue, Receiver&& r)
      : pool_(pool)
      , queue_(queue)
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
      pool_.enqueue(*queue_, op);
    }

    friend void tag_invoke(stdexec::start_t, operation& op) noexcept {
      op.enqueue_(&op);
    }
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // What follows is the implementation for parallel bulk execution on static_thread_pool.
  template <class SenderId, std::integral Shape, class FunId>
  struct static_thread_pool::bulk_sender {
    using Sender = stdexec::__t<SenderId>;
    using Fun = stdexec::__t<FunId>;
    using sender_concept = stdexec::sender_t;

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
      -> completion_signatures<Self, Env> {
      return {};
    }

    friend auto tag_invoke(stdexec::get_env_t, const bulk_sender& self) noexcept
      -> stdexec::env_of_t<const Sender&> {
      return stdexec::get_env(self.sndr_);
    }
  };

  template <class SenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
  struct static_thread_pool::bulk_shared_state {
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
                stdexec::set_error((Receiver&&) sh_state.receiver_, std::move(sh_state.exception_));
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

    std::uint32_t num_agents_required() const {
      return static_cast<std::uint32_t>(
        std::min(shape_, static_cast<Shape>(pool_.available_parallelism())));
    }

    template <class F>
    void apply(F f) {
      std::visit(
        [&](auto& tupl) -> void { std::apply([&](auto&... args) -> void { f(args...); }, tupl); },
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
  struct static_thread_pool::bulk_receiver {
    using receiver_concept = stdexec::receiver_t;
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
  struct static_thread_pool::bulk_op_state {
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

    bulk_op_state(static_thread_pool& pool, Shape shape, Fun fn, Sender&& sender, Receiver receiver)
      : shared_state_(pool, (Receiver&&) receiver, shape, fn)
      , inner_op_{stdexec::connect((Sender&&) sender, bulk_rcvr{shared_state_})} {
    }
  };

#if STDEXEC_HAS_STD_RANGES()
  namespace schedule_all_ {
    template <class Rcvr>
    auto get_allocator(const Rcvr& rcvr) {
      if constexpr (stdexec::__callable<stdexec::get_allocator_t, stdexec::env_of_t<Rcvr>>) {
        return stdexec::get_allocator(stdexec::get_env(rcvr));
      } else {
        return std::allocator<char>{};
      }
    }

    template <class Receiver>
    using allocator_of_t = decltype(get_allocator(stdexec::__declval<Receiver>()));

    template <class Range>
    struct operation_base {
      Range range_;
      static_thread_pool& pool_;
      std::mutex start_mutex_{};
      bool has_started_{false};
      __intrusive_queue<&task_base::next> tasks_{};
      std::atomic<std::size_t> countdown_{std::ranges::size(range_)};
    };

    template <class Range, class ItemReceiverId>
    struct item_operation {
      class __t : private task_base {
        using ItemReceiver = stdexec::__t<ItemReceiverId>;

        static void execute_(task_base* base, std::uint32_t /* tid */) noexcept {
          auto op = static_cast<__t*>(base);
          stdexec::set_value(static_cast<ItemReceiver&&>(op->item_receiver_), *op->it_);
        }

        ItemReceiver item_receiver_;
        std::ranges::iterator_t<Range> it_;
        operation_base<Range>* parent_;

        friend void tag_invoke(stdexec::start_t, __t& op) noexcept {
          std::unique_lock lock{op.parent_->start_mutex_};
          if (!op.parent_->has_started_) {
            op.parent_->tasks_.push_back(static_cast<task_base*>(&op));
          } else {
            lock.unlock();
            op.parent_->pool_.enqueue(static_cast<task_base*>(&op));
          }
        }

       public:
        using __id = item_operation;

        __t(
          ItemReceiver&& item_receiver,
          std::ranges::iterator_t<Range> it,
          operation_base<Range>* parent)
          : task_base{.__execute = execute_}
          , item_receiver_(static_cast<ItemReceiver&&>(item_receiver))
          , it_(it)
          , parent_(parent) {
        }
      };
    };

    template <class Range>
    struct item_sender {
      struct __t {
        using __id = item_sender;
        using sender_concept = stdexec::sender_t;
        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t(
          std::ranges::range_reference_t<Range>)>;

        operation_base<Range>* op_;
        std::ranges::iterator_t<Range> it_;

        struct env {
          static_thread_pool* pool_;

          template <
            stdexec::same_as<stdexec::get_completion_scheduler_t<stdexec::set_value_t>> Query>
          friend auto tag_invoke(Query, const env& e) noexcept -> static_thread_pool::scheduler {
            return e.pool_->get_scheduler();
          }
        };

        template <stdexec::same_as<stdexec::get_env_t> GetEnv, stdexec::__decays_to<__t> Self>
        friend auto tag_invoke(GetEnv, Self&& self) noexcept -> env {
          return {self.op_->pool_};
        }

        template <stdexec::__decays_to<__t> Self, stdexec::receiver ItemReceiver>
          requires stdexec::receiver_of<ItemReceiver, completion_signatures>
        friend auto tag_invoke(stdexec::connect_t, Self&& self, ItemReceiver rcvr) noexcept
          -> stdexec::__t<item_operation<Range, stdexec::__id<ItemReceiver>>> {
          return {static_cast<ItemReceiver&&>(rcvr), self.it_, self.op_};
        }
      };
    };

    template <class Range, class Receiver>
    struct operation_base_with_receiver : operation_base<Range> {
      Receiver receiver_;

      operation_base_with_receiver(Range range, static_thread_pool& pool, Receiver&& receiver)
        : operation_base<Range>{range, pool}
        , receiver_(static_cast<Receiver&&>(receiver)) {
      }
    };

    template <class Range, class Receiver>
    struct next_receiver {
      struct __t {
        using receiver_concept = stdexec::receiver_t;
        operation_base_with_receiver<Range, Receiver>* op_;

        template <stdexec::same_as<stdexec::set_value_t> SetValue, stdexec::same_as<__t> Self>
        friend void tag_invoke(SetValue, Self&& self) noexcept {
          std::size_t countdown = self.op_->countdown_.fetch_sub(1, std::memory_order_relaxed);
          if (countdown == 1) {
            stdexec::set_value((Receiver&&) self.op_->receiver_);
          }
        }

        template <stdexec::same_as<stdexec::set_stopped_t> SetStopped, stdexec::same_as<__t> Self>
        friend void tag_invoke(SetStopped, Self&& self) noexcept {
          std::size_t countdown = self.op_->countdown_.fetch_sub(1, std::memory_order_relaxed);
          if (countdown == 1) {
            stdexec::set_value((Receiver&&) self.op_->receiver_);
          }
        }

        template <stdexec::same_as<stdexec::get_env_t> GetEnv, stdexec::__decays_to<__t> Self>
        friend auto tag_invoke(GetEnv, Self&& self) noexcept -> stdexec::env_of_t<Receiver> {
          return stdexec::get_env(self.op_->receiver_);
        }
      };
    };

    template <class Range, class Receiver>
    struct operation {
      class __t : operation_base_with_receiver<Range, Receiver> {
        using Allocator = allocator_of_t<const Receiver&>;
        using ItemSender = stdexec::__t<item_sender<Range>>;
        using NextSender = next_sender_of_t<Receiver, ItemSender>;
        using NextReceiver = stdexec::__t<next_receiver<Range, Receiver>>;
        using ItemOperation = stdexec::connect_result_t<NextSender, NextReceiver>;

        using ItemAllocator =
          std::allocator_traits<Allocator>::template rebind_alloc<__manual_lifetime<ItemOperation>>;

        std::vector<__manual_lifetime<ItemOperation>, ItemAllocator> items_;

        template <stdexec::same_as<__t> Self>
        friend void tag_invoke(stdexec::start_t, Self& op) noexcept {
          std::size_t size = op.items_.size();
          std::size_t nthreads = op.pool_.available_parallelism();
          bwos_params params = op.pool_.params();
          std::size_t localSize = params.blockSize * params.numBlocks;
          std::size_t chunkSize = std::min<std::size_t>(size / nthreads, localSize * nthreads);

          auto& remote_queue = *op.pool_.get_remote_queue();
          std::ranges::iterator_t<Range> it = std::ranges::begin(op.range_);
          std::size_t i0 = 0;
          while (i0 + chunkSize < size) {
            for (std::size_t i = i0; i < i0 + chunkSize; ++i) {
              op.items_[i].__construct_with([&] {
                return stdexec::connect(
                  set_next(op.receiver_, ItemSender{&op, it + i}), NextReceiver{&op});
              });
              stdexec::start(op.items_[i].__get());
            }
            std::unique_lock lock{op.start_mutex_};
            op.pool_.bulk_enqueue(remote_queue, std::move(op.tasks_));
            lock.unlock();
            i0 += chunkSize;
          }
          for (std::size_t i = i0; i < size; ++i) {
            op.items_[i].__construct_with([&] {
              return stdexec::connect(
                set_next(op.receiver_, ItemSender{&op, it + i}), NextReceiver{&op});
            });
            stdexec::start(op.items_[i].__get());
          }
          std::unique_lock lock{op.start_mutex_};
          op.has_started_ = true;
          lock.unlock();
          op.pool_.bulk_enqueue(remote_queue, std::move(op.tasks_));
        }

       public:
        using __id = operation;

        __t(Range range, static_thread_pool& pool, Receiver&& receiver)
          : operation_base_with_receiver<
            Range,
            Receiver>{std::move(range), pool, static_cast<Receiver&&>(receiver)}
          , items_(std::ranges::size(this->range_), ItemAllocator(get_allocator(this->receiver_))) {
        }

        ~__t() {
          if (this->has_started_) {
            for (auto& item: items_) {
              item.__destroy();
            }
          }
        }
      };
    };

    template <class Range>
    class sequence<Range>::__t {
      using item_sender_t = stdexec::__t<item_sender<Range>>;

      Range range_;
      static_thread_pool* pool_;

     public:
      using __id = sequence;

      using sender_concept = sequence_sender_t;

      using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(),
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_stopped_t()>;

      using item_types = exec::item_types<stdexec::__t<item_sender<Range>>>;

      __t(Range range, static_thread_pool& pool)
        : range_(static_cast<Range&&>(range))
        , pool_(&pool) {
      }

     private:
      template <stdexec::__decays_to<__t> Self, exec::sequence_receiver_of<item_types> Receiver>
      friend auto tag_invoke(exec::subscribe_t, Self&& self, Receiver rcvr) noexcept
        -> stdexec::__t<operation<Range, Receiver>> {
        return {static_cast<Range&&>(self.range_), *self.pool_, static_cast<Receiver&&>(rcvr)};
      }
    };
  }

  struct schedule_all_t {
    template <class Range>
    stdexec::__t<schedule_all_::sequence<stdexec::__decay_t<Range>>>
      operator()(static_thread_pool& pool, Range&& range) const {
      return {static_cast<Range&&>(range), pool};
    }
  };

  inline constexpr schedule_all_t schedule_all{};
#endif

} // namespace exec
