/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates.
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "../stdexec/__detail/__atomic.hpp"
#include "../stdexec/__detail/__config.hpp"
#include "../stdexec/__detail/__intrusive_queue.hpp"
#include "../stdexec/__detail/__manual_lifetime.hpp" // IWYU pragma: keep
#include "../stdexec/__detail/__meta.hpp"            // IWYU pragma: keep
#include "../stdexec/execution.hpp"
#include "__detail/__atomic_intrusive_queue.hpp"
#include "__detail/__bwos_lifo_queue.hpp"
#include "__detail/__numa.hpp"
#include "__detail/__xorshift.hpp"

#include "sequence/iterate.hpp"
#include "sequence_senders.hpp"

#include <algorithm>
#include <compare>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <vector>

namespace exec {
  struct bwos_params {
    std::size_t numBlocks{32};
    std::size_t blockSize{8};
  };

  struct CANNOT_DISPATCH_THE_BULK_ALGORITHM_TO_THE_STATIC_THREAD_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_STATIC_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STATIC_THREAD_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  struct CANNOT_DISPATCH_THE_ITERATE_ALGORITHM_TO_THE_STATIC_THREAD_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_STATIC_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STATIC_THREAD_POOL_SCHEDULER_BEFORE_THE_ITERATE_ALGORITHM;

  namespace _pool_ {
    using namespace STDEXEC;

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
    constexpr auto even_share(Shape n, std::size_t rank, std::size_t size) noexcept //
      -> std::pair<Shape, Shape> {
      STDEXEC_ASSERT(n >= 0);
      using ushape_t = std::make_unsigned_t<Shape>;
      const auto avg_per_thread = static_cast<ushape_t>(n) / size;
      const auto n_big_share = avg_per_thread + 1;
      const auto big_shares = static_cast<ushape_t>(n) % size;
      const auto is_big_share = rank < big_shares;
      const auto begin = is_big_share
                         ? n_big_share * rank
                         : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
      const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

      return std::make_pair(static_cast<Shape>(begin), static_cast<Shape>(end));
    }

#if STDEXEC_HAS_STD_RANGES()
    namespace schedule_all_ {
      template <class Range>
      class sequence;
    } // namespace schedule_all_
#endif

    struct task_base {
      task_base* next = nullptr;
      void (*execute_)(task_base*, std::uint32_t tid) noexcept = nullptr;
    };

    struct remote_queue {
      explicit remote_queue(std::size_t nthreads) noexcept
        : queues_(nthreads) {
      }

      explicit remote_queue(remote_queue* next, std::size_t nthreads) noexcept
        : next_(next)
        , queues_(nthreads) {
      }

      remote_queue* next_{};
      std::vector<__atomic_intrusive_queue<&task_base::next>> queues_{};
      std::thread::id id_{std::this_thread::get_id()};
      // This marks whether the submitter is a thread in the pool or not.
      std::size_t index_{std::numeric_limits<std::size_t>::max()};
    };

    struct remote_queue_list {
     private:
      __std::atomic<remote_queue*> head_;
      remote_queue* tail_;
      std::size_t nthreads_;
      remote_queue this_remotes_;

     public:
      explicit remote_queue_list(std::size_t nthreads) noexcept
        : head_{&this_remotes_}
        , tail_{&this_remotes_}
        , nthreads_(nthreads)
        , this_remotes_(nthreads) {
      }

      ~remote_queue_list() noexcept {
        remote_queue* head = head_.load(__std::memory_order_acquire);
        while (head != tail_) {
          remote_queue* tmp = std::exchange(head, head->next_);
          delete tmp;
        }
      }

      auto pop_all_reversed(std::size_t tid) noexcept -> __intrusive_queue<&task_base::next> {
        remote_queue* head = head_.load(__std::memory_order_acquire);
        __intrusive_queue<&task_base::next> tasks{};
        while (head != nullptr) {
          tasks.append(head->queues_[tid].pop_all_reversed());
          head = head->next_;
        }
        return tasks;
      }

      auto get() -> remote_queue* {
        thread_local std::thread::id this_id = std::this_thread::get_id();
        remote_queue* head = head_.load(__std::memory_order_acquire);
        remote_queue* queue = head;
        while (queue != tail_) {
          if (queue->id_ == this_id) {
            return queue;
          }
          queue = queue->next_;
        }
        auto* new_head = new remote_queue{head, nthreads_};
        while (!head_.compare_exchange_weak(head, new_head, __std::memory_order_acq_rel)) {
          new_head->next_ = head;
        }
        return new_head;
      }
    };

    class _static_thread_pool {
      template <class Receiver>
      struct _opstate;

      template <bool Parallelize, std::integral Shape, class Fun, class Sender>
      struct _bulk_sender;

      template <class Shape, class Fun>
      struct _is_nothrow_bulk_fn {
        template <class... Args>
          requires __callable<Fun, Shape, Shape, __decay_t<Args>&...>
        using __f = __mbool<
          // If function invocation doesn't throw ...
          __nothrow_callable<Fun, Shape, Shape, __decay_t<Args>&...> &&
          // ... and decay-copying the arguments doesn't throw ...
          __nothrow_decay_copyable<Args...>
          // ... then there is no need to advertise completion with `exception_ptr`
        >;
      };

      template <
        bool Parallelize,
        class Shape,
        class Fun,
        bool MayThrow,
        class CvSender,
        class Receiver
      >
      struct _bulk_shared_state;

      template <
        bool Parallelize,
        class Shape,
        class Fun,
        bool MayThrow,
        class CvSender,
        class Receiver
      >
      struct _bulk_receiver;

      template <bool Parallelize, std::integral Shape, class Fun, class CvSender, class Receiver>
      struct _bulk_opstate;

      struct _transform_bulk {
        template <
          STDEXEC::__one_of<bulk_chunked_t, bulk_unchunked_t> Tag,
          class Data,
          class CvSender
        >
        auto operator()(Tag, Data&& data, CvSender&& sndr) const {
          auto [pol, shape, fun] = static_cast<Data&&>(data);
          using policy_t = std::remove_cvref_t<decltype(pol.__get())>;
          constexpr bool parallelize = std::same_as<policy_t, parallel_policy>
                                    || std::same_as<policy_t, parallel_unsequenced_policy>;

          if constexpr (__same_as<Tag, bulk_unchunked_t>) {
            // Turn a bulk_unchunked into a bulk_chunked operation
            using fun_t = STDEXEC::__bulk::__as_bulk_chunked_fn<decltype(fun)>;
            using sender_t = _bulk_sender<parallelize, decltype(shape), fun_t, __decay_t<CvSender>>;
            return sender_t{pool_, static_cast<CvSender&&>(sndr), shape, fun_t(std::move(fun))};
          } else {
            using fun_t = decltype(fun);
            using sender_t = _bulk_sender<parallelize, decltype(shape), fun_t, __decay_t<CvSender>>;
            return sender_t{pool_, static_cast<CvSender&&>(sndr), shape, std::move(fun)};
          }
        }

        _static_thread_pool& pool_;
      };

#if STDEXEC_HAS_STD_RANGES()
      struct _transform_iterate {
        template <class Range>
        auto
          operator()(exec::iterate_t, Range&& range) -> schedule_all_::sequence<__decay_t<Range>> {
          return {static_cast<Range&&>(range), pool_};
        }

        _static_thread_pool& pool_;
      };
#endif

      static auto _hardware_concurrency() noexcept -> unsigned int {
        unsigned int const n = std::thread::hardware_concurrency();
        return n == 0 ? 1 : n;
      }

     public:
      struct domain : STDEXEC::default_domain {
        // transform the generic bulk_chunked sender into a parallel thread-pool bulk sender
        template <sender_expr Sender, class Env>
          requires __one_of<tag_of_t<Sender>, bulk_chunked_t, bulk_unchunked_t>
        constexpr auto
          transform_sender(STDEXEC::set_value_t, Sender&& sndr, const Env& env) const noexcept {
          if constexpr (__completes_on<Sender, _static_thread_pool::scheduler, Env>) {
            auto sched = STDEXEC::get_completion_scheduler<set_value_t>(get_env(sndr), env);
            static_assert(std::is_same_v<decltype(sched), _static_thread_pool::scheduler>);
            return __apply(_transform_bulk{*sched.pool_}, static_cast<Sender&&>(sndr));
          } else {
            return STDEXEC::__not_a_sender<
              STDEXEC::_WHAT_(
                CANNOT_DISPATCH_THE_BULK_ALGORITHM_TO_THE_STATIC_THREAD_POOL_SCHEDULER),
              STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_STATIC_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
              STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, tag_of_t<Sender>),
              STDEXEC::_TO_FIX_THIS_ERROR_(
                ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STATIC_THREAD_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
              STDEXEC::_WITH_PRETTY_SENDER_<Sender>,
              STDEXEC::_WITH_ENVIRONMENT_(Env)
            >();
          }
        }

#if STDEXEC_HAS_STD_RANGES()
        template <sender_expr_for<exec::iterate_t> Sender, class Env>
        constexpr auto
          transform_sender(STDEXEC::set_value_t, Sender&& sndr, const Env& env) const noexcept {
          if constexpr (__completes_on<Sender, _static_thread_pool::scheduler, Env>) {
            auto sched = STDEXEC::get_scheduler(env);
            return __apply(_transform_iterate{*sched.pool_}, static_cast<Sender&&>(sndr));
          } else {
            return STDEXEC::__not_a_sender<
              STDEXEC::_WHAT_(
                CANNOT_DISPATCH_THE_ITERATE_ALGORITHM_TO_THE_STATIC_THREAD_POOL_SCHEDULER),
              STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_STATIC_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
              STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, exec::iterate_t),
              STDEXEC::_TO_FIX_THIS_ERROR_(
                ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STATIC_THREAD_POOL_SCHEDULER_BEFORE_THE_ITERATE_ALGORITHM),
              STDEXEC::_WITH_PRETTY_SENDER_<Sender>,
              STDEXEC::_WITH_ENVIRONMENT_(Env)
            >();
          }
        }
#endif
      };

     public:
      _static_thread_pool();
      _static_thread_pool(
        std::uint32_t threadCount,
        bwos_params params = {},
        numa_policy numa = get_numa_policy());
      ~_static_thread_pool();

      struct scheduler {
       private:
        template <class Receiver>
        friend struct _opstate;

        class _sender {
          struct env {
            _static_thread_pool& pool_;
            remote_queue* queue_;

            template <class CPO>
            auto query(get_completion_scheduler_t<CPO>, __ignore = {}) const noexcept
              -> _static_thread_pool::scheduler {
              return _static_thread_pool::scheduler{pool_, *queue_};
            }

            template <class CPO>
            auto query(get_completion_domain_t<CPO>, __ignore = {}) const noexcept -> domain {
              return {};
            }
          };

         public:
          using sender_concept = sender_t;
          template <class Receiver>
          using _opstate_t = _opstate<Receiver>;

          using completion_signatures =
            STDEXEC::completion_signatures<set_value_t(), set_stopped_t()>;

          [[nodiscard]]
          auto get_env() const noexcept -> env {
            return env{.pool_ = pool_, .queue_ = queue_};
          }

          template <receiver Receiver>
          auto connect(Receiver rcvr) const -> _opstate_t<Receiver> {
            return _opstate_t<Receiver>{
              pool_, queue_, static_cast<Receiver&&>(rcvr), threadIndex_, constraints_};
          }

         private:
          friend struct _static_thread_pool::scheduler;

          explicit _sender(
            _static_thread_pool& pool,
            remote_queue* queue,
            std::size_t threadIndex,
            const nodemask& constraints) noexcept
            : pool_(pool)
            , queue_(queue)
            , threadIndex_(threadIndex)
            , constraints_(constraints) {
          }

          _static_thread_pool& pool_;
          remote_queue* queue_;
          std::size_t threadIndex_{std::numeric_limits<std::size_t>::max()};
          nodemask constraints_{};
        };

        friend class _static_thread_pool;

        explicit scheduler(
          _static_thread_pool& pool,
          const nodemask* mask = &nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{pool.get_remote_queue()}
          , nodemask_{mask} {
        }

        explicit scheduler(
          _static_thread_pool& pool,
          remote_queue& queue,
          const nodemask* mask = &nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{&queue}
          , nodemask_{mask} {
        }

        explicit scheduler(
          _static_thread_pool& pool,
          remote_queue& queue,
          std::size_t threadIndex) noexcept
          : pool_(&pool)
          , queue_{&queue}
          , thread_idx_{threadIndex} {
        }

        _static_thread_pool* pool_;
        remote_queue* queue_;
        const nodemask* nodemask_ = &nodemask::any();
        std::size_t thread_idx_{std::numeric_limits<std::size_t>::max()};

       public:
        auto operator==(const scheduler&) const -> bool = default;

        [[nodiscard]]
        auto schedule() const noexcept -> _sender {
          return _sender{*pool_, queue_, thread_idx_, *nodemask_};
        }

        [[nodiscard]]
        auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee {
          return forward_progress_guarantee::parallel;
        }

        [[nodiscard]]
        auto query(get_completion_domain_t<set_value_t>, __ignore = {}) const noexcept -> domain {
          return {};
        }

        [[nodiscard]]
        auto query(get_completion_scheduler_t<set_value_t>, __ignore = {}) const noexcept
          -> scheduler {
          return scheduler{*this};
        }
      };

      auto get_scheduler() noexcept -> scheduler {
        return scheduler{*this};
      }

      auto get_scheduler_on_thread(std::size_t threadIndex) noexcept -> scheduler {
        return scheduler{*this, *get_remote_queue(), threadIndex};
      }

      // The caller must ensure that the constraints object is valid for the lifetime of the scheduler.
      auto get_constrained_scheduler(const nodemask* constraints) noexcept -> scheduler {
        return scheduler{*this, *get_remote_queue(), constraints};
      }

      auto get_remote_queue() noexcept -> remote_queue* {
        remote_queue* queue = remotes_.get();
        std::size_t index = 0;
        for (std::thread& t: threads_) {
          if (t.get_id() == queue->id_) {
            queue->index_ = index;
            break;
          }

          ++index;
        }
        return queue;
      }

      void request_stop() noexcept;

      [[nodiscard]]
      auto available_parallelism() const noexcept -> std::uint32_t {
        return thread_count_;
      }

      [[nodiscard]]
      auto params() const noexcept -> bwos_params {
        return params_;
      }

      void enqueue(task_base* task, const nodemask& contraints = nodemask::any()) noexcept;
      void enqueue(
        remote_queue& queue,
        task_base* task,
        const nodemask& contraints = nodemask::any()) noexcept;
      void enqueue(remote_queue& queue, task_base* task, std::size_t thread_index) noexcept;

      //! Enqueue a contiguous span of tasks across task queues.
      //! Note: We use the concrete `Task` because we enqueue
      //! tasks `task + 0`, `task + 1`, etc. so std::span<task_base>
      //! wouldn't be correct.
      //! This is O(n_threads) on the calling thread.
      template <std::derived_from<task_base> Task>
      void bulk_enqueue(std::span<Task> tasks) noexcept;
      void bulk_enqueue(
        remote_queue& queue,
        __intrusive_queue<&task_base::next> tasks,
        std::size_t tasks_size,
        const nodemask& constraints = nodemask::any()) noexcept;

     private:
      class workstealing_victim {
       public:
        explicit workstealing_victim(
          bwos::lifo_queue<task_base*, numa_allocator<task_base*>>* queue,
          std::uint32_t index,
          int numa_node) noexcept
          : queue_(queue)
          , index_(index)
          , numa_node_(numa_node) {
        }

        auto try_steal() noexcept -> task_base* {
          return queue_->steal_front();
        }

        [[nodiscard]]
        auto index() const noexcept -> std::uint32_t {
          return index_;
        }

        [[nodiscard]]
        auto numa_node() const noexcept -> int {
          return numa_node_;
        }

       private:
        bwos::lifo_queue<task_base*, numa_allocator<task_base*>>* queue_;
        std::uint32_t index_;
        int numa_node_;
      };

      struct thread_state_base {
        explicit thread_state_base(std::uint32_t index, const numa_policy& numa) noexcept
          : index_(index)
          , numa_node_(numa.thread_index_to_node(index)) {
        }

        std::uint32_t index_;
        int numa_node_;
      };

      class thread_state : private thread_state_base {
       public:
        struct pop_result {
          task_base* task;
          std::uint32_t queue_index;
        };

        explicit thread_state(
          _static_thread_pool* pool,
          std::uint32_t index,
          bwos_params params,
          const numa_policy& numa) noexcept
          : thread_state_base(index, numa)
          , local_queue_(
              params.numBlocks,
              params.blockSize,
              numa_allocator<task_base*>(this->numa_node_))
          , state_(state::running)
          , pool_(pool) {
          std::random_device rd;
          rng_.seed(rd);
        }

        auto pop() -> pop_result;
        void push_local(task_base* task);
        void push_local(__intrusive_queue<&task_base::next>&& tasks);

        auto notify() -> bool;
        void request_stop();

        void victims(const std::vector<workstealing_victim>& victims) {
          for (workstealing_victim v: victims) {
            if (v.index() == index_) {
              // skip self
              continue;
            }
            if (v.numa_node() == numa_node_) {
              near_victims_.push_back(v);
            }
            all_victims_.push_back(v);
          }
        }

        [[nodiscard]]
        auto index() const noexcept -> std::uint32_t {
          return index_;
        }

        [[nodiscard]]
        auto numa_node() const noexcept -> int {
          return numa_node_;
        }

        auto as_victim() noexcept -> workstealing_victim {
          return workstealing_victim{&local_queue_, index_, numa_node_};
        }

       private:
        enum state {
          running,
          stealing,
          sleeping,
          notified
        };

        auto try_pop() -> pop_result;
        auto try_remote() -> pop_result;
        auto try_steal(std::span<workstealing_victim> victims) -> pop_result;
        auto try_steal_near() -> pop_result;
        auto try_steal_any() -> pop_result;

        void notify_one_sleeping();
        void set_stealing();
        void clear_stealing();
        void set_sleeping();
        void clear_sleeping();

        bwos::lifo_queue<task_base*, numa_allocator<task_base*>> local_queue_;
        __intrusive_queue<&task_base::next> pending_queue_{};
        std::mutex mut_{};
        std::condition_variable cv_{};
        bool stop_requested_{false};
        std::vector<workstealing_victim> near_victims_{};
        std::vector<workstealing_victim> all_victims_{};
        __std::atomic<state> state_;
        _static_thread_pool* pool_;
        xorshift rng_{};
      };

      void run(std::uint32_t index) noexcept;
      void join() noexcept;

      alignas(64) __std::atomic<std::uint32_t> num_active_{};
      alignas(64) remote_queue_list remotes_;
      std::uint32_t thread_count_;
      std::uint32_t max_steals_{thread_count_ + 1};
      bwos_params params_;
      std::vector<std::thread> threads_;
      std::vector<std::optional<thread_state>> thread_states_;
      numa_policy numa_;

      struct thread_index_by_numa_node {
        int numa_node;
        std::size_t thread_index;

        friend auto operator==(
          const thread_index_by_numa_node& lhs,
          const thread_index_by_numa_node& rhs) noexcept -> bool {
          return lhs.numa_node == rhs.numa_node;
        }

        friend auto operator<=>(
          const thread_index_by_numa_node& lhs,
          const thread_index_by_numa_node& rhs) noexcept -> std::strong_ordering {
          return lhs.numa_node <=> rhs.numa_node;
        }
      };

      std::vector<thread_index_by_numa_node> thread_index_by_numa_node_;

      [[nodiscard]]
      auto num_threads(int numa) const noexcept -> std::size_t;
      [[nodiscard]]
      auto num_threads(nodemask constraints) const noexcept -> std::size_t;
      [[nodiscard]]
      auto get_thread_index(int numa, std::size_t index) const noexcept -> std::size_t;
      auto random_thread_index_with_constraints(const nodemask& contraints) noexcept -> std::size_t;
    };

    inline _static_thread_pool::_static_thread_pool()
      : _static_thread_pool(_hardware_concurrency()) {
    }

    inline _static_thread_pool::_static_thread_pool(
      std::uint32_t thread_count,
      bwos_params params,
      numa_policy numa)
      : remotes_(thread_count)
      , thread_count_(thread_count)
      , params_(params)
      , thread_states_(thread_count)
      , numa_(std::move(numa)) {
      STDEXEC_ASSERT(thread_count > 0);

      for (std::uint32_t index = 0; index < thread_count; ++index) {
        thread_states_[index].emplace(this, index, params, numa_);
        thread_index_by_numa_node_.push_back(
          thread_index_by_numa_node{
            .numa_node = thread_states_[index]->numa_node(), .thread_index = index});
      }

      // NOLINTNEXTLINE(modernize-use-ranges) we still support platforms without the std::ranges algorithms
      std::sort(thread_index_by_numa_node_.begin(), thread_index_by_numa_node_.end());
      std::vector<workstealing_victim> victims{};
      for (auto& state: thread_states_) {
        victims.emplace_back(state->as_victim());
      }
      for (auto& state: thread_states_) {
        state->victims(victims);
      }
      threads_.reserve(thread_count);

      STDEXEC_TRY {
        num_active_.store(thread_count << 16u, __std::memory_order_relaxed);
        for (std::uint32_t i = 0; i < thread_count; ++i) {
          threads_.emplace_back([this, i] { run(i); });
        }
      }
      STDEXEC_CATCH_ALL {
        request_stop();
        join();
        STDEXEC_THROW();
      }
    }

    inline _static_thread_pool::~_static_thread_pool() {
      request_stop();
      join();
    }

    inline void _static_thread_pool::request_stop() noexcept {
      for (auto& state: thread_states_) {
        state->request_stop();
      }
    }

    inline void _static_thread_pool::run(std::uint32_t thread_index) noexcept {
      STDEXEC_ASSERT(thread_index < thread_count_);
      // NOLINTNEXTLINE(bugprone-unused-return-value)
      numa_.bind_to_node(thread_states_[thread_index]->numa_node());
      while (true) {
        // Make a blocking call to de-queue a task if we don't already have one.
        auto [task, queue_index] = thread_states_[thread_index]->pop();
        if (!task) {
          return; // pop() only returns null when request_stop() was called.
        }
        task->execute_(task, queue_index);
      }
    }

    inline void _static_thread_pool::join() noexcept {
      for (auto& t: threads_) {
        t.join();
      }
      threads_.clear();
    }

    inline void
      _static_thread_pool::enqueue(task_base* task, const nodemask& constraints) noexcept {
      this->enqueue(*get_remote_queue(), task, constraints);
    }

    inline auto _static_thread_pool::num_threads(int numa) const noexcept -> std::size_t {
      thread_index_by_numa_node key{.numa_node = numa, .thread_index = 0};
      // NOLINTNEXTLINE(modernize-use-ranges) we still support platforms without the std::ranges algorithms
      auto it =
        std::lower_bound(thread_index_by_numa_node_.begin(), thread_index_by_numa_node_.end(), key);
      if (it == thread_index_by_numa_node_.end()) {
        return 0;
      }
      auto it_end = std::upper_bound(it, thread_index_by_numa_node_.end(), key);
      return static_cast<std::size_t>(std::distance(it, it_end));
    }

    inline auto
      _static_thread_pool::num_threads(nodemask constraints) const noexcept -> std::size_t {
      const std::size_t n_nodes = static_cast<unsigned>(
        thread_index_by_numa_node_.back().numa_node + 1);
      std::size_t n_threads = 0;
      for (std::size_t node_index = 0; node_index < n_nodes; ++node_index) {
        if (!constraints[node_index]) {
          continue;
        }

        n_threads += num_threads(static_cast<int>(node_index));
      }
      return n_threads;
    }

    inline auto
      _static_thread_pool::get_thread_index(int node_index, std::size_t thread_index) const noexcept
      -> std::size_t {
      thread_index_by_numa_node key{.numa_node = node_index, .thread_index = 0};
      // NOLINTNEXTLINE(modernize-use-ranges) we still support platforms without the std::ranges algorithms
      auto it =
        std::lower_bound(thread_index_by_numa_node_.begin(), thread_index_by_numa_node_.end(), key);
      STDEXEC_ASSERT(it != thread_index_by_numa_node_.end());
      std::advance(it, thread_index);
      return it->thread_index;
    }

    inline auto _static_thread_pool::random_thread_index_with_constraints(
      const nodemask& constraints) noexcept -> std::size_t {
      thread_local std::uint64_t start_index{std::uint64_t(std::random_device{}())};
      start_index += 1;
      std::size_t target_index = start_index % thread_count_;
      std::size_t n_threads = num_threads(constraints);
      if (n_threads != 0) {
        for (std::size_t node_index = 0; node_index < numa_.num_nodes(); ++node_index) {
          if (!constraints[node_index]) {
            continue;
          }
          std::size_t n_threads = num_threads(static_cast<int>(node_index));
          if (target_index < n_threads) {
            return get_thread_index(static_cast<int>(node_index), target_index);
          }
          target_index -= n_threads;
        }
      }
      return target_index;
    }

    inline void _static_thread_pool::enqueue(
      remote_queue& queue,
      task_base* task,
      const nodemask& constraints) noexcept {
      static thread_local std::thread::id this_id = std::this_thread::get_id();
      remote_queue* correct_queue = this_id == queue.id_ ? &queue : get_remote_queue();
      std::size_t idx = correct_queue->index_;
      if (idx < thread_states_.size()) {
        auto this_node = static_cast<std::size_t>(thread_states_[idx]->numa_node());
        if (constraints[this_node]) {
          thread_states_[idx]->push_local(task);
          return;
        }
      }

      const std::size_t thread_index = random_thread_index_with_constraints(constraints);
      queue.queues_[thread_index].push_front(task);
      thread_states_[thread_index]->notify();
    }

    inline void _static_thread_pool::enqueue(
      remote_queue& queue,
      task_base* task,
      std::size_t thread_index) noexcept {
      thread_index %= thread_count_;
      queue.queues_[thread_index].push_front(task);
      thread_states_[thread_index]->notify();
    }

    template <std::derived_from<task_base> Task>
    void _static_thread_pool::bulk_enqueue(std::span<Task> tasks) noexcept {
      auto& queue = *this->get_remote_queue();
      for (std::uint32_t i = 0; i < tasks.size(); ++i) {
        std::uint32_t index = i % this->available_parallelism();
        queue.queues_[index].push_front(&tasks[i]);
        thread_states_[index]->notify();
      }
      // At this point the calling thread can exit and the pool will take over.
      // Ultimately, the last completing thread passes the result forward.
      // See `if (is_last_thread)` above.
    }

    inline void _static_thread_pool::bulk_enqueue(
      remote_queue& queue,
      __intrusive_queue<&task_base::next> tasks,
      std::size_t tasks_size,
      const nodemask& constraints) noexcept {
      static thread_local std::thread::id const this_id = std::this_thread::get_id();
      remote_queue* const correct_queue = this_id == queue.id_ ? &queue : get_remote_queue();
      std::size_t const idx = correct_queue->index_;
      if (idx < thread_states_.size()) {
        auto this_node = static_cast<std::size_t>(thread_states_[idx]->numa_node());
        if (constraints[this_node]) {
          thread_states_[idx]->push_local(std::move(tasks));
          return;
        }
      }

      std::uint32_t const total_threads = available_parallelism();
      for (std::uint32_t i = 0; i < total_threads; ++i) {
        auto [begin, end] = even_share(tasks_size, i, total_threads);
        if (begin == end) {
          continue;
        }
        __intrusive_queue<&task_base::next> tmp{};
        for (auto j = begin; j < end; ++j) {
          tmp.push_back(tasks.pop_front());
        }
        correct_queue->queues_[i].prepend(std::move(tmp));
        thread_states_[i]->notify();
      }
    }

    inline void move_pending_to_local(
      __intrusive_queue<&task_base::next>& pending_queue,
      bwos::lifo_queue<task_base*, numa_allocator<task_base*>>& local_queue) {
      auto const last = local_queue.push_back(pending_queue.begin(), pending_queue.end());
      __intrusive_queue<&task_base::next> tmp{};
      tmp.splice(tmp.begin(), pending_queue, pending_queue.begin(), last);
      tmp.clear();
    }

    inline auto _static_thread_pool::thread_state::try_remote()
      -> _static_thread_pool::thread_state::pop_result {
      pop_result result{.task = nullptr, .queue_index = index_};
      __intrusive_queue<&task_base::next> remotes = pool_->remotes_.pop_all_reversed(index_);
      pending_queue_.append(std::move(remotes));
      if (!pending_queue_.empty()) {
        move_pending_to_local(pending_queue_, local_queue_);
        result.task = local_queue_.pop_back();
      }

      return result;
    }

    inline auto _static_thread_pool::thread_state::try_pop()
      -> _static_thread_pool::thread_state::pop_result {
      pop_result result{.task = nullptr, .queue_index = index_};
      result.task = local_queue_.pop_back();
      if (result.task) [[likely]] {
        return result;
      }
      return try_remote();
    }

    inline auto _static_thread_pool::thread_state::try_steal(std::span<workstealing_victim> victims)
      -> _static_thread_pool::thread_state::pop_result {
      if (victims.empty()) {
        return {.task = nullptr, .queue_index = index_};
      }
      std::uniform_int_distribution<std::uint32_t> dist(
        0, static_cast<std::uint32_t>(victims.size() - 1));
      std::uint32_t victim_index = dist(rng_);
      auto& v = victims[victim_index];
      return {.task = v.try_steal(), .queue_index = v.index()};
    }

    inline auto _static_thread_pool::thread_state::try_steal_near()
      -> _static_thread_pool::thread_state::pop_result {
      return try_steal(near_victims_);
    }

    inline auto _static_thread_pool::thread_state::try_steal_any()
      -> _static_thread_pool::thread_state::pop_result {
      return try_steal(all_victims_);
    }

    inline void _static_thread_pool::thread_state::push_local(task_base* task) {
      if (!local_queue_.push_back(task)) {
        pending_queue_.push_back(task);
      }
    }

    inline void
      _static_thread_pool::thread_state::push_local(__intrusive_queue<&task_base::next>&& tasks) {
      pending_queue_.prepend(std::move(tasks));
    }

    inline void _static_thread_pool::thread_state::set_sleeping() {
      pool_->num_active_.fetch_sub(1u << 16u, __std::memory_order_relaxed);
    }

    // wakeup a worker thread and maintain the invariant that we always one active thief as long as a potential victim is awake
    inline void _static_thread_pool::thread_state::clear_sleeping() {
      const std::uint32_t num_active = pool_->num_active_
                                         .fetch_add(1u << 16u, __std::memory_order_relaxed);
      if (num_active == 0) {
        notify_one_sleeping();
      }
    }

    inline void _static_thread_pool::thread_state::set_stealing() {
      const std::uint32_t diff = 1u - (1u << 16u);
      pool_->num_active_.fetch_add(diff, __std::memory_order_relaxed);
    }

    // put a thief to sleep but maintain the invariant that we always have one active thief as long as a potential victim is awake
    inline void _static_thread_pool::thread_state::clear_stealing() {
      constexpr std::uint32_t diff = 1 - (1u << 16u);
      const std::uint32_t num_active = pool_->num_active_
                                         .fetch_sub(diff, __std::memory_order_relaxed);
      const std::uint32_t num_victims = num_active >> 16u;
      const std::uint32_t num_thiefs = num_active & 0xffffu;
      if (num_thiefs == 1 && num_victims != 0) {
        notify_one_sleeping();
      }
    }

    inline void _static_thread_pool::thread_state::notify_one_sleeping() {
      std::uniform_int_distribution<std::uint32_t> dist(0, pool_->thread_count_ - 1);
      std::uint32_t start_index = dist(rng_);
      for (std::uint32_t i = 0; i < pool_->thread_count_; ++i) {
        std::uint32_t index = (start_index + i) % pool_->thread_count_;
        if (index == index_) {
          continue;
        }
        if (pool_->thread_states_[index]->notify()) {
          return;
        }
      }
    }

    inline auto _static_thread_pool::thread_state::pop() //
      -> _static_thread_pool::thread_state::pop_result {
      pop_result result = try_pop();
      while (!result.task) {
        set_stealing();
        for (std::size_t i = 0; i < pool_->max_steals_; ++i) {
          result = try_steal_near();
          if (result.task) {
            clear_stealing();
            return result;
          }
        }

        for (std::size_t i = 0; i < pool_->max_steals_; ++i) {
          result = try_steal_any();
          if (result.task) {
            clear_stealing();
            return result;
          }
        }
        std::this_thread::yield();
        clear_stealing();

        std::unique_lock lock{mut_};
        if (stop_requested_) {
          return result;
        }
        state expected = state::running;
        if (state_.compare_exchange_weak(expected, state::sleeping, __std::memory_order_relaxed)) {
          result = try_remote();
          if (result.task) {
            return result;
          }
          set_sleeping();
          cv_.wait(lock);
          lock.unlock();
          clear_sleeping();
        }
        if (lock.owns_lock()) {
          lock.unlock();
        }
        state_.store(state::running, __std::memory_order_relaxed);
        result = try_pop();
      }
      return result;
    }

    inline auto _static_thread_pool::thread_state::notify() -> bool {
      if (state_.exchange(state::notified, __std::memory_order_relaxed) == state::sleeping) {
        {
          std::lock_guard lock{mut_};
        }
        cv_.notify_one();
        return true;
      }
      return false;
    }

    inline void _static_thread_pool::thread_state::request_stop() {
      {
        std::lock_guard lock{mut_};
        stop_requested_ = true;
      }
      cv_.notify_one();
    }

    template <class Receiver>
    struct _static_thread_pool::_opstate : task_base {
     private:
      friend _static_thread_pool::scheduler::_sender;

      explicit _opstate(
        _static_thread_pool& pool,
        remote_queue* queue,
        Receiver rcvr,
        std::size_t tid,
        const nodemask& constraints)
        : pool_(pool)
        , queue_(queue)
        , rcvr_(static_cast<Receiver&&>(rcvr))
        , thread_index_{tid}
        , constraints_{constraints} {
        this->execute_ = [](task_base* t, const std::uint32_t /* tid */) noexcept {
          auto& op = *static_cast<_opstate*>(t);
          auto stoken = get_stop_token(get_env(op.rcvr_));
          if constexpr (STDEXEC::unstoppable_token<decltype(stoken)>) {
            STDEXEC::set_value(static_cast<Receiver&&>(op.rcvr_));
          } else {
            if (stoken.stop_requested()) {
              STDEXEC::set_stopped(static_cast<Receiver&&>(op.rcvr_));
            } else {
              STDEXEC::set_value(static_cast<Receiver&&>(op.rcvr_));
            }
          }
        };
      }

      void enqueue_(task_base* op) const {
        if (thread_index_ < pool_.available_parallelism()) {
          pool_.enqueue(*queue_, op, thread_index_);
        } else {
          pool_.enqueue(*queue_, op, constraints_);
        }
      }

      _static_thread_pool& pool_;
      remote_queue* queue_;
      Receiver rcvr_;
      std::size_t thread_index_{};
      nodemask constraints_{};

     public:
      void start() & noexcept {
        enqueue_(this);
      }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // What follows is the implementation for parallel bulk execution on _static_thread_pool.
    template <bool Parallelize, std::integral Shape, class Fun, class Sender>
    struct _static_thread_pool::_bulk_sender {
      using sender_concept = sender_t;

      template <class CvSender, class... Env>
      using _with_error_invoke_t = __if_c<
        __value_types_t<
          __completion_signatures_of_t<CvSender, Env...>,
          _is_nothrow_bulk_fn<Shape, Fun>,
          __q<__mand>
        >::value,
        completion_signatures<>,
        __eptr_completion
      >;

      template <class... Tys>
      using _set_value_t = completion_signatures<set_value_t(STDEXEC::__decay_t<Tys>...)>;

      template <class Self, class... Env>
      using _completions_t = STDEXEC::transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
        _with_error_invoke_t<__copy_cvref_t<Self, Sender>, Env...>,
        _set_value_t
      >;

      template <class Receiver>
      using _bulk_opstate_t =
        _static_thread_pool::_bulk_opstate<Parallelize, Shape, Fun, Sender, Receiver>;

      template <__decays_to<_bulk_sender> Self, receiver Receiver>
        requires receiver_of<Receiver, _completions_t<Self, env_of_t<Receiver>>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        noexcept(__nothrow_constructible_from<
                 _bulk_opstate_t<Receiver>,
                 _static_thread_pool&,
                 Shape,
                 Fun,
                 Sender,
                 Receiver
        >) -> _bulk_opstate_t<Receiver> {
        return _bulk_opstate_t<Receiver>{
          self.pool_,
          self.shape_,
          self.fun_,
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<_bulk_sender> Self, class... Env>
      static consteval auto get_completion_signatures() -> _completions_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> env_of_t<const Sender&> {
        return STDEXEC::get_env(sndr_);
      }

      _static_thread_pool& pool_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;
    };

    //! The customized operation state for `STDEXEC::bulk` operations
    template <bool Parallelize, class Shape, class Fun, bool MayThrow, class CvSender, class Receiver>
    struct _static_thread_pool::_bulk_shared_state {
      //! The actual `bulk_task` holds a pointer to the shared state
      //! and its `execute_` function reads from that shared state.
      struct bulk_task : task_base {
        _bulk_shared_state* sh_state_;

        bulk_task(_bulk_shared_state* sh_state)
          : sh_state_(sh_state) {
          this->execute_ = [](task_base* t, const std::uint32_t tid) noexcept {
            auto& sh_state = *static_cast<bulk_task*>(t)->sh_state_;
            auto total_threads = sh_state.num_agents_required();

            auto computation = [&](auto&... args) {
              // Each computation does one or more call to the the bulk function.
              // In the case that the shape is much larger than the total number of threads,
              // then each call to computation will call the function many times.
              auto [begin, end] = even_share(sh_state.shape_, tid, total_threads);
              sh_state.fun_(begin, end, args...);
            };

            auto completion = [&](auto&... args) {
              STDEXEC::set_value(static_cast<Receiver&&>(sh_state.rcvr_), std::move(args)...);
            };

            if constexpr (MayThrow) {
              STDEXEC_TRY {
                sh_state.apply(computation);
              }
              STDEXEC_CATCH_ALL {
                std::uint32_t expected = total_threads;

                if (sh_state.thread_with_exception_.compare_exchange_strong(
                      expected, tid, __std::memory_order_relaxed, __std::memory_order_relaxed)) {
                  sh_state.exception_ = std::current_exception();
                }
              }

              const bool is_last_thread = sh_state.finished_threads_.fetch_add(1)
                                       == (total_threads - 1);

              if (is_last_thread) {
                if (sh_state.exception_) {
                  STDEXEC::set_error(
                    static_cast<Receiver&&>(sh_state.rcvr_), std::move(sh_state.exception_));
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

      using variant_t = __value_types_of_t<
        CvSender,
        env_of_t<Receiver>,
        __q<__decayed_std_tuple>,
        __q<__nullable_std_variant>
      >;

      variant_t data_;
      _static_thread_pool& pool_;
      Receiver rcvr_;
      Shape shape_;
      Fun fun_;

      __std::atomic<std::uint32_t> finished_threads_{0};
      __std::atomic<std::uint32_t> thread_with_exception_{0};
      std::exception_ptr exception_;
      std::vector<bulk_task> tasks_;

      //! The number of agents required is the minimum of `shape_` and the available parallelism.
      //! That is, we don't need an agent for each of the shape values.
      [[nodiscard]]
      auto num_agents_required() const noexcept -> std::uint32_t {
        if constexpr (Parallelize) {
          return static_cast<std::uint32_t>(
            (std::min) (shape_, static_cast<Shape>(pool_.available_parallelism())));
        } else {
          return static_cast<std::uint32_t>(1);
        }
      }

      template <class F>
      void apply(F f) {
        std::visit(
          [&]<class Tuple>(Tuple& tupl) -> void {
            if constexpr (__std::same_as<Tuple, std::monostate>) {
              STDEXEC_TERMINATE();
            } else {
              std::apply([&](auto&... args) -> void { f(args...); }, tupl);
            }
          },
          data_);
      }

      //! Construct from a pool, receiver, shape, and function.
      //! Allocates O(min(shape, available_parallelism())) memory.
      _bulk_shared_state(_static_thread_pool& pool, Receiver rcvr, Shape shape, Fun fun)
        : pool_{pool}
        , rcvr_{static_cast<Receiver&&>(rcvr)}
        , shape_{shape}
        , fun_{fun}
        , thread_with_exception_{num_agents_required()}
        , tasks_{num_agents_required(), {this}} {
      }
    };

    //! A customized receiver to allow parallel execution of `STDEXEC::bulk` operations:
    template <bool Parallelize, class Shape, class Fun, bool MayThrow, class CvSender, class Receiver>
    struct _static_thread_pool::_bulk_receiver {
      using receiver_concept = receiver_t;

      using shared_state =
        _bulk_shared_state<Parallelize, Shape, Fun, MayThrow, CvSender, Receiver>;

      void enqueue() noexcept {
        STDEXEC_ASSERT(shared_state_.tasks_.size() == shared_state_.num_agents_required());
        shared_state_.pool_.bulk_enqueue(std::span{shared_state_.tasks_});
      }

      template <class... As>
      void set_value(As&&... as) noexcept {
        using tuple_t = __decayed_std_tuple<As...>;

        shared_state& state = shared_state_;

        STDEXEC_TRY {
          state.data_.template emplace<tuple_t>(static_cast<As&&>(as)...);
        }
        STDEXEC_CATCH_ALL {
          if constexpr (MayThrow) {
            STDEXEC::set_error(std::move(state.rcvr_), std::current_exception());
          }
        }

        if (state.shape_) {
          enqueue();
        } else {
          state.apply(
            [&](auto&... args) { STDEXEC::set_value(std::move(state.rcvr_), std::move(args)...); });
        }
      }

      template <class Error>
      void set_error(Error&& error) noexcept {
        shared_state& state = shared_state_;
        STDEXEC::set_error(static_cast<Receiver&&>(state.rcvr_), static_cast<Error&&>(error));
      }

      void set_stopped() noexcept {
        shared_state& state = shared_state_;
        STDEXEC::set_stopped(static_cast<Receiver&&>(state.rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<Receiver> {
        return STDEXEC::get_env(shared_state_.rcvr_);
      }

      shared_state& shared_state_;
    };

    template <bool Parallelize, std::integral Shape, class Fun, class CvSender, class Receiver>
    struct _static_thread_pool::_bulk_opstate {
      static constexpr bool may_throw = !__value_types_of_t<
        CvSender,
        env_of_t<Receiver>,
        _is_nothrow_bulk_fn<Shape, Fun>,
        __q<__mand>
      >::value;

      using receiver_t = _bulk_receiver<Parallelize, Shape, Fun, may_throw, CvSender, Receiver>;
      using shared_state_t =
        _bulk_shared_state<Parallelize, Shape, Fun, may_throw, CvSender, Receiver>;
      using inner_opstate_t = connect_result_t<CvSender, receiver_t>;

      shared_state_t shared_state_;
      inner_opstate_t inner_op_;

      void start() & noexcept {
        STDEXEC::start(inner_op_);
      }

      _bulk_opstate(_static_thread_pool& pool, Shape shape, Fun fun, CvSender&& sndr, Receiver rcvr)
        : shared_state_(pool, static_cast<Receiver&&>(rcvr), shape, fun)
        , inner_op_{STDEXEC::connect(static_cast<CvSender&&>(sndr), receiver_t{shared_state_})} {
      }
    };

#if STDEXEC_HAS_STD_RANGES()
    namespace schedule_all_ {
      template <class Rcvr>
      auto get_allocator(const Rcvr& rcvr) {
        if constexpr (__callable<get_allocator_t, env_of_t<Rcvr>>) {
          return STDEXEC::get_allocator(STDEXEC::get_env(rcvr));
        } else {
          return std::allocator<char>{};
        }
      }

      template <class Receiver>
      using allocator_of_t = decltype(get_allocator(__declval<Receiver>()));

      template <class Range>
      struct operation_base {
        Range range_;
        _static_thread_pool& pool_;
        std::mutex start_mutex_{};
        bool has_started_{false};
        __intrusive_queue<&task_base::next> tasks_{};
        std::size_t tasks_size_{};
        __std::atomic<std::size_t> countdown_{std::ranges::size(range_)};
      };

      template <class Range, class ItemReceiver>
      class item_operation : task_base {
        static void execute_(task_base* base, std::uint32_t /* tid */) noexcept {
          auto op = static_cast<item_operation*>(base);
          STDEXEC::set_value(static_cast<ItemReceiver&&>(op->item_receiver_), *op->it_);
        }

        ItemReceiver item_receiver_;
        std::ranges::iterator_t<Range> it_;
        operation_base<Range>* parent_;

       public:
        item_operation(
          ItemReceiver&& item_receiver,
          std::ranges::iterator_t<Range> it,
          operation_base<Range>* parent)
          : task_base{.execute_ = execute_}
          , item_receiver_(static_cast<ItemReceiver&&>(item_receiver))
          , it_(it)
          , parent_(parent) {
        }

        void start() & noexcept {
          std::unique_lock lock{parent_->start_mutex_};
          if (!parent_->has_started_) {
            parent_->tasks_.push_back(static_cast<task_base*>(this));
            parent_->tasks_size_ += 1;
          } else {
            lock.unlock();
            parent_->pool_.enqueue(static_cast<task_base*>(this));
          }
        }
      };

      template <class Range>
      struct item_sender {
        using sender_concept = sender_t;
        using completion_signatures =
          STDEXEC::completion_signatures<set_value_t(std::ranges::range_reference_t<Range>)>;

        operation_base<Range>* op_;
        std::ranges::iterator_t<Range> it_;

        struct attrs {
          _static_thread_pool* pool_;

          auto query(get_completion_scheduler_t<set_value_t>, __ignore = {}) noexcept
            -> _static_thread_pool::scheduler {
            return pool_->get_scheduler();
          }
        };

        auto get_env() const noexcept -> attrs {
          return {&op_->pool_};
        }

        template <receiver ItemReceiver>
          requires receiver_of<ItemReceiver, completion_signatures>
        auto connect(ItemReceiver rcvr) const noexcept -> item_operation<Range, ItemReceiver> {
          return {static_cast<ItemReceiver&&>(rcvr), it_, op_};
        }
      };

      template <class Range, class Receiver>
      struct operation_base_with_receiver : operation_base<Range> {
        Receiver rcvr_;

        operation_base_with_receiver(Range range, _static_thread_pool& pool, Receiver rcvr)
          : operation_base<Range>{range, pool}
          , rcvr_(static_cast<Receiver&&>(rcvr)) {
        }
      };

      template <class Range, class Receiver>
      struct next_receiver {
        using receiver_concept = receiver_t;

        void set_value() noexcept {
          std::size_t countdown = op_->countdown_.fetch_sub(1, __std::memory_order_relaxed);
          if (countdown == 1) {
            STDEXEC::set_value(static_cast<Receiver&&>(op_->rcvr_));
          }
        }

        void set_stopped() noexcept {
          std::size_t countdown = op_->countdown_.fetch_sub(1, __std::memory_order_relaxed);
          if (countdown == 1) {
            STDEXEC::set_value(static_cast<Receiver&&>(op_->rcvr_));
          }
        }

        auto get_env() const noexcept -> env_of_t<Receiver> {
          return STDEXEC::get_env(op_->rcvr_);
        }

        operation_base_with_receiver<Range, Receiver>* op_;
      };

      template <class Range, class Receiver>
      class operation : operation_base_with_receiver<Range, Receiver> {
        using allocator_t = allocator_of_t<const Receiver&>;
        using item_sender_t = item_sender<Range>;
        using next_sender_t = next_sender_of_t<Receiver, item_sender_t>;
        using next_receiver_t = next_receiver<Range, Receiver>;
        using item_operation_t = connect_result_t<next_sender_t, next_receiver_t>;

        using item_allocator_t = std::allocator_traits<allocator_t>::template rebind_alloc<
          STDEXEC::__manual_lifetime<item_operation_t>
        >;

        std::vector<__manual_lifetime<item_operation_t>, item_allocator_t> items_;

       public:
        operation(Range range, _static_thread_pool& pool, Receiver rcvr)
          : operation_base_with_receiver<Range, Receiver>{
              std::move(range),
              pool,
              static_cast<Receiver&&>(rcvr)}
          , items_(std::ranges::size(this->range_), item_allocator_t(get_allocator(this->rcvr_))) {
        }

        ~operation() {
          if (this->has_started_) {
            for (auto& item: items_) {
              item.__destroy();
            }
          }
        }

        void start() & noexcept {
          std::size_t size = items_.size();
          std::size_t nthreads = this->pool_.available_parallelism();
          bwos_params params = this->pool_.params();
          std::size_t local_size = params.blockSize * params.numBlocks;
          std::size_t chunk_size = (std::min) (size / nthreads, local_size * nthreads);
          auto& remote_queue = *this->pool_.get_remote_queue();
          auto it = std::ranges::begin(this->range_);
          std::size_t i0 = 0;
          while (i0 + chunk_size < size) {
            for (std::size_t i = i0; i < i0 + chunk_size; ++i) {
              items_[i].__construct_from(
                STDEXEC::connect,
                set_next(this->rcvr_, item_sender_t{this, it + i}),
                next_receiver_t{this});
              STDEXEC::start(items_[i].__get());
            }

            std::unique_lock lock{this->start_mutex_};
            this->pool_.bulk_enqueue(remote_queue, std::move(this->tasks_), this->tasks_size_);
            lock.unlock();
            i0 += chunk_size;
          }
          for (std::size_t i = i0; i < size; ++i) {
            items_[i].__construct_from(
              STDEXEC::connect,
              set_next(this->rcvr_, item_sender_t{this, it + i}),
              next_receiver_t{this});
            STDEXEC::start(items_[i].__get());
          }
          std::unique_lock lock{this->start_mutex_};
          this->has_started_ = true;
          this->pool_.bulk_enqueue(remote_queue, std::move(this->tasks_), this->tasks_size_);
        }
      };

      template <class Range>
      class sequence {
        Range range_;
        _static_thread_pool* pool_;

       public:
        using sender_concept = sequence_sender_t;

        using completion_signatures = STDEXEC::completion_signatures<
          set_value_t(),
          set_error_t(std::exception_ptr),
          set_stopped_t()
        >;

        using item_types = exec::item_types<item_sender<Range>>;

        sequence(Range range, _static_thread_pool& pool)
          : range_(static_cast<Range&&>(range))
          , pool_(&pool) {
        }

        template <exec::sequence_receiver_of<item_types> Receiver>
        auto subscribe(Receiver rcvr) && noexcept -> operation<Range, Receiver> {
          return {static_cast<Range&&>(range_), *pool_, static_cast<Receiver&&>(rcvr)};
        }

        template <exec::sequence_receiver_of<item_types> Receiver>
          requires __decay_copyable<Range const &>
        auto subscribe(Receiver rcvr) const & noexcept -> operation<Range, Receiver> {
          return {range_, *pool_, static_cast<Receiver&&>(rcvr)};
        }
      };
    } // namespace schedule_all_

    struct schedule_all_t;
#endif
  } // namespace _pool_

  struct static_thread_pool : private _pool_::_static_thread_pool {
#if STDEXEC_HAS_STD_RANGES()
    friend struct _pool_::schedule_all_t;
#endif
    using task_base = _pool_::task_base;

    static_thread_pool() = default;

    static_thread_pool(
      std::uint32_t thread_count,
      bwos_params params = {},
      numa_policy numa = get_numa_policy())
      : _pool_::_static_thread_pool(thread_count, params, std::move(numa)) {
    }

    // struct scheduler;
    using _pool_::_static_thread_pool::scheduler;

    // scheduler get_scheduler() noexcept;
    using _pool_::_static_thread_pool::get_scheduler;

    // scheduler get_scheduler_on_thread(std::size_t thread_index) noexcept;
    using _pool_::_static_thread_pool::get_scheduler_on_thread;

    // scheduler get_constrained_scheduler(const nodemask& constraints) noexcept;
    using _pool_::_static_thread_pool::get_constrained_scheduler;

    // void request_stop() noexcept;
    using _pool_::_static_thread_pool::request_stop;

    // std::uint32_t available_parallelism() const;
    using _pool_::_static_thread_pool::available_parallelism;

    // bwos_params params() const;
    using _pool_::_static_thread_pool::params;
  };

#if STDEXEC_HAS_STD_RANGES()
  namespace _pool_ {
    struct schedule_all_t {
      template <class Range>
      auto operator()(static_thread_pool& pool, Range&& range) const
        -> schedule_all_::sequence<__decay_t<Range>> {
        return {static_cast<Range&&>(range), pool};
      }
    };
  } // namespace _pool_

  inline constexpr _pool_::schedule_all_t schedule_all{};
#endif

} // namespace exec
