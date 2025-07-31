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

#include "../stdexec/execution.hpp"
#include "../stdexec/__detail/__config.hpp"
#include "../stdexec/__detail/__intrusive_queue.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__manual_lifetime.hpp"
#include "__detail/__atomic_intrusive_queue.hpp"
#include "__detail/__bwos_lifo_queue.hpp"
#include "__detail/__xorshift.hpp"
#include "__detail/__numa.hpp"

#include "sequence_senders.hpp"
#include "sequence/iterate.hpp"

#include <algorithm>
#include <atomic>
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

  namespace _pool_ {
    using namespace stdexec;

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
    auto
      even_share(Shape n, std::size_t rank, std::size_t size) noexcept -> std::pair<Shape, Shape> {
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
      struct sequence {
        class __t;
      };
    } // namespace schedule_all_
#endif

    template <class>
    struct not_a_sender {
      using sender_concept = sender_t;
    };

    struct task_base {
      task_base* next = nullptr;
      void (*__execute)(task_base*, std::uint32_t tid) noexcept = nullptr;
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
      std::atomic<remote_queue*> head_;
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
        remote_queue* head = head_.load(std::memory_order_acquire);
        while (head != tail_) {
          remote_queue* tmp = std::exchange(head, head->next_);
          delete tmp;
        }
      }

      auto pop_all_reversed(std::size_t tid) noexcept -> __intrusive_queue<&task_base::next> {
        remote_queue* head = head_.load(std::memory_order_acquire);
        __intrusive_queue<&task_base::next> tasks{};
        while (head != nullptr) {
          tasks.append(head->queues_[tid].pop_all_reversed());
          head = head->next_;
        }
        return tasks;
      }

      auto get() -> remote_queue* {
        thread_local std::thread::id this_id = std::this_thread::get_id();
        remote_queue* head = head_.load(std::memory_order_acquire);
        remote_queue* queue = head;
        while (queue != tail_) {
          if (queue->id_ == this_id) {
            return queue;
          }
          queue = queue->next_;
        }
        auto* new_head = new remote_queue{head, nthreads_};
        while (!head_.compare_exchange_weak(head, new_head, std::memory_order_acq_rel)) {
          new_head->next_ = head;
        }
        return new_head;
      }
    };

    class static_thread_pool_ {
      template <class ReceiverId>
      struct operation {
        using Receiver = stdexec::__t<ReceiverId>;
        class __t;
      };

      struct schedule_tag {
        // TODO: code to reconstitute a static_thread_pool_ schedule sender
      };

      template <class SenderId, bool parallelize, std::integral Shape, class Fun>
      struct bulk_sender {
        using Sender = stdexec::__t<SenderId>;
        struct __t;
      };

      template <sender Sender, bool parallelize, std::integral Shape, class Fun>
      using bulk_sender_t = __t<bulk_sender<__id<__decay_t<Sender>>, parallelize, Shape, Fun>>;

#if STDEXEC_MSVC()
      // MSVCBUG https://developercommunity.visualstudio.com/t/Alias-template-with-pack-expansion-in-no/10437850

      template <class... Args>
      struct __bulk_non_throwing {
        using __t = __decayed_std_tuple<Args...>;
        static constexpr bool __v = noexcept(__t(std::declval<Args>()...));
      };
#endif

      template <class Fun, class Shape, class... Args>
        requires __callable<Fun, Shape, Shape, Args&...>
      using bulk_non_throwing = __mbool<
        // If function invocation doesn't throw
        __nothrow_callable<Fun, Shape, Shape, Args&...> &&
      // and emplacing a tuple doesn't throw
#if STDEXEC_MSVC()
        __bulk_non_throwing<Args...>::__v
#else
        noexcept(__decayed_std_tuple<Args...>(std::declval<Args>()...))
#endif
        // there's no need to advertise completion with `exception_ptr`
      >;

      template <
        class CvrefSender,
        class Receiver,
        bool parallelize,
        class Shape,
        class Fun,
        bool MayThrow
      >
      struct bulk_shared_state;

      template <
        class CvrefSenderId,
        class ReceiverId,
        bool parallelize,
        class Shape,
        class Fun,
        bool MayThrow
      >
      struct bulk_receiver {
        using CvrefSender = __cvref_t<CvrefSenderId>;
        using Receiver = stdexec::__t<ReceiverId>;
        struct __t;
      };

      template <
        class CvrefSender,
        class Receiver,
        bool parallelize,
        class Shape,
        class Fun,
        bool MayThrow
      >
      using bulk_receiver_t = __t<
        bulk_receiver<__cvref_id<CvrefSender>, __id<Receiver>, parallelize, Shape, Fun, MayThrow>
      >;

      template <
        class CvrefSenderId,
        class ReceiverId,
        bool parallelize,
        std::integral Shape,
        class Fun
      >
      struct bulk_op_state {
        using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
        using Receiver = stdexec::__t<ReceiverId>;
        struct __t;
      };

      template <class Sender, class Receiver, bool parallelize, std::integral Shape, class Fun>
      using bulk_op_state_t = __t<
        bulk_op_state<__id<__decay_t<Sender>>, __id<__decay_t<Receiver>>, parallelize, Shape, Fun>
      >;

      struct transform_bulk {
        template <class Data, class Sender>
        auto operator()(bulk_chunked_t, Data&& data, Sender&& sndr) {
          auto [pol, shape, fun] = static_cast<Data&&>(data);
          using policy_t = std::remove_cvref_t<decltype(pol.__get())>;
          constexpr bool parallelize = std::same_as<policy_t, parallel_policy>
                                    || std::same_as<policy_t, parallel_unsequenced_policy>;
          return bulk_sender_t<Sender, parallelize, decltype(shape), decltype(fun)>{
            pool_, static_cast<Sender&&>(sndr), shape, std::move(fun)};
        }

        static_thread_pool_& pool_;
      };

#if STDEXEC_HAS_STD_RANGES()
      struct transform_iterate {
        template <class Range>
        auto operator()(exec::iterate_t, Range&& range) -> __t<schedule_all_::sequence<Range>> {
          return {static_cast<Range&&>(range), pool_};
        }

        static_thread_pool_& pool_;
      };
#endif

      static auto _hardware_concurrency() noexcept -> unsigned int {
        unsigned int n = std::thread::hardware_concurrency();
        return n == 0 ? 1 : n;
      }

     public:
      struct domain : stdexec::default_domain {
        // For eager customization
        template <sender_expr_for<bulk_chunked_t> Sender>
        auto transform_sender(Sender&& sndr) const noexcept {
          if constexpr (__completes_on<Sender, static_thread_pool_::scheduler>) {
            auto sched = get_completion_scheduler<set_value_t>(get_env(sndr));
            return __sexpr_apply(static_cast<Sender&&>(sndr), transform_bulk{*sched.pool_});
          } else {
            static_assert(
              __completes_on<Sender, static_thread_pool_::scheduler>,
              "No static_thread_pool instance can be found in the sender's attributes "
              "on which to schedule bulk work.");
            return not_a_sender<__name_of<Sender>>();
          }
        }

        // transform the generic bulk_chunked sender into a parallel thread-pool bulk sender
        template <sender_expr_for<bulk_chunked_t> Sender, class Env>
        auto transform_sender(Sender&& sndr, const Env& env) const noexcept {
          if constexpr (__starts_on<Sender, static_thread_pool_::scheduler, Env>) {
            auto sched = stdexec::get_scheduler(env);
            return __sexpr_apply(static_cast<Sender&&>(sndr), transform_bulk{*sched.pool_});
          } else {
            static_assert(
              __starts_on<Sender, static_thread_pool_::scheduler, Env>,
              "No static_thread_pool instance can be found in the receiver's "
              "environment on which to schedule bulk work.");
            return not_a_sender<__name_of<Sender>>();
          }
        }

#if STDEXEC_HAS_STD_RANGES()
        template <sender_expr_for<exec::iterate_t> Sender>
        auto transform_sender(Sender&& sndr) const noexcept {
          auto sched = get_completion_scheduler<set_value_t>(get_env(sndr));
          return __sexpr_apply(static_cast<Sender&&>(sndr), transform_iterate{*sched.pool_});
        }

        template <sender_expr_for<exec::iterate_t> Sender, class Env>
          requires __callable<get_scheduler_t, Env>
        auto transform_sender(Sender&& sndr, const Env& env) const noexcept {
          auto sched = stdexec::get_scheduler(env);
          return __sexpr_apply(static_cast<Sender&&>(sndr), transform_iterate{*sched.pool_});
        }
#endif
      };

     public:
      static_thread_pool_();
      static_thread_pool_(
        std::uint32_t threadCount,
        bwos_params params = {},
        numa_policy numa = get_numa_policy());
      ~static_thread_pool_();

      struct scheduler {
       private:
        template <typename ReceiverId>
        friend struct operation;

        class _sender {
          struct env {
            static_thread_pool_& pool_;
            remote_queue* queue_;

            template <class CPO>
            auto query(get_completion_scheduler_t<CPO>) const noexcept
              -> static_thread_pool_::scheduler {
              return static_thread_pool_::scheduler{pool_, *queue_};
            }
          };

         public:
          using __t = _sender;
          using __id = _sender;
          using sender_concept = sender_t;
          template <class Receiver>
          using operation_t = stdexec::__t<operation<stdexec::__id<Receiver>>>;

          using completion_signatures =
            stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

          [[nodiscard]]
          auto get_env() const noexcept -> env {
            return env{.pool_ = pool_, .queue_ = queue_};
          }

          template <receiver Receiver>
          auto connect(Receiver rcvr) const -> operation_t<Receiver> {
            return operation_t<Receiver>{
              pool_, queue_, static_cast<Receiver&&>(rcvr), threadIndex_, constraints_};
          }

         private:
          friend struct static_thread_pool_::scheduler;

          explicit _sender(
            static_thread_pool_& pool,
            remote_queue* queue,
            std::size_t threadIndex,
            const nodemask& constraints) noexcept
            : pool_(pool)
            , queue_(queue)
            , threadIndex_(threadIndex)
            , constraints_(constraints) {
          }

          static_thread_pool_& pool_;
          remote_queue* queue_;
          std::size_t threadIndex_{std::numeric_limits<std::size_t>::max()};
          nodemask constraints_{};
        };

        friend class static_thread_pool_;

        explicit scheduler(
          static_thread_pool_& pool,
          const nodemask* mask = &nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{pool.get_remote_queue()}
          , nodemask_{mask} {
        }

        explicit scheduler(
          static_thread_pool_& pool,
          remote_queue& queue,
          const nodemask* mask = &nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{&queue}
          , nodemask_{mask} {
        }

        explicit scheduler(
          static_thread_pool_& pool,
          remote_queue& queue,
          std::size_t threadIndex) noexcept
          : pool_(&pool)
          , queue_{&queue}
          , thread_idx_{threadIndex} {
        }

        static_thread_pool_* pool_;
        remote_queue* queue_;
        const nodemask* nodemask_ = &nodemask::any();
        std::size_t thread_idx_{std::numeric_limits<std::size_t>::max()};

       public:
        using __t = scheduler;
        using __id = scheduler;
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
        auto query(get_domain_t) const noexcept -> domain {
          return {};
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
      auto available_parallelism() const -> std::uint32_t {
        return threadCount_;
      }

      [[nodiscard]]
      auto params() const -> bwos_params {
        return params_;
      }

      void enqueue(task_base* task, const nodemask& contraints = nodemask::any()) noexcept;
      void enqueue(
        remote_queue& queue,
        task_base* task,
        const nodemask& contraints = nodemask::any()) noexcept;
      void enqueue(remote_queue& queue, task_base* task, std::size_t threadIndex) noexcept;

      //! Enqueue a contiguous span of tasks across task queues.
      //! Note: We use the concrete `TaskT` because we enqueue
      //! tasks `task + 0`, `task + 1`, etc. so std::span<task_base>
      //! wouldn't be correct.
      //! This is O(n_threads) on the calling thread.
      template <std::derived_from<task_base> TaskT>
      void bulk_enqueue(TaskT* task, std::uint32_t n_threads) noexcept;
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
          std::uint32_t queueIndex;
        };

        explicit thread_state(
          static_thread_pool_* pool,
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
        bool stopRequested_{false};
        std::vector<workstealing_victim> near_victims_{};
        std::vector<workstealing_victim> all_victims_{};
        std::atomic<state> state_;
        static_thread_pool_* pool_;
        xorshift rng_{};
      };

      void run(std::uint32_t index) noexcept;
      void join() noexcept;

      alignas(64) std::atomic<std::uint32_t> numActive_{};
      alignas(64) remote_queue_list remotes_;
      std::uint32_t threadCount_;
      std::uint32_t maxSteals_{threadCount_ + 1};
      bwos_params params_;
      std::vector<std::thread> threads_;
      std::vector<std::optional<thread_state>> threadStates_;
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

      std::vector<thread_index_by_numa_node> threadIndexByNumaNode_;

      [[nodiscard]]
      auto num_threads(int numa) const noexcept -> std::size_t;
      [[nodiscard]]
      auto num_threads(nodemask constraints) const noexcept -> std::size_t;
      [[nodiscard]]
      auto get_thread_index(int numa, std::size_t index) const noexcept -> std::size_t;
      auto random_thread_index_with_constraints(const nodemask& contraints) noexcept -> std::size_t;
    };

    inline static_thread_pool_::static_thread_pool_()
      : static_thread_pool_(_hardware_concurrency()) {
    }

    inline static_thread_pool_::static_thread_pool_(
      std::uint32_t threadCount,
      bwos_params params,
      numa_policy numa)
      : remotes_(threadCount)
      , threadCount_(threadCount)
      , params_(params)
      , threadStates_(threadCount)
      , numa_(std::move(numa)) {
      STDEXEC_ASSERT(threadCount > 0);

      for (std::uint32_t index = 0; index < threadCount; ++index) {
        threadStates_[index].emplace(this, index, params, numa_);
        threadIndexByNumaNode_.push_back(
          thread_index_by_numa_node{
            .numa_node = threadStates_[index]->numa_node(), .thread_index = index});
      }

      // NOLINTNEXTLINE(modernize-use-ranges) we still support platforms without the std::ranges algorithms
      std::sort(threadIndexByNumaNode_.begin(), threadIndexByNumaNode_.end());
      std::vector<workstealing_victim> victims{};
      for (auto& state: threadStates_) {
        victims.emplace_back(state->as_victim());
      }
      for (auto& state: threadStates_) {
        state->victims(victims);
      }
      threads_.reserve(threadCount);

      STDEXEC_TRY {
        numActive_.store(threadCount << 16u, std::memory_order_relaxed);
        for (std::uint32_t i = 0; i < threadCount; ++i) {
          threads_.emplace_back([this, i] { run(i); });
        }
      }
      STDEXEC_CATCH_ALL {
        request_stop();
        join();
        STDEXEC_THROW();
      }
    }

    inline static_thread_pool_::~static_thread_pool_() {
      request_stop();
      join();
    }

    inline void static_thread_pool_::request_stop() noexcept {
      for (auto& state: threadStates_) {
        state->request_stop();
      }
    }

    inline void static_thread_pool_::run(std::uint32_t threadIndex) noexcept {
      STDEXEC_ASSERT(threadIndex < threadCount_);
      // NOLINTNEXTLINE(bugprone-unused-return-value)
      numa_.bind_to_node(threadStates_[threadIndex]->numa_node());
      while (true) {
        // Make a blocking call to de-queue a task if we don't already have one.
        auto [task, queueIndex] = threadStates_[threadIndex]->pop();
        if (!task) {
          return; // pop() only returns null when request_stop() was called.
        }
        task->__execute(task, queueIndex);
      }
    }

    inline void static_thread_pool_::join() noexcept {
      for (auto& t: threads_) {
        t.join();
      }
      threads_.clear();
    }

    inline void
      static_thread_pool_::enqueue(task_base* task, const nodemask& constraints) noexcept {
      this->enqueue(*get_remote_queue(), task, constraints);
    }

    inline auto static_thread_pool_::num_threads(int numa) const noexcept -> std::size_t {
      thread_index_by_numa_node key{.numa_node = numa, .thread_index = 0};
      // NOLINTNEXTLINE(modernize-use-ranges) we still support platforms without the std::ranges algorithms
      auto it = std::lower_bound(threadIndexByNumaNode_.begin(), threadIndexByNumaNode_.end(), key);
      if (it == threadIndexByNumaNode_.end()) {
        return 0;
      }
      auto itEnd = std::upper_bound(it, threadIndexByNumaNode_.end(), key);
      return static_cast<std::size_t>(std::distance(it, itEnd));
    }

    inline auto
      static_thread_pool_::num_threads(nodemask constraints) const noexcept -> std::size_t {
      const std::size_t nNodes = static_cast<unsigned>(threadIndexByNumaNode_.back().numa_node + 1);
      std::size_t nThreads = 0;
      for (std::size_t nodeIndex = 0; nodeIndex < nNodes; ++nodeIndex) {
        if (!constraints[nodeIndex]) {
          continue;
        }

        nThreads += num_threads(static_cast<int>(nodeIndex));
      }
      return nThreads;
    }

    inline auto
      static_thread_pool_::get_thread_index(int nodeIndex, std::size_t threadIndex) const noexcept
      -> std::size_t {
      thread_index_by_numa_node key{.numa_node = nodeIndex, .thread_index = 0};
      // NOLINTNEXTLINE(modernize-use-ranges) we still support platforms without the std::ranges algorithms
      auto it = std::lower_bound(threadIndexByNumaNode_.begin(), threadIndexByNumaNode_.end(), key);
      STDEXEC_ASSERT(it != threadIndexByNumaNode_.end());
      std::advance(it, threadIndex);
      return it->thread_index;
    }

    inline auto static_thread_pool_::random_thread_index_with_constraints(
      const nodemask& constraints) noexcept -> std::size_t {
      thread_local std::uint64_t startIndex{std::uint64_t(std::random_device{}())};
      startIndex += 1;
      std::size_t targetIndex = startIndex % threadCount_;
      std::size_t nThreads = num_threads(constraints);
      if (nThreads != 0) {
        for (std::size_t nodeIndex = 0; nodeIndex < numa_.num_nodes(); ++nodeIndex) {
          if (!constraints[nodeIndex]) {
            continue;
          }
          std::size_t nThreads = num_threads(static_cast<int>(nodeIndex));
          if (targetIndex < nThreads) {
            return get_thread_index(static_cast<int>(nodeIndex), targetIndex);
          }
          targetIndex -= nThreads;
        }
      }
      return targetIndex;
    }

    inline void static_thread_pool_::enqueue(
      remote_queue& queue,
      task_base* task,
      const nodemask& constraints) noexcept {
      static thread_local std::thread::id this_id = std::this_thread::get_id();
      remote_queue* correct_queue = this_id == queue.id_ ? &queue : get_remote_queue();
      std::size_t idx = correct_queue->index_;
      if (idx < threadStates_.size()) {
        auto this_node = static_cast<std::size_t>(threadStates_[idx]->numa_node());
        if (constraints[this_node]) {
          threadStates_[idx]->push_local(task);
          return;
        }
      }

      const std::size_t threadIndex = random_thread_index_with_constraints(constraints);
      queue.queues_[threadIndex].push_front(task);
      threadStates_[threadIndex]->notify();
    }

    inline void static_thread_pool_::enqueue(
      remote_queue& queue,
      task_base* task,
      std::size_t threadIndex) noexcept {
      threadIndex %= threadCount_;
      queue.queues_[threadIndex].push_front(task);
      threadStates_[threadIndex]->notify();
    }

    template <std::derived_from<task_base> TaskT>
    void static_thread_pool_::bulk_enqueue(TaskT* task, std::uint32_t n_threads) noexcept {
      auto& queue = *this->get_remote_queue();
      for (std::uint32_t i = 0; i < n_threads; ++i) {
        std::uint32_t index = i % this->available_parallelism();
        queue.queues_[index].push_front(task + i);
        threadStates_[index]->notify();
      }
      // At this point the calling thread can exit and the pool will take over.
      // Ultimately, the last completing thread passes the result forward.
      // See `if (is_last_thread)` above.
    }

    inline void static_thread_pool_::bulk_enqueue(
      remote_queue& queue,
      __intrusive_queue<&task_base::next> tasks,
      std::size_t tasks_size,
      const nodemask& constraints) noexcept {
      static thread_local std::thread::id this_id = std::this_thread::get_id();
      remote_queue* correct_queue = this_id == queue.id_ ? &queue : get_remote_queue();
      std::size_t idx = correct_queue->index_;
      if (idx < threadStates_.size()) {
        auto this_node = static_cast<std::size_t>(threadStates_[idx]->numa_node());
        if (constraints[this_node]) {
          threadStates_[idx]->push_local(std::move(tasks));
          return;
        }
      }

      std::size_t nThreads = available_parallelism();
      for (std::size_t i = 0; i < nThreads; ++i) {
        auto [i0, iEnd] =
          even_share(tasks_size, static_cast<std::uint32_t>(i), available_parallelism());
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
      bwos::lifo_queue<task_base*, numa_allocator<task_base*>>& local_queue) {
      auto last = local_queue.push_back(pending_queue.begin(), pending_queue.end());
      __intrusive_queue<&task_base::next> tmp{};
      tmp.splice(tmp.begin(), pending_queue, pending_queue.begin(), last);
      tmp.clear();
    }

    inline auto static_thread_pool_::thread_state::try_remote()
      -> static_thread_pool_::thread_state::pop_result {
      pop_result result{.task = nullptr, .queueIndex = index_};
      __intrusive_queue<&task_base::next> remotes = pool_->remotes_.pop_all_reversed(index_);
      pending_queue_.append(std::move(remotes));
      if (!pending_queue_.empty()) {
        move_pending_to_local(pending_queue_, local_queue_);
        result.task = local_queue_.pop_back();
      }

      return result;
    }

    inline auto static_thread_pool_::thread_state::try_pop()
      -> static_thread_pool_::thread_state::pop_result {
      pop_result result{.task = nullptr, .queueIndex = index_};
      result.task = local_queue_.pop_back();
      if (result.task) [[likely]] {
        return result;
      }
      return try_remote();
    }

    inline auto static_thread_pool_::thread_state::try_steal(std::span<workstealing_victim> victims)
      -> static_thread_pool_::thread_state::pop_result {
      if (victims.empty()) {
        return {.task = nullptr, .queueIndex = index_};
      }
      std::uniform_int_distribution<std::uint32_t> dist(
        0, static_cast<std::uint32_t>(victims.size() - 1));
      std::uint32_t victimIndex = dist(rng_);
      auto& v = victims[victimIndex];
      return {.task = v.try_steal(), .queueIndex = v.index()};
    }

    inline auto static_thread_pool_::thread_state::try_steal_near()
      -> static_thread_pool_::thread_state::pop_result {
      return try_steal(near_victims_);
    }

    inline auto static_thread_pool_::thread_state::try_steal_any()
      -> static_thread_pool_::thread_state::pop_result {
      return try_steal(all_victims_);
    }

    inline void static_thread_pool_::thread_state::push_local(task_base* task) {
      if (!local_queue_.push_back(task)) {
        pending_queue_.push_back(task);
      }
    }

    inline void
      static_thread_pool_::thread_state::push_local(__intrusive_queue<&task_base::next>&& tasks) {
      pending_queue_.prepend(std::move(tasks));
    }

    inline void static_thread_pool_::thread_state::set_sleeping() {
      pool_->numActive_.fetch_sub(1u << 16u, std::memory_order_relaxed);
    }

    // wakeup a worker thread and maintain the invariant that we always one active thief as long as a potential victim is awake
    inline void static_thread_pool_::thread_state::clear_sleeping() {
      const std::uint32_t numActive = pool_->numActive_
                                        .fetch_add(1u << 16u, std::memory_order_relaxed);
      if (numActive == 0) {
        notify_one_sleeping();
      }
    }

    inline void static_thread_pool_::thread_state::set_stealing() {
      const std::uint32_t diff = 1u - (1u << 16u);
      pool_->numActive_.fetch_add(diff, std::memory_order_relaxed);
    }

    // put a thief to sleep but maintain the invariant that we always have one active thief as long as a potential victim is awake
    inline void static_thread_pool_::thread_state::clear_stealing() {
      constexpr std::uint32_t diff = 1 - (1u << 16u);
      const std::uint32_t numActive = pool_->numActive_.fetch_sub(diff, std::memory_order_relaxed);
      const std::uint32_t numVictims = numActive >> 16u;
      const std::uint32_t numThiefs = numActive & 0xffffu;
      if (numThiefs == 1 && numVictims != 0) {
        notify_one_sleeping();
      }
    }

    inline void static_thread_pool_::thread_state::notify_one_sleeping() {
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

    inline auto
      static_thread_pool_::thread_state::pop() -> static_thread_pool_::thread_state::pop_result {
      pop_result result = try_pop();
      while (!result.task) {
        set_stealing();
        for (std::size_t i = 0; i < pool_->maxSteals_; ++i) {
          result = try_steal_near();
          if (result.task) {
            clear_stealing();
            return result;
          }
        }

        for (std::size_t i = 0; i < pool_->maxSteals_; ++i) {
          result = try_steal_any();
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
          set_sleeping();
          cv_.wait(lock);
          lock.unlock();
          clear_sleeping();
        }
        if (lock.owns_lock()) {
          lock.unlock();
        }
        state_.store(state::running, std::memory_order_relaxed);
        result = try_pop();
      }
      return result;
    }

    inline auto static_thread_pool_::thread_state::notify() -> bool {
      if (state_.exchange(state::notified, std::memory_order_relaxed) == state::sleeping) {
        {
          std::lock_guard lock{mut_};
        }
        cv_.notify_one();
        return true;
      }
      return false;
    }

    inline void static_thread_pool_::thread_state::request_stop() {
      {
        std::lock_guard lock{mut_};
        stopRequested_ = true;
      }
      cv_.notify_one();
    }

    template <typename ReceiverId>
    class static_thread_pool_::operation<ReceiverId>::__t : public task_base {
      using __id = operation;
      friend static_thread_pool_::scheduler::_sender;

      static_thread_pool_& pool_;
      remote_queue* queue_;
      Receiver rcvr_;
      std::size_t threadIndex_{};
      nodemask constraints_{};

      explicit __t(
        static_thread_pool_& pool,
        remote_queue* queue,
        Receiver rcvr,
        std::size_t tid,
        const nodemask& constraints)
        : pool_(pool)
        , queue_(queue)
        , rcvr_(static_cast<Receiver&&>(rcvr))
        , threadIndex_{tid}
        , constraints_{constraints} {
        this->__execute = [](task_base* t, const std::uint32_t /* tid */) noexcept {
          auto& op = *static_cast<__t*>(t);
          auto stoken = get_stop_token(get_env(op.rcvr_));
          if constexpr (stdexec::unstoppable_token<decltype(stoken)>) {
            stdexec::set_value(static_cast<Receiver&&>(op.rcvr_));
          } else {
            if (stoken.stop_requested()) {
              stdexec::set_stopped(static_cast<Receiver&&>(op.rcvr_));
            } else {
              stdexec::set_value(static_cast<Receiver&&>(op.rcvr_));
            }
          }
        };
      }

      void enqueue_(task_base* op) const {
        if (threadIndex_ < pool_.available_parallelism()) {
          pool_.enqueue(*queue_, op, threadIndex_);
        } else {
          pool_.enqueue(*queue_, op, constraints_);
        }
      }

     public:
      void start() & noexcept {
        enqueue_(this);
      }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // What follows is the implementation for parallel bulk execution on static_thread_pool_.
    template <class SenderId, bool parallelize, std::integral Shape, class Fun>
    struct static_thread_pool_::bulk_sender<SenderId, parallelize, Shape, Fun>::__t {
      using __id = bulk_sender;
      using sender_concept = sender_t;

      static_thread_pool_& pool_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      template <class Sender, class... Env>
      using with_error_invoke_t = __if_c<
        __v<__value_types_t<
          __completion_signatures_of_t<Sender, Env...>,
          __mbind_front_q<bulk_non_throwing, Fun, Shape>,
          __q<__mand>
        >>,
        completion_signatures<>,
        __eptr_completion
      >;

      template <class... Tys>
      using set_value_t = completion_signatures<set_value_t(stdexec::__decay_t<Tys>...)>;

      template <class Self, class... Env>
      using __completions_t = stdexec::transform_completion_signatures<
        __completion_signatures_of_t<__copy_cvref_t<Self, Sender>, Env...>,
        with_error_invoke_t<__copy_cvref_t<Self, Sender>, Env...>,
        set_value_t
      >;

      template <class Self, class Receiver>
      using bulk_op_state_t = stdexec::__t<
        bulk_op_state<__cvref_id<Self, Sender>, stdexec::__id<Receiver>, parallelize, Shape, Fun>
      >;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, __completions_t<Self, env_of_t<Receiver>>>
      static auto connect(Self&& self, Receiver rcvr) noexcept(__nothrow_constructible_from<
                                                               bulk_op_state_t<Self, Receiver>,
                                                               static_thread_pool_&,
                                                               Shape,
                                                               Fun,
                                                               Sender,
                                                               Receiver
      >) -> bulk_op_state_t<Self, Receiver> {
        return bulk_op_state_t<Self, Receiver>{
          self.pool_,
          self.shape_,
          self.fun_,
          static_cast<Self&&>(self).sndr_,
          static_cast<Receiver&&>(rcvr)};
      }

      template <__decays_to<__t> Self, class... Env>
      static auto get_completion_signatures(Self&&, Env&&...) -> __completions_t<Self, Env...> {
        return {};
      }

      auto get_env() const noexcept -> env_of_t<const Sender&> {
        return stdexec::get_env(sndr_);
      }
    };

    //! The customized operation state for `stdexec::bulk` operations
    template <
      class CvrefSender,
      class Receiver,
      bool parallelize,
      class Shape,
      class Fun,
      bool MayThrow
    >
    struct static_thread_pool_::bulk_shared_state {
      //! The actual `bulk_task` holds a pointer to the shared state
      //! and its `__execute` function reads from that shared state.
      struct bulk_task : task_base {
        bulk_shared_state* sh_state_;

        bulk_task(bulk_shared_state* sh_state)
          : sh_state_(sh_state) {
          this->__execute = [](task_base* t, const std::uint32_t tid) noexcept {
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
              stdexec::set_value(static_cast<Receiver&&>(sh_state.rcvr_), std::move(args)...);
            };

            if constexpr (MayThrow) {
              STDEXEC_TRY {
                sh_state.apply(computation);
              }
              STDEXEC_CATCH_ALL {
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
        CvrefSender,
        env_of_t<Receiver>,
        __q<__decayed_std_tuple>,
        __q<__nullable_std_variant>
      >;

      variant_t data_;
      static_thread_pool_& pool_;
      Receiver rcvr_;
      Shape shape_;
      Fun fun_;

      std::atomic<std::uint32_t> finished_threads_{0};
      std::atomic<std::uint32_t> thread_with_exception_{0};
      std::exception_ptr exception_;
      std::vector<bulk_task> tasks_;

      //! The number of agents required is the minimum of `shape_` and the available parallelism.
      //! That is, we don't need an agent for each of the shape values.
      [[nodiscard]]
      auto num_agents_required() const -> std::uint32_t {
        if constexpr (parallelize) {
          return static_cast<std::uint32_t>(
            std::min(shape_, static_cast<Shape>(pool_.available_parallelism())));
        } else {
          return static_cast<std::uint32_t>(1);
        }
      }

      template <class F>
      void apply(F f) {
        std::visit(
          [&](auto& tupl) -> void {
            if constexpr (same_as<__decay_t<decltype(tupl)>, std::monostate>) {
              std::terminate();
            } else {
              std::apply([&](auto&... args) -> void { f(args...); }, tupl);
            }
          },
          data_);
      }

      //! Construct from a pool, receiver, shape, and function.
      //! Allocates O(min(shape, available_parallelism())) memory.
      bulk_shared_state(static_thread_pool_& pool, Receiver rcvr, Shape shape, Fun fun)
        : pool_{pool}
        , rcvr_{static_cast<Receiver&&>(rcvr)}
        , shape_{shape}
        , fun_{fun}
        , thread_with_exception_{num_agents_required()}
        , tasks_{num_agents_required(), {this}} {
      }
    };

    //! A customized receiver to allow parallel execution of `stdexec::bulk` operations:
    template <
      class CvrefSenderId,
      class ReceiverId,
      bool parallelize,
      class Shape,
      class Fun,
      bool MayThrow
    >
    struct static_thread_pool_::bulk_receiver<
      CvrefSenderId,
      ReceiverId,
      parallelize,
      Shape,
      Fun,
      MayThrow
    >::__t {
      using __id = bulk_receiver;
      using receiver_concept = receiver_t;

      using shared_state =
        bulk_shared_state<CvrefSender, Receiver, parallelize, Shape, Fun, MayThrow>;

      shared_state& shared_state_;

      void enqueue() noexcept {
        shared_state_.pool_
          .bulk_enqueue(shared_state_.tasks_.data(), shared_state_.num_agents_required());
      }

      template <class... As>
      void set_value(As&&... as) noexcept {
        using tuple_t = __decayed_std_tuple<As...>;

        shared_state& state = shared_state_;

        if constexpr (MayThrow) {
          STDEXEC_TRY {
            state.data_.template emplace<tuple_t>(static_cast<As&&>(as)...);
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(std::move(state.rcvr_), std::current_exception());
          }
        } else {
          state.data_.template emplace<tuple_t>(static_cast<As&&>(as)...);
        }

        if (state.shape_) {
          enqueue();
        } else {
          state.apply(
            [&](auto&... args) { stdexec::set_value(std::move(state.rcvr_), std::move(args)...); });
        }
      }

      template <class Error>
      void set_error(Error&& error) noexcept {
        shared_state& state = shared_state_;
        stdexec::set_error(static_cast<Receiver&&>(state.rcvr_), static_cast<Error&&>(error));
      }

      void set_stopped() noexcept {
        shared_state& state = shared_state_;
        stdexec::set_stopped(static_cast<Receiver&&>(state.rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<Receiver> {
        return stdexec::get_env(shared_state_.rcvr_);
      }
    };

    template <class CvrefSenderId, class ReceiverId, bool parallelize, std::integral Shape, class Fun>
    struct static_thread_pool_::bulk_op_state<
      CvrefSenderId,
      ReceiverId,
      parallelize,
      Shape,
      Fun
    >::__t {
      using __id = bulk_op_state;

      static constexpr bool may_throw = !__v<__value_types_of_t<
        CvrefSender,
        env_of_t<Receiver>,
        __mbind_front_q<bulk_non_throwing, Fun, Shape>,
        __q<__mand>
      >>;

      using bulk_rcvr = bulk_receiver_t<CvrefSender, Receiver, parallelize, Shape, Fun, may_throw>;
      using shared_state =
        bulk_shared_state<CvrefSender, Receiver, parallelize, Shape, Fun, may_throw>;
      using inner_op_state = connect_result_t<CvrefSender, bulk_rcvr>;

      shared_state shared_state_;

      inner_op_state inner_op_;

      void start() & noexcept {
        stdexec::start(inner_op_);
      }

      __t(static_thread_pool_& pool, Shape shape, Fun fun, CvrefSender&& sndr, Receiver rcvr)
        : shared_state_(pool, static_cast<Receiver&&>(rcvr), shape, fun)
        , inner_op_{stdexec::connect(static_cast<CvrefSender&&>(sndr), bulk_rcvr{shared_state_})} {
      }
    };

#if STDEXEC_HAS_STD_RANGES()
    namespace schedule_all_ {
      template <class Rcvr>
      auto get_allocator(const Rcvr& rcvr) {
        if constexpr (__callable<get_allocator_t, env_of_t<Rcvr>>) {
          return get_allocator(get_env(rcvr));
        } else {

          return std::allocator<char>{};
        }
      }

      template <class Receiver>
      using allocator_of_t = decltype(get_allocator(__declval<Receiver>()));

      template <class Range>
      struct operation_base {
        Range range_;
        static_thread_pool_& pool_;
        std::mutex start_mutex_{};
        bool has_started_{false};
        __intrusive_queue<&task_base::next> tasks_{};
        std::size_t tasks_size_{};
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
      };

      template <class Range>
      struct item_sender {
        struct __t {
          using __id = item_sender;
          using sender_concept = sender_t;
          using completion_signatures =
            stdexec::completion_signatures<set_value_t(std::ranges::range_reference_t<Range>)>;

          operation_base<Range>* op_;
          std::ranges::iterator_t<Range> it_;

          struct env {
            static_thread_pool_* pool_;

            auto query(get_completion_scheduler_t<set_value_t>) noexcept
              -> static_thread_pool_::scheduler {
              return pool_->get_scheduler();
            }
          };

          auto get_env() const noexcept -> env {
            return {op_->pool_};
          }

          template <receiver ItemReceiver>
            requires receiver_of<ItemReceiver, completion_signatures>
          auto connect(ItemReceiver rcvr) const noexcept
            -> stdexec::__t<item_operation<Range, stdexec::__id<ItemReceiver>>> {
            return {static_cast<ItemReceiver&&>(rcvr), it_, op_};
          }
        };
      };

      template <class Range, class Receiver>
      struct operation_base_with_receiver : operation_base<Range> {
        Receiver rcvr_;

        operation_base_with_receiver(Range range, static_thread_pool_& pool, Receiver rcvr)
          : operation_base<Range>{range, pool}
          , rcvr_(static_cast<Receiver&&>(rcvr)) {
        }
      };

      template <class Range, class ReceiverId>
      struct next_receiver {
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t {
          using __id = next_receiver;
          using receiver_concept = receiver_t;
          operation_base_with_receiver<Range, Receiver>* op_;

          void set_value() noexcept {
            std::size_t countdown = op_->countdown_.fetch_sub(1, std::memory_order_relaxed);
            if (countdown == 1) {
              stdexec::set_value(static_cast<Receiver&&>(op_->rcvr_));
            }
          }

          void set_stopped() noexcept {
            std::size_t countdown = op_->countdown_.fetch_sub(1, std::memory_order_relaxed);
            if (countdown == 1) {
              stdexec::set_value(static_cast<Receiver&&>(op_->rcvr_));
            }
          }

          auto get_env() const noexcept -> env_of_t<Receiver> {
            return stdexec::get_env(op_->rcvr_);
          }
        };
      };

      template <class Range, class ReceiverId>
      struct operation {
        using Receiver = stdexec::__t<ReceiverId>;

        class __t : operation_base_with_receiver<Range, Receiver> {
          using Allocator = allocator_of_t<const Receiver&>;
          using ItemSender = stdexec::__t<item_sender<Range>>;
          using NextSender = next_sender_of_t<Receiver, ItemSender>;
          using NextReceiver = stdexec::__t<next_receiver<Range, ReceiverId>>;
          using ItemOperation = connect_result_t<NextSender, NextReceiver>;

          using ItemAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<
            stdexec::__manual_lifetime<ItemOperation>
          >;

          std::vector<__manual_lifetime<ItemOperation>, ItemAllocator> items_;

         public:
          using __id = operation;

          __t(Range range, static_thread_pool_& pool, Receiver rcvr)
            : operation_base_with_receiver<
                Range,
                Receiver
              >{std::move(range), pool, static_cast<Receiver&&>(rcvr)}
            , items_(std::ranges::size(this->range_), ItemAllocator(get_allocator(this->rcvr_))) {
          }

          ~__t() {
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
            std::size_t localSize = params.blockSize * params.numBlocks;
            std::size_t chunkSize = std::min<std::size_t>(size / nthreads, localSize * nthreads);
            auto& remote_queue = *this->pool_.get_remote_queue();
            std::ranges::iterator_t<Range> it = std::ranges::begin(this->range_);
            std::size_t i0 = 0;
            while (i0 + chunkSize < size) {
              for (std::size_t i = i0; i < i0 + chunkSize; ++i) {
                items_[i].__construct_from([&] {
                  return stdexec::connect(
                    set_next(this->rcvr_, ItemSender{this, it + i}), NextReceiver{this});
                });
                stdexec::start(items_[i].__get());
              }

              std::unique_lock lock{this->start_mutex_};
              this->pool_.bulk_enqueue(remote_queue, std::move(this->tasks_), this->tasks_size_);
              lock.unlock();
              i0 += chunkSize;
            }
            for (std::size_t i = i0; i < size; ++i) {
              items_[i].__construct_from([&] {
                return stdexec::connect(
                  set_next(this->rcvr_, ItemSender{this, it + i}), NextReceiver{this});
              });
              stdexec::start(items_[i].__get());
            }
            std::unique_lock lock{this->start_mutex_};
            this->has_started_ = true;
            this->pool_.bulk_enqueue(remote_queue, std::move(this->tasks_), this->tasks_size_);
          }
        };
      };

      template <class Range>
      class sequence<Range>::__t {
        Range range_;
        static_thread_pool_* pool_;

       public:
        using __id = sequence;

        using sender_concept = sequence_sender_t;

        using completion_signatures = stdexec::completion_signatures<
          set_value_t(),
          set_error_t(std::exception_ptr),
          set_stopped_t()
        >;

        using item_types = exec::item_types<stdexec::__t<item_sender<Range>>>;

        __t(Range range, static_thread_pool_& pool)
          : range_(static_cast<Range&&>(range))
          , pool_(&pool) {
        }

        template <exec::sequence_receiver_of<item_types> Receiver>
        auto subscribe(Receiver rcvr) && noexcept
          -> stdexec::__t<operation<Range, stdexec::__id<Receiver>>> {
          return {static_cast<Range&&>(range_), *pool_, static_cast<Receiver&&>(rcvr)};
        }

        template <exec::sequence_receiver_of<item_types> Receiver>
          requires __decay_copyable<Range const &>
        auto subscribe(Receiver rcvr) const & noexcept
          -> stdexec::__t<operation<Range, stdexec::__id<Receiver>>> {
          return {range_, *pool_, static_cast<Receiver&&>(rcvr)};
        }
      };
    } // namespace schedule_all_

    struct schedule_all_t;
#endif
  } // namespace _pool_

  struct static_thread_pool : private _pool_::static_thread_pool_ {
#if STDEXEC_HAS_STD_RANGES()
    friend struct _pool_::schedule_all_t;
#endif
    using task_base = _pool_::task_base;

    static_thread_pool() = default;

    static_thread_pool(
      std::uint32_t threadCount,
      bwos_params params = {},
      numa_policy numa = get_numa_policy())
      : _pool_::static_thread_pool_(threadCount, params, std::move(numa)) {
    }

    // struct scheduler;
    using _pool_::static_thread_pool_::scheduler;

    // scheduler get_scheduler() noexcept;
    using _pool_::static_thread_pool_::get_scheduler;

    // scheduler get_scheduler_on_thread(std::size_t threadIndex) noexcept;
    using _pool_::static_thread_pool_::get_scheduler_on_thread;

    // scheduler get_constrained_scheduler(const nodemask& constraints) noexcept;
    using _pool_::static_thread_pool_::get_constrained_scheduler;

    // void request_stop() noexcept;
    using _pool_::static_thread_pool_::request_stop;

    // std::uint32_t available_parallelism() const;
    using _pool_::static_thread_pool_::available_parallelism;

    // bwos_params params() const;
    using _pool_::static_thread_pool_::params;
  };

#if STDEXEC_HAS_STD_RANGES()
  namespace _pool_ {
    struct schedule_all_t {
      template <class Range>
      auto operator()(static_thread_pool& pool, Range&& range) const
        -> stdexec::__t<schedule_all_::sequence<__decay_t<Range>>> {
        return {static_cast<Range&&>(range), pool};
      }
    };
  } // namespace _pool_

  inline constexpr _pool_::schedule_all_t schedule_all{};
#endif

} // namespace exec
