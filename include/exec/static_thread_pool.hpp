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
#include "./__detail/__numa.hpp"

#include "./sequence_senders.hpp"
#include "./sequence/iterate.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <vector>

namespace exec {
  struct bwos_params {
    std::size_t numBlocks{8};
    std::size_t blockSize{1024};
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
      using sender_concept = sender_t;
    };

    struct task_base {
      task_base* next;
      void (*__execute)(task_base*, std::uint32_t tid) noexcept;
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
        remote_queue* new_head = new remote_queue{head, nthreads_};
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

      template <class SenderId, std::integral Shape, class Fun>
      struct bulk_sender {
        using Sender = stdexec::__t<SenderId>;
        struct __t;
      };

      template <sender Sender, std::integral Shape, class Fun>
      using bulk_sender_t = __t<bulk_sender< __id<__decay_t<Sender>>, Shape, Fun>>;

#if STDEXEC_MSVC()
      // MSVCBUG https://developercommunity.visualstudio.com/t/Alias-template-with-pack-expansion-in-no/10437850

      template <class... Args>
      struct __bulk_non_throwing {
        using __t = __decayed_tuple<Args...>;
        static constexpr bool __v = noexcept(__t(std::declval<Args>()...));
      };
#endif

      template <class Fun, class Shape, class... Args>
        requires __callable<Fun, Shape, Args&...>
      using bulk_non_throwing = //
        __mbool<
          // If function invocation doesn't throw
          __nothrow_callable<Fun, Shape, Args&...> &&
      // and emplacing a tuple doesn't throw
#if STDEXEC_MSVC()
          __bulk_non_throwing<Args...>::__v
#else
          noexcept(__decayed_tuple<Args...>(std::declval<Args>()...))
#endif
          // there's no need to advertise completion with `exception_ptr`
          >;

      template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
      struct bulk_shared_state;

      template <class CvrefSenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
      struct bulk_receiver {
        using CvrefSender = __cvref_t<CvrefSenderId>;
        using Receiver = stdexec::__t<ReceiverId>;
        struct __t;
      };

      template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
      using bulk_receiver_t = __t<
        bulk_receiver< __cvref_id<CvrefSender>, __id<Receiver>, Shape, Fun, MayThrow>>;

      template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
      struct bulk_op_state {
        using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
        using Receiver = stdexec::__t<ReceiverId>;
        struct __t;
      };

      template <class Sender, class Receiver, std::integral Shape, class Fun>
      using bulk_op_state_t =
        __t<bulk_op_state< __id<__decay_t<Sender>>, __id<__decay_t<Receiver>>, Shape, Fun>>;

      struct transform_bulk {
        template <class Data, class Sender>
        auto operator()(bulk_t, Data&& data, Sender&& sndr) {
          auto [shape, fun] = (Data&&) data;
          return bulk_sender_t<Sender, decltype(shape), decltype(fun)>{
            pool_, (Sender&&) sndr, shape, std::move(fun)};
        }

        static_thread_pool_& pool_;
      };

#if STDEXEC_HAS_STD_RANGES()
      struct transform_iterate {
        template <class Range>
        __t<schedule_all_::sequence<Range>> operator()(exec::iterate_t, Range&& range) {
          return {static_cast<Range&&>(range), pool_};
        }

        static_thread_pool_& pool_;
      };
#endif

     public:
      struct domain {
        // For eager customization
        template <sender_expr_for<bulk_t> Sender>
        auto transform_sender(Sender&& sndr) const noexcept {
          if constexpr (__completes_on<Sender, static_thread_pool_::scheduler>) {
            auto sched = get_completion_scheduler<set_value_t>(get_env(sndr));
            return __sexpr_apply((Sender&&) sndr, transform_bulk{*sched.pool_});
          } else {
            static_assert(
              __completes_on<Sender, static_thread_pool_::scheduler>,
              "No static_thread_pool_ instance can be found in the sender's environment "
              "on which to schedule bulk work.");
            return not_a_sender<__name_of<Sender>>();
          }
          STDEXEC_UNREACHABLE();
        }

        // transform the generic bulk sender into a parallel thread-pool bulk sender
        template <sender_expr_for<bulk_t> Sender, class Env>
        auto transform_sender(Sender&& sndr, const Env& env) const noexcept {
          if constexpr (__completes_on<Sender, static_thread_pool_::scheduler>) {
            auto sched = get_completion_scheduler<set_value_t>(get_env(sndr));
            return __sexpr_apply((Sender&&) sndr, transform_bulk{*sched.pool_});
          } else if constexpr (__starts_on<Sender, static_thread_pool_::scheduler, Env>) {
            auto sched = stdexec::get_scheduler(env);
            return __sexpr_apply((Sender&&) sndr, transform_bulk{*sched.pool_});
          } else {
            static_assert( //
              __starts_on<Sender, static_thread_pool_::scheduler, Env>
                || __completes_on<Sender, static_thread_pool_::scheduler>,
              "No static_thread_pool_ instance can be found in the sender's or receiver's "
              "environment on which to schedule bulk work.");
            return not_a_sender<__name_of<Sender>>();
          }
          STDEXEC_UNREACHABLE();
        }

#if STDEXEC_HAS_STD_RANGES()
        template <sender_expr_for<exec::iterate_t> Sender>
        auto transform_sender(Sender&& sndr) const noexcept {
          auto sched = get_completion_scheduler<set_value_t>(get_env(sndr));
          return __sexpr_apply((Sender&&) sndr, transform_iterate{*sched.pool_});
        }

        template <sender_expr_for<exec::iterate_t> Sender, class Env>
          requires __callable<get_scheduler_t, Env>
        auto transform_sender(Sender&& sndr, const Env& env) const noexcept {
          auto sched = stdexec::get_scheduler(env);
          return __sexpr_apply((Sender&&) sndr, transform_iterate{*sched.pool_});
        }
#endif
      };

     public:
      static_thread_pool_();
      static_thread_pool_(
        std::uint32_t threadCount,
        bwos_params params = {},
        numa_policy* numa = get_numa_policy());
      ~static_thread_pool_();

      struct scheduler {
        using __t = scheduler;
        using __id = scheduler;
        bool operator==(const scheduler&) const = default;

       private:
        template <typename ReceiverId>
        friend struct operation;

        class sender {
         public:
          using __t = sender;
          using __id = sender;
          using sender_concept = sender_t;
          using completion_signatures = stdexec::completion_signatures< set_value_t(), set_stopped_t()>;
         private:
          template <class Receiver>
          using operation_t = stdexec::__t<operation<stdexec::__id<Receiver>>>;

          template <typename Receiver>
          auto make_operation_(Receiver rcvr) const -> operation_t<Receiver> {
            return operation_t<Receiver>{pool_, queue_, (Receiver&&) rcvr, threadIndex_, constraints_};
          }

          template <receiver Receiver>
          friend auto tag_invoke(connect_t, sender sndr, Receiver rcvr) -> operation_t<Receiver> {
            return sndr.make_operation_((Receiver&&) rcvr);
          }

          struct env {
            static_thread_pool_& pool_;
            remote_queue* queue_;

            template <class CPO>
            friend static_thread_pool_::scheduler
              tag_invoke(get_completion_scheduler_t<CPO>, const env& self) noexcept {
              return self.make_scheduler_();
            }

            static_thread_pool_::scheduler make_scheduler_() const {
              return static_thread_pool_::scheduler{pool_, *queue_};
            }
          };

          friend env tag_invoke(get_env_t, const sender& self) noexcept {
            return env{self.pool_, self.queue_};
          }

          friend struct static_thread_pool_::scheduler;

          explicit sender(
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

        sender make_sender_() const {
          return sender{*pool_, queue_, thread_idx_, nodemask_};
        }

        friend sender tag_invoke(schedule_t, const scheduler& sch) noexcept {
          return sch.make_sender_();
        }

        friend forward_progress_guarantee
          tag_invoke(get_forward_progress_guarantee_t, const static_thread_pool_&) noexcept {
          return forward_progress_guarantee::parallel;
        }

        friend domain tag_invoke(get_domain_t, scheduler) noexcept {
          return {};
        }

        friend class static_thread_pool_;

        explicit scheduler(
          static_thread_pool_& pool,
          const nodemask& mask = nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{pool.get_remote_queue()}
          , nodemask_{mask} {
        }

        explicit scheduler(
          static_thread_pool_& pool,
          remote_queue& queue,
          const nodemask& mask = nodemask::any()) noexcept
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
        nodemask nodemask_;
        std::size_t thread_idx_{std::numeric_limits<std::size_t>::max()};
      };

      scheduler get_scheduler() noexcept {
        return scheduler{*this};
      }

      scheduler get_scheduler_on_thread(std::size_t threadIndex) noexcept {
        return scheduler{*this, *get_remote_queue(), threadIndex};
      }

      scheduler get_constrained_scheduler(const nodemask& constraints) noexcept {
        return scheduler{*this, *get_remote_queue(), constraints};
      }

      remote_queue* get_remote_queue() noexcept {
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

      std::uint32_t available_parallelism() const {
        return threadCount_;
      }

      bwos_params params() const {
        return params_;
      }

      void enqueue(task_base* task, const nodemask& contraints = nodemask::any()) noexcept;
      void enqueue(
        remote_queue& queue,
        task_base* task,
        const nodemask& contraints = nodemask::any()) noexcept;
      void enqueue(remote_queue& queue, task_base* task, std::size_t threadIndex) noexcept;

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

        task_base* try_steal() noexcept {
          return queue_->steal_front();
        }

        std::uint32_t index() const noexcept {
          return index_;
        }

        int numa_node() const noexcept {
          return numa_node_;
        }

       private:
        bwos::lifo_queue<task_base*, numa_allocator<task_base*>>* queue_;
        std::uint32_t index_;
        int numa_node_;
      };

      struct thread_state_base {
        explicit thread_state_base(std::uint32_t index, numa_policy* numa) noexcept
          : index_(index)
          , numa_node_(numa->thread_index_to_node(index)) {
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
          numa_policy* numa) noexcept
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

        pop_result pop();
        void push_local(task_base* task);
        void push_local(__intrusive_queue<&task_base::next>&& tasks);

        bool notify();
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

        std::uint32_t index() const noexcept {
          return index_;
        }

        int numa_node() const noexcept {
          return numa_node_;
        }

        workstealing_victim as_victim() noexcept {
          return workstealing_victim{&local_queue_, index_, numa_node_};
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
        pop_result try_steal(std::span<workstealing_victim> victims);
        pop_result try_steal_near();
        pop_result try_steal_any();

        void notify_one_sleeping();
        void set_stealing();
        void clear_stealing();

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

      void run(std::uint32_t index, numa_policy* numa) noexcept;
      void join() noexcept;

      alignas(64) std::atomic<std::uint32_t> numThiefs_{};
      alignas(64) remote_queue_list remotes_;
      std::uint32_t threadCount_;
      std::uint32_t maxSteals_{threadCount_ + 1};
      bwos_params params_;
      std::vector<std::thread> threads_;
      std::vector<std::optional<thread_state>> threadStates_;
      numa_policy* numa_;

      struct thread_index_by_numa_node {
        int numa_node;
        int thread_index;

        friend bool operator<(
          const thread_index_by_numa_node& lhs,
          const thread_index_by_numa_node& rhs) noexcept {
          return lhs.numa_node < rhs.numa_node;
        }
      };

      std::vector<thread_index_by_numa_node> threadIndexByNumaNode_;

      std::size_t num_threads(int numa) const noexcept;
      std::size_t num_threads(nodemask constraints) const noexcept;
      std::size_t get_thread_index(int numa, std::size_t index) const noexcept;
      std::size_t random_thread_index_with_constraints(const nodemask& contraints) noexcept;
    };

    inline static_thread_pool_::static_thread_pool_()
      : static_thread_pool_(std::thread::hardware_concurrency()) {
    }

    inline static_thread_pool_::static_thread_pool_(
      std::uint32_t threadCount,
      bwos_params params,
      numa_policy* numa)
      : remotes_(threadCount)
      , threadCount_(threadCount)
      , params_(params)
      , threadStates_(threadCount)
      , numa_{numa} {
      STDEXEC_ASSERT(threadCount > 0);

      for (std::uint32_t index = 0; index < threadCount; ++index) {
        threadStates_[index].emplace(this, index, params, numa);
        threadIndexByNumaNode_.push_back(
          thread_index_by_numa_node{threadStates_[index]->numa_node(), static_cast<int>(index)});
      }
      std::sort(threadIndexByNumaNode_.begin(), threadIndexByNumaNode_.end());
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
          threads_.emplace_back([this, i, numa] { run(i, numa); });
        }
      } catch (...) {
        request_stop();
        join();
        throw;
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

    inline void static_thread_pool_::run(std::uint32_t threadIndex, numa_policy* numa) noexcept {
      numa->bind_to_node(threadStates_[threadIndex]->numa_node());
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

    inline std::size_t static_thread_pool_::num_threads(int numa) const noexcept {
      thread_index_by_numa_node key{numa, 0};
      auto it = std::lower_bound(threadIndexByNumaNode_.begin(), threadIndexByNumaNode_.end(), key);
      if (it == threadIndexByNumaNode_.end()) {
        return 0;
      }
      auto itEnd = std::upper_bound(it, threadIndexByNumaNode_.end(), key);
      return std::distance(it, itEnd);
    }

    inline std::size_t static_thread_pool_::num_threads(nodemask constraints) const noexcept {
      const std::size_t nNodes = threadIndexByNumaNode_.back().numa_node + 1;
      std::size_t nThreads = 0;
      for (std::size_t nodeIndex = 0; nodeIndex < nNodes; ++nodeIndex) {
        if (!constraints[nodeIndex]) {
          continue;
        }
        nThreads += num_threads(nodeIndex);
      }
      return nThreads;
    }

    inline std::size_t
      static_thread_pool_::get_thread_index(int nodeIndex, std::size_t threadIndex) const noexcept {
      thread_index_by_numa_node key{nodeIndex, 0};
      auto it = std::lower_bound(threadIndexByNumaNode_.begin(), threadIndexByNumaNode_.end(), key);
      STDEXEC_ASSERT(it != threadIndexByNumaNode_.end());
      std::advance(it, threadIndex);
      return it->thread_index;
    }

    inline std::size_t static_thread_pool_::random_thread_index_with_constraints(
      const nodemask& constraints) noexcept {
      thread_local std::uint64_t startIndex{std::uint64_t(std::random_device{}())};
      startIndex += 1;
      std::size_t targetIndex = startIndex % threadCount_;
      std::size_t nThreads = num_threads(constraints);
      if (nThreads != 0) {
        for (std::size_t nodeIndex = 0; nodeIndex < numa_->num_nodes(); ++nodeIndex) {
          if (!constraints[nodeIndex]) {
            continue;
          }
          std::size_t nThreads = num_threads(nodeIndex);
          if (targetIndex < nThreads) {
            return get_thread_index(nodeIndex, targetIndex);
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
        std::size_t this_node = threadStates_[idx]->numa_node();
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
      auto& queue = *get_remote_queue();
      for (std::size_t i = 0; i < n_threads; ++i) {
        std::uint32_t index = i % available_parallelism();
        queue.queues_[index].push_front(task + i);
        threadStates_[index]->notify();
      }
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
        std::size_t this_node = threadStates_[idx]->numa_node();
        if (constraints[this_node]) {
          threadStates_[idx]->push_local(std::move(tasks));
          return;
        }
      }
      std::size_t nThreads = available_parallelism();
      for (std::size_t i = 0; i < nThreads; ++i) {
        auto [i0, iEnd] = even_share(tasks_size, i, available_parallelism());
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

    inline static_thread_pool_::thread_state::pop_result
      static_thread_pool_::thread_state::try_remote() {
      pop_result result{nullptr, index_};
      __intrusive_queue<& task_base::next> remotes = pool_->remotes_.pop_all_reversed(index_);
      pending_queue_.append(std::move(remotes));
      if (!pending_queue_.empty()) {
        move_pending_to_local(pending_queue_, local_queue_);
        result.task = local_queue_.pop_back();
      }
      return result;
    }

    inline static_thread_pool_::thread_state::pop_result
      static_thread_pool_::thread_state::try_pop() {
      pop_result result{nullptr, index_};
      result.task = local_queue_.pop_back();
      if (result.task) [[likely]] {
        return result;
      }
      return try_remote();
    }

    inline static_thread_pool_::thread_state::pop_result
      static_thread_pool_::thread_state::try_steal(std::span<workstealing_victim> victims) {
      if (victims.empty()) {
        return {nullptr, index_};
      }
      std::uniform_int_distribution<std::uint32_t> dist(0, victims.size() - 1);
      std::uint32_t victimIndex = dist(rng_);
      auto& v = victims[victimIndex];
      return {v.try_steal(), v.index()};
    }

    inline static_thread_pool_::thread_state::pop_result
      static_thread_pool_::thread_state::try_steal_near() {
      return try_steal(near_victims_);
    }

    inline static_thread_pool_::thread_state::pop_result
      static_thread_pool_::thread_state::try_steal_any() {
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

    inline void static_thread_pool_::thread_state::set_stealing() {
      pool_->numThiefs_.fetch_add(1, std::memory_order_relaxed);
    }

    inline void static_thread_pool_::thread_state::clear_stealing() {
      if (pool_->numThiefs_.fetch_sub(1, std::memory_order_relaxed) == 1) {
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

    inline static_thread_pool_::thread_state::pop_result static_thread_pool_::thread_state::pop() {
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
          cv_.wait(lock);
        }
        lock.unlock();
        state_.store(state::running, std::memory_order_relaxed);
        result = try_pop();
      }
      return result;
    }

    inline bool static_thread_pool_::thread_state::notify() {
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
      friend static_thread_pool_::scheduler::sender;

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
        , rcvr_((Receiver&&) rcvr)
        , threadIndex_{tid}
        , constraints_{constraints} {
        this->__execute = [](task_base* t, const std::uint32_t /* tid */) noexcept {
          auto& op = *static_cast<__t*>(t);
          auto stoken = get_stop_token(get_env(op.rcvr_));
          if constexpr (std::unstoppable_token<decltype(stoken)>) {
            set_value((Receiver&&) op.rcvr_);
          } else if (stoken.stop_requested()) {
            set_stopped((Receiver&&) op.rcvr_);
          } else {
            set_value((Receiver&&) op.rcvr_);
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

      friend void tag_invoke(start_t, __t& op) noexcept {
        op.enqueue_(&op);
      }
    };

    //////////////////////////////////////////////////////////////////////////////////////////////////
    // What follows is the implementation for parallel bulk execution on static_thread_pool_.
    template <class SenderId, std::integral Shape, class Fun>
    struct static_thread_pool_::bulk_sender<SenderId, Shape, Fun>::__t {
      using __id = bulk_sender;
      using sender_concept = sender_t;

      static_thread_pool_& pool_;
      Sender sndr_;
      Shape shape_;
      Fun fun_;

      template <class Sender, class Env>
      using with_error_invoke_t = //
        __if_c<
          __v<__value_types_of_t<
            Sender,
            Env,
            __mbind_front_q<bulk_non_throwing, Fun, Shape>,
            __q<__mand>>>,
          completion_signatures<>,
          __with_exception_ptr>;

      template <class... Tys>
      using set_value_t = completion_signatures< set_value_t(__decay_t<Tys>...)>;

      template <class Self, class Env>
      using __completions_t = //
        stdexec::__try_make_completion_signatures<
          __copy_cvref_t<Self, Sender>,
          Env,
          with_error_invoke_t<__copy_cvref_t<Self, Sender>, Env>,
          __q<set_value_t>>;

      template <class Self, class Receiver>
      using bulk_op_state_t = //
        stdexec::__t<bulk_op_state< __cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun>>;

      template <__decays_to<__t> Self, receiver Receiver>
        requires receiver_of<Receiver, __completions_t<Self, env_of_t<Receiver>>>
      friend bulk_op_state_t<Self, Receiver>                //
        tag_invoke(connect_t, Self&& self, Receiver rcvr) //
        noexcept(__nothrow_constructible_from<
                 bulk_op_state_t<Self, Receiver>,
                 static_thread_pool_&,
                 Shape,
                 Fun,
                 Sender,
                 Receiver>) {
        return bulk_op_state_t<Self, Receiver>{
          self.pool_, self.shape_, self.fun_, ((Self&&) self).sndr_, (Receiver&&) rcvr};
      }

      template <__decays_to<__t> Self, class Env>
      friend auto tag_invoke(get_completion_signatures_t, Self&&, Env&&)
        -> __completions_t<Self, Env> {
        return {};
      }

      friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<const Sender&> {
        return get_env(self.sndr_);
      }
    };

    template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
    struct static_thread_pool_::bulk_shared_state {
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
                sh_state.fun_(i, args...);
              }
            };

            auto completion = [&](auto&... args) {
              set_value((Receiver&&) sh_state.rcvr_, std::move(args)...);
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
                  set_error((Receiver&&) sh_state.rcvr_, std::move(sh_state.exception_));
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
        __value_types_of_t< CvrefSender, env_of_t<Receiver>, __q<__decayed_tuple>, __q<__variant>>;

      variant_t data_;
      static_thread_pool_& pool_;
      Receiver rcvr_;
      Shape shape_;
      Fun fun_;

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

      bulk_shared_state(static_thread_pool_& pool, Receiver rcvr, Shape shape, Fun fun)
        : pool_{pool}
        , rcvr_{(Receiver&&) rcvr}
        , shape_{shape}
        , fun_{fun}
        , thread_with_exception_{num_agents_required()}
        , tasks_{num_agents_required(), {this}} {
      }
    };

    template <class CvrefSenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
    struct static_thread_pool_::bulk_receiver<CvrefSenderId, ReceiverId, Shape, Fun, MayThrow>::__t {
      using __id = bulk_receiver;
      using receiver_concept = receiver_t;

      using shared_state = bulk_shared_state<CvrefSender, Receiver, Shape, Fun, MayThrow>;

      shared_state& shared_state_;

      void enqueue() noexcept {
        shared_state_.pool_.bulk_enqueue(
          shared_state_.tasks_.data(), shared_state_.num_agents_required());
      }

      template <class... As>
      friend void tag_invoke(same_as<set_value_t> auto, __t&& self, As&&... as) noexcept {
        using tuple_t = __decayed_tuple<As...>;

        shared_state& state = self.shared_state_;

        if constexpr (MayThrow) {
          try {
            state.data_.template emplace<tuple_t>((As&&) as...);
          } catch (...) {
            set_error(std::move(state.rcvr_), std::current_exception());
          }
        } else {
          state.data_.template emplace<tuple_t>((As&&) as...);
        }

        if (state.shape_) {
          self.enqueue();
        } else {
          state.apply([&](auto&... args) {
            set_value(std::move(state.rcvr_), std::move(args)...);
          });
        }
      }

      template <__one_of<set_error_t, set_stopped_t> Tag, class... As>
      friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
        shared_state& state = self.shared_state_;
        tag((Receiver&&) state.rcvr_, (As&&) as...);
      }

      friend auto tag_invoke(get_env_t, const __t& self) noexcept -> env_of_t<Receiver> {
        return get_env(self.shared_state_.rcvr_);
      }
    };

    template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
    struct static_thread_pool_::bulk_op_state<CvrefSenderId, ReceiverId, Shape, Fun>::__t {
      using __id = bulk_op_state;

      static constexpr bool may_throw = //
        !__v<__value_types_of_t<
          CvrefSender,
          env_of_t<Receiver>,
          __mbind_front_q<bulk_non_throwing, Fun, Shape>,
          __q<__mand>>>;

      using bulk_rcvr = bulk_receiver_t<CvrefSender, Receiver, Shape, Fun, may_throw>;
      using shared_state = bulk_shared_state<CvrefSender, Receiver, Shape, Fun, may_throw>;
      using inner_op_state = connect_result_t<CvrefSender, bulk_rcvr>;

      shared_state shared_state_;

      inner_op_state inner_op_;

      friend void tag_invoke(start_t, __t& op) noexcept {
        start(op.inner_op_);
      }

      __t(static_thread_pool_& pool, Shape shape, Fun fun, CvrefSender&& sndr, Receiver rcvr)
        : shared_state_(pool, (Receiver&&) rcvr, shape, fun)
        , inner_op_{connect((CvrefSender&&) sndr, bulk_rcvr{shared_state_})} {
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
            set_value(static_cast<ItemReceiver&&>(op->item_receiver_), *op->it_);
          }

          ItemReceiver item_receiver_;
          std::ranges::iterator_t<Range> it_;
          operation_base<Range>* parent_;

          friend void tag_invoke(start_t, __t& op) noexcept {
            std::unique_lock lock{op.parent_->start_mutex_};
            if (!op.parent_->has_started_) {
              op.parent_->tasks_.push_back(static_cast<task_base*>(&op));
              op.parent_->tasks_size_ += 1;
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
          using sender_concept = sender_t;
          using completion_signatures =
            stdexec::completion_signatures<set_value_t(std::ranges::range_reference_t<Range>)>;

          operation_base<Range>* op_;
          std::ranges::iterator_t<Range> it_;

          struct env {
            static_thread_pool_* pool_;

            template < same_as<get_completion_scheduler_t<set_value_t>> Query>
            friend auto tag_invoke(Query, const env& e) noexcept -> static_thread_pool_::scheduler {
              return e.pool_->get_scheduler();
            }
          };

          template <same_as<get_env_t> GetEnv, __decays_to<__t> Self>
          friend auto tag_invoke(GetEnv, Self&& self) noexcept -> env {
            return {self.op_->pool_};
          }

          template <__decays_to<__t> Self, receiver ItemReceiver>
            requires receiver_of<ItemReceiver, completion_signatures>
          friend auto tag_invoke(connect_t, Self&& self, ItemReceiver rcvr) noexcept
            -> stdexec::__t<item_operation<Range, stdexec::__id<ItemReceiver>>> {
            return {static_cast<ItemReceiver&&>(rcvr), self.it_, self.op_};
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

          template <same_as<set_value_t> SetValue, same_as<__t> Self>
          friend void tag_invoke(SetValue, Self&& self) noexcept {
            std::size_t countdown = self.op_->countdown_.fetch_sub(1, std::memory_order_relaxed);
            if (countdown == 1) {
              set_value((Receiver&&) self.op_->rcvr_);
            }
          }

          template <same_as<set_stopped_t> SetStopped, same_as<__t> Self>
          friend void tag_invoke(SetStopped, Self&& self) noexcept {
            std::size_t countdown = self.op_->countdown_.fetch_sub(1, std::memory_order_relaxed);
            if (countdown == 1) {
              set_value((Receiver&&) self.op_->rcvr_);
            }
          }

          template <same_as<get_env_t> GetEnv, __decays_to<__t> Self>
          friend auto tag_invoke(GetEnv, Self&& self) noexcept -> env_of_t<Receiver> {
            return get_env(self.op_->rcvr_);
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

          using ItemAllocator = std::allocator_traits<Allocator>::template rebind_alloc<
            __manual_lifetime<ItemOperation>>;

          std::vector<__manual_lifetime<ItemOperation>, ItemAllocator> items_;

          template <same_as<__t> Self>
          friend void tag_invoke(start_t, Self& op) noexcept {
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
                  return connect(
                    set_next(op.rcvr_, ItemSender{&op, it + i}), NextReceiver{&op});
                });
                start(op.items_[i].__get());
              }
              std::unique_lock lock{op.start_mutex_};
              op.pool_.bulk_enqueue(remote_queue, std::move(op.tasks_), op.tasks_size_);
              lock.unlock();
              i0 += chunkSize;
            }
            for (std::size_t i = i0; i < size; ++i) {
              op.items_[i].__construct_with([&] {
                return connect(set_next(op.rcvr_, ItemSender{&op, it + i}), NextReceiver{&op});
              });
              start(op.items_[i].__get());
            }
            std::unique_lock lock{op.start_mutex_};
            op.has_started_ = true;
            op.pool_.bulk_enqueue(remote_queue, std::move(op.tasks_), op.tasks_size_);
          }

         public:
          using __id = operation;

          __t(Range range, static_thread_pool_& pool, Receiver rcvr)
            : operation_base_with_receiver<
              Range,
              Receiver>{std::move(range), pool, static_cast<Receiver&&>(rcvr)}
            , items_(
                std::ranges::size(this->range_),
                ItemAllocator(get_allocator(this->rcvr_))) {
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
        static_thread_pool_* pool_;

       public:
        using __id = sequence;

        using sender_concept = sequence_sender_t;

        using completion_signatures =
          stdexec::completion_signatures< set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;

        using item_types = exec::item_types<stdexec::__t<item_sender<Range>>>;

        __t(Range range, static_thread_pool_& pool)
          : range_(static_cast<Range&&>(range))
          , pool_(&pool) {
        }

       private:
        template <__decays_to<__t> Self, exec::sequence_receiver_of<item_types> Receiver>
        friend auto tag_invoke(exec::subscribe_t, Self&& self, Receiver rcvr) noexcept
          -> stdexec::__t<operation<Range, stdexec::__id<Receiver>>> {
          return {static_cast<Range&&>(self.range_), *self.pool_, static_cast<Receiver&&>(rcvr)};
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
      numa_policy* numa = get_numa_policy())
      : _pool_::static_thread_pool_(threadCount, params, numa) {
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
      stdexec::__t<schedule_all_::sequence<__decay_t<Range>>>
        operator()(static_thread_pool& pool, Range&& range) const {
        return {static_cast<Range&&>(range), pool};
      }
    };
  } // namespace _pool_

  inline constexpr _pool_::schedule_all_t schedule_all{};
#endif

} // namespace exec
