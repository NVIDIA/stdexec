/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates.
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "../stdexec/__detail/__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)
import std;
import stdexec;
#else
#  include "../stdexec/__detail/__domain.hpp"
#  include "../stdexec/__detail/__execution_fwd.hpp"
#  include "../stdexec/__detail/__execution_legacy.hpp"  // IWYU pragma: keep
#  include "../stdexec/__detail/__get_completion_signatures.hpp"
#  include "../stdexec/__detail/__intrusive_queue.hpp"
#  include "../stdexec/__detail/__meta.hpp"
#  include "../stdexec/__detail/__optional.hpp"
#  include "../stdexec/__detail/__receivers.hpp"
#  include "../stdexec/__detail/__tuple.hpp"
#  include "../stdexec/__detail/__type_traits.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <compare>
#    include <condition_variable>
#    include <cstdint>
#    include <limits>
#    include <mutex>
#    include <random>
#    include <span>
#    include <thread>
#    include <vector>
#  endif
#endif

#include "detail/atomic_intrusive_queue.hpp"
#include "detail/bwos_lifo_queue.hpp"
#include "detail/numa.hpp"
#include "detail/xorshift.hpp"

#include "sender_for.hpp"
#include "sequence/iterate.hpp"
#include "thread_pool_base.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(is_constant_evaluated_in_nonconstexpr_context)

namespace experimental::execution
{
  struct CANNOT_DISPATCH_THE_ITERATE_ALGORITHM_TO_THE_STATIC_THREAD_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_STATIC_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STATIC_THREAD_POOL_SCHEDULER_BEFORE_THE_ITERATE_ALGORITHM;

  struct static_thread_pool;

  namespace _pool
  {
    using namespace STDEXEC;

    struct _remote_queue
    {
      explicit _remote_queue(std::size_t nthreads) noexcept
        : queues_(nthreads)
      {}

      explicit _remote_queue(_remote_queue* next, std::size_t nthreads) noexcept
        : next_(next)
        , queues_(nthreads)
      {}

      _remote_queue*                                            next_{};
      std::vector<__atomic_intrusive_queue<&_task_base::next_>> queues_{};
      std::thread::id                                           id_{std::this_thread::get_id()};
      // This marks whether the submitter is a thread in the pool or not.
      std::size_t index_{(std::numeric_limits<std::size_t>::max)()};
    };

    enum class _remote_poll_mode
    {
      _speculative,
      _before_sleep
    };

    struct _remote_queue_list
    {
     private:
      __std::atomic<_remote_queue*> head_;
      _remote_queue*                tail_;
      std::size_t                   nthreads_;
      _remote_queue                 this_remotes_;

     public:
      explicit _remote_queue_list(std::size_t nthreads) noexcept
        : head_{&this_remotes_}
        , tail_{&this_remotes_}
        , nthreads_(nthreads)
        , this_remotes_(nthreads)
      {}

      ~_remote_queue_list() noexcept
      {
        _remote_queue* head = head_.load(__std::memory_order_acquire);
        while (head != tail_)
        {
          _remote_queue* tmp = std::exchange(head, head->next_);
          delete tmp;
        }
      }

      auto pop_all_reversed(std::size_t tid, _remote_poll_mode mode) noexcept
        -> __intrusive_queue<&_task_base::next_>
      {
        _remote_queue*                        head = head_.load(__std::memory_order_acquire);
        __intrusive_queue<&_task_base::next_> tasks{};
        while (head != nullptr)
        {
          auto& queue = head->queues_[tid];
          if (mode == _remote_poll_mode::_before_sleep || !queue.empty())
          {
            tasks.append(queue.pop_all_reversed());
          }
          head = head->next_;
        }
        return tasks;
      }

      auto get() -> _remote_queue*
      {
        _remote_queue* head  = head_.load(__std::memory_order_acquire);
        _remote_queue* queue = head;
        while (queue != tail_)
        {
          if (queue->id_ == std::this_thread::get_id())
          {
            return queue;
          }
          queue = queue->next_;
        }
        auto* new_head = new _remote_queue{head, nthreads_};
        while (!head_.compare_exchange_weak(head, new_head, __std::memory_order_acq_rel))
        {
          new_head->next_ = head;
        }
        return new_head;
      }
    };

    class _static_thread_pool : public thread_pool_base<static_thread_pool>
    {
      friend struct static_thread_pool;

      template <class Receiver>
      struct _opstate;

      struct _transform_iterate
      {
        template <class Range>
        auto operator()(exec::iterate_t, Range&& range)
          -> _schedule_all::_sequence<__decay_t<Range>, static_thread_pool>
        {
          return {static_cast<Range&&>(range), pool_};
        }

        static_thread_pool& pool_;
      };

      static auto _hardware_concurrency() noexcept -> unsigned int
      {
        unsigned int const n = std::thread::hardware_concurrency();
        return n == 0 ? 1 : n;
      }

     public:
      // The bulk algorithms are handled by the generic pool domain; only `iterate`
      // is specific to this pool.
      struct domain : _pool::_domain<static_thread_pool>
      {
        // Bring in the base class's transform_sender for bulk_[un]chunked
        using _pool::_domain<static_thread_pool>::transform_sender;

        template <sender_for<exec::iterate_t> Sender, class Env>
        constexpr auto
        transform_sender(STDEXEC::set_value_t, Sender&& sndr, Env const & env) const noexcept
        {
          if constexpr (__completes_on<Sender, scheduler, Env>)
          {
            auto  sched = STDEXEC::get_completion_scheduler<set_value_t>(get_env(sndr), env);
            auto& pool  = thread_pool_access::get_pool(sched);
            return __apply(_transform_iterate{pool}, static_cast<Sender&&>(sndr));
          }
          else
          {
            return __not_a_sender<
              _WHAT_(CANNOT_DISPATCH_THE_ITERATE_ALGORITHM_TO_THE_STATIC_THREAD_POOL_SCHEDULER),
              _WHY_(BECAUSE_THERE_IS_NO_STATIC_THREAD_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
              _WHERE_(_IN_ALGORITHM_, exec::iterate_t),
              _TO_FIX_THIS_ERROR_(
                ADD_A_CONTINUES_ON_TRANSITION_TO_THE_STATIC_THREAD_POOL_SCHEDULER_BEFORE_THE_ITERATE_ALGORITHM),
              _WITH_SENDER_<Sender>,
              _WITH_ENVIRONMENT_(Env)>();
          }
        }
      };

      _static_thread_pool();
      explicit _static_thread_pool(std::uint32_t threadCount,
                                   bwos_params   params = {},
                                   numa_policy   numa   = get_numa_policy());
      ~_static_thread_pool();

     public:
      struct scheduler
      {
        using scheduler_concept = scheduler_tag;

       private:
        template <class Receiver>
        friend struct _opstate;

        friend thread_pool_access;  // needs access to pool_

        class _sender
        {
          struct attrs
          {
            template <class CPO>
            auto query(get_completion_scheduler_t<CPO>, __ignore = {}) const noexcept -> scheduler
            {
              return scheduler{pool_, *queue_};
            }

            template <class CPO>
            auto query(get_completion_domain_t<CPO>, __ignore = {}) const noexcept -> domain
            {
              return {};
            }

            _static_thread_pool& pool_;
            _remote_queue*       queue_;
          };

         public:
          using sender_concept = sender_tag;

          template <class _Self, class _Env>
          static consteval auto get_completion_signatures() noexcept
          {
            if constexpr (unstoppable_token<stop_token_of_t<_Env>>)
            {
              return STDEXEC::completion_signatures<set_value_t()>();
            }
            else
            {
              return STDEXEC::completion_signatures<set_value_t(), set_stopped_t()>();
            }
          }

          [[nodiscard]]
          auto get_env() const noexcept -> attrs
          {
            return attrs{.pool_ = pool_, .queue_ = queue_};
          }

          template <receiver Receiver>
          auto connect(Receiver rcvr) const -> _opstate<Receiver>
          {
            return _opstate<Receiver>{pool_, queue_, std::move(rcvr), threadIndex_, constraints_};
          }

         private:
          friend struct scheduler;

          explicit _sender(_static_thread_pool& pool,
                           _remote_queue*       queue,
                           std::size_t          threadIndex,
                           nodemask const &     constraints) noexcept
            : pool_(pool)
            , queue_(queue)
            , threadIndex_(threadIndex)
            , constraints_(constraints)
          {}

          _static_thread_pool& pool_;
          _remote_queue*       queue_;
          std::size_t          threadIndex_{(std::numeric_limits<std::size_t>::max)()};
          nodemask             constraints_{};
        };

        friend class _static_thread_pool;

        explicit scheduler(_static_thread_pool& pool,
                           nodemask const *     mask = &nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{pool.get_remote_queue()}
          , nodemask_{mask}
        {}

        explicit scheduler(_static_thread_pool& pool,
                           _remote_queue&       queue,
                           nodemask const *     mask = &nodemask::any()) noexcept
          : pool_(&pool)
          , queue_{&queue}
          , nodemask_{mask}
        {}

        explicit scheduler(_static_thread_pool& pool,
                           _remote_queue&       queue,
                           std::size_t          threadIndex) noexcept
          : pool_(&pool)
          , queue_{&queue}
          , thread_idx_{threadIndex}
        {}

        _static_thread_pool* pool_;
        _remote_queue*       queue_;
        nodemask const *     nodemask_ = &nodemask::any();
        std::size_t          thread_idx_{(std::numeric_limits<std::size_t>::max)()};

       public:
        auto operator==(scheduler const &) const -> bool = default;

        [[nodiscard]]
        auto schedule() const noexcept -> _sender
        {
          return _sender{*pool_, queue_, thread_idx_, *nodemask_};
        }

        [[nodiscard]]
        auto query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee
        {
          return forward_progress_guarantee::parallel;
        }

        [[nodiscard]]
        auto query(get_completion_domain_t<set_value_t>, __ignore = {}) const noexcept -> domain
        {
          return {};
        }

        [[nodiscard]]
        auto
        query(get_completion_scheduler_t<set_value_t>, __ignore = {}) const noexcept -> scheduler
        {
          return scheduler{*this};
        }
      };

      // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
      auto get_scheduler() noexcept -> scheduler
      {
        return scheduler{*this};
      }

      auto get_scheduler_on_thread(std::size_t threadIndex) noexcept -> scheduler
      {
        return scheduler{*this, *get_remote_queue(), threadIndex};
      }

      // The caller must ensure that the constraints object is valid for the
      // lifetime of the scheduler.
      auto get_constrained_scheduler(nodemask const * constraints) noexcept -> scheduler
      {
        return scheduler{*this, *get_remote_queue(), constraints};
      }

      auto get_remote_queue() noexcept -> _remote_queue*
      {
        _remote_queue* queue = remotes_.get();
        std::size_t    index = 0;
        for (std::thread& t: threads_)
        {
          if (t.get_id() == queue->id_)
          {
            queue->index_ = index;
            break;
          }

          ++index;
        }
        return queue;
      }

      void request_stop() noexcept;

      // NOLINTNEXTLINE(bugprone-derived-method-shadowing-base-method)
      [[nodiscard]]
      auto available_parallelism() const noexcept -> std::uint32_t
      {
        return thread_count_;
      }

      [[nodiscard]]
      auto params() const noexcept -> bwos_params
      {
        return params_;
      }

      void enqueue(_task_base* task, nodemask const & contraints = nodemask::any()) noexcept;
      void enqueue(_remote_queue&   queue,
                   _task_base*      task,
                   nodemask const & contraints = nodemask::any()) noexcept;
      void enqueue(_remote_queue& queue, _task_base* task, std::size_t thread_index) noexcept;

      //! Enqueue a contiguous span of tasks across task queues. Note: We use
      //! the concrete `Task` because we enqueue tasks `task + 0`, `task + 1`,
      //! etc. so std::span<_task_base> wouldn't be correct. This is
      //! O(n_threads) on the calling thread.
      template <std::derived_from<_task_base> Task>
      void bulk_enqueue(std::span<Task> tasks) noexcept;
      void bulk_enqueue(_remote_queue&                        queue,
                        __intrusive_queue<&_task_base::next_> tasks,
                        std::size_t                           tasks_size,
                        nodemask const & constraints = nodemask::any()) noexcept;

     private:
      class _workstealing_victim
      {
       public:
        explicit _workstealing_victim(
          bwos::lifo_queue<_task_base*, numa_allocator<_task_base*>>* queue,
          std::uint32_t                                               index,
          int                                                         numa_node) noexcept
          : queue_(queue)
          , index_(index)
          , numa_node_(numa_node)
        {}

        auto try_steal() noexcept -> _task_base*
        {
          return queue_->steal_front();
        }

        [[nodiscard]]
        auto index() const noexcept -> std::uint32_t
        {
          return index_;
        }

        [[nodiscard]]
        auto numa_node() const noexcept -> int
        {
          return numa_node_;
        }

       private:
        bwos::lifo_queue<_task_base*, numa_allocator<_task_base*>>* queue_;
        std::uint32_t                                               index_;
        int                                                         numa_node_;
      };

      struct _thread_state_base
      {
        explicit _thread_state_base(std::uint32_t index, numa_policy const & numa) noexcept
          : index_(index)
          , numa_node_(numa.thread_index_to_node(index))
        {}

        std::uint32_t index_;
        int           numa_node_;
      };

      class _thread_state : private _thread_state_base
      {
       public:
        struct pop_result
        {
          _task_base*   task;
          std::uint32_t queue_index;
        };

        explicit _thread_state(_static_thread_pool* pool,
                               std::uint32_t        index,
                               bwos_params          params,
                               numa_policy const &  numa) noexcept
          : _thread_state_base(index, numa)
          , local_queue_(params.numBlocks,
                         params.blockSize,
                         numa_allocator<_task_base*>(this->numa_node_))
          , state_(state::running)
          , pool_(pool)
        {
          std::random_device rd;
          rng_.seed(rd);
        }

        auto pop() -> pop_result;
        void push_local(_task_base* task);
        void push_local(__intrusive_queue<&_task_base::next_>&& tasks);

        auto notify() -> bool;
        void request_stop();

        void victims(std::vector<_workstealing_victim> const & victims)
        {
          for (_workstealing_victim v: victims)
          {
            if (v.index() == index_)
            {
              // skip self
              continue;
            }
            if (v.numa_node() == numa_node_)
            {
              near_victims_.push_back(v);
            }
            all_victims_.push_back(v);
          }
        }

        [[nodiscard]]
        auto index() const noexcept -> std::uint32_t
        {
          return index_;
        }

        [[nodiscard]]
        auto numa_node() const noexcept -> int
        {
          return numa_node_;
        }

        auto as_victim() noexcept -> _workstealing_victim
        {
          return _workstealing_victim{&local_queue_, index_, numa_node_};
        }

       private:
        enum state
        {
          running,
          stealing,
          sleeping,
          notified
        };

        auto try_pop() -> pop_result;
        auto try_remote(_remote_poll_mode mode) -> pop_result;
        auto try_steal(std::span<_workstealing_victim> victims) -> pop_result;
        auto try_steal_near() -> pop_result;
        auto try_steal_any() -> pop_result;

        void notify_one_sleeping();
        void set_stealing();
        void clear_stealing();
        void set_sleeping();
        void clear_sleeping();

        bwos::lifo_queue<_task_base*, numa_allocator<_task_base*>> local_queue_;
        __intrusive_queue<&_task_base::next_>                      pending_queue_{};
        std::mutex                                                 mut_{};
        std::condition_variable                                    cv_{};
        bool                                                       stop_requested_{false};
        std::vector<_workstealing_victim>                          near_victims_{};
        std::vector<_workstealing_victim>                          all_victims_{};
        __std::atomic<state>                                       state_;
        _static_thread_pool*                                       pool_;
        xorshift                                                   rng_{};
      };

      void run(std::uint32_t index) noexcept;
      void join() noexcept;

      alignas(64) __std::atomic<std::uint32_t> num_active_{};
      alignas(64) _remote_queue_list remotes_;
      std::uint32_t                          thread_count_;
      std::uint32_t                          max_steals_{thread_count_ + 1};
      bwos_params                            params_;
      std::vector<std::thread>               threads_;
      std::vector<__optional<_thread_state>> thread_states_;
      numa_policy                            numa_;

      struct _thread_index_by_numa_node
      {
        int         numa_node;
        std::size_t thread_index;

        friend auto operator==(_thread_index_by_numa_node const & lhs,
                               _thread_index_by_numa_node const & rhs) noexcept -> bool
        {
          return lhs.numa_node == rhs.numa_node;
        }

        friend auto
        operator<=>(_thread_index_by_numa_node const & lhs,
                    _thread_index_by_numa_node const & rhs) noexcept -> std::strong_ordering
        {
          return lhs.numa_node <=> rhs.numa_node;
        }
      };

      std::vector<_thread_index_by_numa_node> thread_index_by_numa_node_;

      [[nodiscard]]
      auto num_threads(int numa) const noexcept -> std::size_t;
      [[nodiscard]]
      auto num_threads(nodemask constraints) const noexcept -> std::size_t;
      [[nodiscard]]
      auto get_thread_index(int numa, std::size_t index) const noexcept -> std::size_t;
      auto
      random_thread_index_with_constraints(nodemask const & contraints) noexcept -> std::size_t;
    };

    inline _static_thread_pool::_static_thread_pool()
      : _static_thread_pool(_hardware_concurrency())
    {}

    inline _static_thread_pool::_static_thread_pool(std::uint32_t thread_count,
                                                    bwos_params   params,
                                                    numa_policy   numa)
      : remotes_(thread_count)
      , thread_count_(thread_count)
      , params_(params)
      , thread_states_(thread_count)
      , numa_(std::move(numa))
    {
      STDEXEC_ASSERT(thread_count > 0);

      for (std::uint32_t index = 0; index < thread_count; ++index)
      {
        thread_states_[index].emplace(this, index, params, numa_);
        thread_index_by_numa_node_.push_back(
          _thread_index_by_numa_node{.numa_node    = thread_states_[index]->numa_node(),
                                     .thread_index = index});
      }

      std::ranges::sort(thread_index_by_numa_node_);
      std::vector<_workstealing_victim> victims{};
      for (auto& state: thread_states_)
      {
        victims.emplace_back(state->as_victim());
      }
      for (auto& state: thread_states_)
      {
        state->victims(victims);
      }
      threads_.reserve(thread_count);

      STDEXEC_TRY
      {
        num_active_.store(thread_count << 16u, __std::memory_order_relaxed);
        for (std::uint32_t i = 0; i < thread_count; ++i)
        {
          threads_.emplace_back([this, i] { run(i); });
        }
      }
      STDEXEC_CATCH_ALL
      {
        request_stop();
        join();
        STDEXEC_THROW();
      }
    }

    inline _static_thread_pool::~_static_thread_pool()
    {
      request_stop();
      join();
    }

    inline void _static_thread_pool::request_stop() noexcept
    {
      for (auto& state: thread_states_)
      {
        state->request_stop();
      }
    }

    inline void _static_thread_pool::run(std::uint32_t thread_index) noexcept
    {
      STDEXEC_ASSERT(thread_index < thread_count_);
      // NOLINTNEXTLINE(bugprone-unused-return-value)
      numa_.bind_to_node(thread_states_[thread_index]->numa_node());
      while (true)
      {
        // Make a blocking call to de-queue a task if we don't already have one.
        auto [task, queue_index] = thread_states_[thread_index]->pop();
        if (!task)
        {
          return;  // pop() only returns null when request_stop() was called.
        }
        task->execute_(task, queue_index);
      }
    }

    inline void _static_thread_pool::join() noexcept
    {
      for (auto& t: threads_)
      {
        t.join();
      }
      threads_.clear();
    }

    inline void
    _static_thread_pool::enqueue(_task_base* task, nodemask const & constraints) noexcept
    {
      this->enqueue(*get_remote_queue(), task, constraints);
    }

    inline auto _static_thread_pool::num_threads(int numa) const noexcept -> std::size_t
    {
      _thread_index_by_numa_node key{.numa_node = numa, .thread_index = 0};
      auto                       it = std::ranges::lower_bound(thread_index_by_numa_node_, key);
      if (it == thread_index_by_numa_node_.end())
      {
        return 0;
      }
      auto it_end = std::ranges::upper_bound(it, thread_index_by_numa_node_.end(), key);
      return static_cast<std::size_t>(std::distance(it, it_end));
    }

    inline auto _static_thread_pool::num_threads(nodemask constraints) const noexcept -> std::size_t
    {
      std::size_t const n_nodes = static_cast<unsigned>(thread_index_by_numa_node_.back().numa_node
                                                        + 1);
      std::size_t       n_threads = 0;
      for (std::size_t node_index = 0; node_index < n_nodes; ++node_index)
      {
        if (!constraints[node_index])
        {
          continue;
        }

        n_threads += num_threads(static_cast<int>(node_index));
      }
      return n_threads;
    }

    inline auto
    _static_thread_pool::get_thread_index(int node_index, std::size_t thread_index) const noexcept
      -> std::size_t
    {
      _thread_index_by_numa_node key{.numa_node = node_index, .thread_index = 0};
      auto                       it = std::ranges::lower_bound(thread_index_by_numa_node_, key);
      STDEXEC_ASSERT(it != thread_index_by_numa_node_.end());
      std::advance(it, thread_index);
      return it->thread_index;
    }

    inline auto
    _static_thread_pool::random_thread_index_with_constraints(nodemask const & constraints) noexcept
      -> std::size_t
    {
      thread_local std::uint64_t start_index{std::uint64_t(std::random_device{}())};
      start_index += 1;
      std::size_t const n_threads    = num_threads(constraints);
      std::size_t       target_index = start_index % (n_threads == 0 ? thread_count_ : n_threads);
      if (n_threads != 0)
      {
        for (std::size_t node_index = 0; node_index < numa_.num_nodes(); ++node_index)
        {
          if (!constraints[node_index])
          {
            continue;
          }
          std::size_t node_threads = num_threads(static_cast<int>(node_index));
          if (target_index < node_threads)
          {
            return get_thread_index(static_cast<int>(node_index), target_index);
          }
          target_index -= node_threads;
        }
      }
      return target_index;
    }

    inline void _static_thread_pool::enqueue(_remote_queue&   queue,
                                             _task_base*      task,
                                             nodemask const & constraints) noexcept
    {
      _remote_queue* correct_queue = std::this_thread::get_id() == queue.id_ ? &queue
                                                                             : get_remote_queue();
      std::size_t    idx           = correct_queue->index_;
      if (idx < thread_states_.size())
      {
        auto this_node = static_cast<std::size_t>(thread_states_[idx]->numa_node());
        if (constraints[this_node])
        {
          thread_states_[idx]->push_local(task);
          return;
        }
      }

      std::size_t const thread_index = random_thread_index_with_constraints(constraints);
      if (queue.queues_[thread_index].push_front(task))
      {
        thread_states_[thread_index]->notify();
      }
    }

    inline void _static_thread_pool::enqueue(_remote_queue& queue,
                                             _task_base*    task,
                                             std::size_t    thread_index) noexcept
    {
      thread_index %= thread_count_;
      if (queue.queues_[thread_index].push_front(task))
      {
        thread_states_[thread_index]->notify();
      }
    }

    template <std::derived_from<_task_base> Task>
    void _static_thread_pool::bulk_enqueue(std::span<Task> tasks) noexcept
    {
      auto& queue = *this->get_remote_queue();
      for (std::uint32_t i = 0; i < tasks.size(); ++i)
      {
        std::uint32_t index = i % this->available_parallelism();
        if (queue.queues_[index].push_front(&tasks[i]))
        {
          thread_states_[index]->notify();
        }
      }
      // At this point the calling thread can exit and the pool will take over.
      // Ultimately, the last completing thread passes the result forward.
      // See `if (is_last_thread)` above.
    }

    inline void _static_thread_pool::bulk_enqueue(_remote_queue&                        queue,
                                                  __intrusive_queue<&_task_base::next_> tasks,
                                                  std::size_t                           tasks_size,
                                                  nodemask const & constraints) noexcept
    {
      _remote_queue* const correct_queue = std::this_thread::get_id() == queue.id_
                                           ? &queue
                                           : get_remote_queue();
      std::size_t const    idx           = correct_queue->index_;
      if (idx < thread_states_.size())
      {
        auto this_node = static_cast<std::size_t>(thread_states_[idx]->numa_node());
        if (constraints[this_node])
        {
          thread_states_[idx]->push_local(std::move(tasks));
          return;
        }
      }

      std::uint32_t const total_threads = available_parallelism();
      for (std::uint32_t i = 0; i < total_threads; ++i)
      {
        auto [begin, end] = _pool::_even_share(tasks_size, i, total_threads);
        if (begin == end)
        {
          continue;
        }
        __intrusive_queue<&_task_base::next_> tmp{};
        for (auto j = begin; j < end; ++j)
        {
          tmp.push_back(tasks.pop_front());
        }
        correct_queue->queues_[i].prepend(std::move(tmp));
        thread_states_[i]->notify();
      }
    }

    inline void
    move_pending_to_local(__intrusive_queue<&_task_base::next_>&                      pending_queue,
                          bwos::lifo_queue<_task_base*, numa_allocator<_task_base*>>& local_queue)
    {
      auto const last = local_queue.push_back(pending_queue.begin(), pending_queue.end());
      __intrusive_queue<&_task_base::next_> tmp{};
      tmp.splice(tmp.begin(), pending_queue, pending_queue.begin(), last);
      tmp.clear();
    }

    inline auto _static_thread_pool::_thread_state::try_remote(_remote_poll_mode mode)
      -> _static_thread_pool::_thread_state::pop_result
    {
      pop_result                            result{.task = nullptr, .queue_index = index_};
      __intrusive_queue<&_task_base::next_> remotes = pool_->remotes_.pop_all_reversed(index_,
                                                                                       mode);
      pending_queue_.append(std::move(remotes));
      if (!pending_queue_.empty())
      {
        move_pending_to_local(pending_queue_, local_queue_);
        result.task = local_queue_.pop_back();
      }

      return result;
    }

    inline auto
    _static_thread_pool::_thread_state::try_pop() -> _static_thread_pool::_thread_state::pop_result
    {
      pop_result result{.task = nullptr, .queue_index = index_};
      result.task = local_queue_.pop_back();
      if (result.task) [[likely]]
      {
        return result;
      }
      return try_remote(_remote_poll_mode::_speculative);
    }

    inline auto
    _static_thread_pool::_thread_state::try_steal(std::span<_workstealing_victim> victims)
      -> _static_thread_pool::_thread_state::pop_result
    {
      if (victims.empty())
      {
        return {.task = nullptr, .queue_index = index_};
      }
      std::uniform_int_distribution<std::uint32_t> dist(0,
                                                        static_cast<std::uint32_t>(victims.size()
                                                                                   - 1));
      std::uint32_t                                victim_index = dist(rng_);
      auto&                                        v            = victims[victim_index];
      return {.task = v.try_steal(), .queue_index = v.index()};
    }

    inline auto _static_thread_pool::_thread_state::try_steal_near()
      -> _static_thread_pool::_thread_state::pop_result
    {
      return try_steal(near_victims_);
    }

    inline auto _static_thread_pool::_thread_state::try_steal_any()
      -> _static_thread_pool::_thread_state::pop_result
    {
      return try_steal(all_victims_);
    }

    inline void _static_thread_pool::_thread_state::push_local(_task_base* task)
    {
      if (!local_queue_.push_back(task))
      {
        pending_queue_.push_back(task);
      }
    }

    inline void
    _static_thread_pool::_thread_state::push_local(__intrusive_queue<&_task_base::next_>&& tasks)
    {
      pending_queue_.prepend(std::move(tasks));
    }

    inline void _static_thread_pool::_thread_state::set_sleeping()
    {
      pool_->num_active_.fetch_sub(1u << 16u, __std::memory_order_relaxed);
    }

    // wakeup a worker thread and maintain the invariant that we always one active thief as long as a potential victim is awake
    inline void _static_thread_pool::_thread_state::clear_sleeping()
    {
      std::uint32_t const num_active = pool_->num_active_.fetch_add(1u << 16u,
                                                                    __std::memory_order_relaxed);
      if (num_active == 0)
      {
        notify_one_sleeping();
      }
    }

    inline void _static_thread_pool::_thread_state::set_stealing()
    {
      std::uint32_t const diff = 1u - (1u << 16u);
      pool_->num_active_.fetch_add(diff, __std::memory_order_relaxed);
    }

    // put a thief to sleep but maintain the invariant that we always have one active thief as long as a potential victim is awake
    inline void _static_thread_pool::_thread_state::clear_stealing()
    {
      constexpr std::uint32_t diff        = 1 - (1u << 16u);
      std::uint32_t const     num_active  = pool_->num_active_.fetch_sub(diff,
                                                                    __std::memory_order_relaxed);
      std::uint32_t const     num_victims = num_active >> 16u;
      std::uint32_t const     num_thiefs  = num_active & 0xffffu;
      if (num_thiefs == 1 && num_victims != 0)
      {
        notify_one_sleeping();
      }
    }

    inline void _static_thread_pool::_thread_state::notify_one_sleeping()
    {
      std::uniform_int_distribution<std::uint32_t> dist(0, pool_->thread_count_ - 1);
      std::uint32_t                                start_index = dist(rng_);
      for (std::uint32_t i = 0; i < pool_->thread_count_; ++i)
      {
        std::uint32_t index = (start_index + i) % pool_->thread_count_;
        if (index == index_)
        {
          continue;
        }
        if (pool_->thread_states_[index]->notify())
        {
          return;
        }
      }
    }

    inline auto _static_thread_pool::_thread_state::pop()  //
      -> _static_thread_pool::_thread_state::pop_result
    {
      pop_result result = try_pop();
      while (!result.task)
      {
        set_stealing();
        for (std::size_t i = 0; i < pool_->max_steals_; ++i)
        {
          result = try_steal_near();
          if (result.task)
          {
            clear_stealing();
            return result;
          }
        }

        for (std::size_t i = 0; i < pool_->max_steals_; ++i)
        {
          result = try_steal_any();
          if (result.task)
          {
            clear_stealing();
            return result;
          }
        }
        std::this_thread::yield();
        clear_stealing();

        std::unique_lock lock{mut_};
        if (stop_requested_)
        {
          return result;
        }
        state expected = state::running;
        if (state_.compare_exchange_weak(expected,
                                         state::sleeping,
                                         __std::memory_order_relaxed,
                                         __std::memory_order_relaxed))
        {
          // The relaxed empty probe is safe during normal polling, but the
          // running-to-sleeping boundary must perform the CAS dequeue so work
          // published before the transition cannot be missed.
          result = try_remote(_remote_poll_mode::_before_sleep);
          if (result.task)
          {
            state expected_sleeping = state::sleeping;
            state_.compare_exchange_strong(expected_sleeping,
                                           state::running,
                                           __std::memory_order_relaxed,
                                           __std::memory_order_relaxed);
            return result;
          }
          set_sleeping();
          cv_.wait(lock);
          lock.unlock();
          clear_sleeping();
        }
        if (lock.owns_lock())
        {
          lock.unlock();
        }
        state_.exchange(state::running, __std::memory_order_acquire);
        result = try_pop();
      }
      return result;
    }

    inline auto _static_thread_pool::_thread_state::notify() -> bool
    {
      if (state_.exchange(state::notified, __std::memory_order_release) == state::sleeping)
      {
        {
          std::lock_guard lock{mut_};
        }
        cv_.notify_one();
        return true;
      }
      return false;
    }

    inline void _static_thread_pool::_thread_state::request_stop()
    {
      {
        std::lock_guard lock{mut_};
        stop_requested_ = true;
      }
      cv_.notify_one();
    }

    template <class Receiver>
    struct _static_thread_pool::_opstate : _task_base
    {
      void start() & noexcept
      {
        enqueue_(this);
      }

     private:
      friend class scheduler::_sender;

      explicit _opstate(_static_thread_pool& pool,
                        _remote_queue*       queue,
                        Receiver             rcvr,
                        std::size_t          tid,
                        nodemask const &     constraints) noexcept
        : _task_base{&_opstate::do_execute_}
        , pool_(pool)
        , queue_(queue)
        , rcvr_(static_cast<Receiver&&>(rcvr))
        , thread_index_{tid}
        , constraints_{constraints}
      {}

      void enqueue_(_task_base* op) const
      {
        if (thread_index_ < pool_.available_parallelism())
        {
          pool_.enqueue(*queue_, op, thread_index_);
        }
        else
        {
          pool_.enqueue(*queue_, op, constraints_);
        }
      }

      static void do_execute_(_task_base* t, [[maybe_unused]] std::uint32_t const tid) noexcept
      {
        auto& op     = *static_cast<_opstate*>(t);
        auto  stoken = get_stop_token(get_env(op.rcvr_));

        if constexpr (STDEXEC::unstoppable_token<decltype(stoken)>)
        {  // NOLINT(bugprone-branch-clone)
          STDEXEC::set_value(static_cast<Receiver&&>(op.rcvr_));
        }
        else if (stoken.stop_requested())
        {
          STDEXEC::set_stopped(static_cast<Receiver&&>(op.rcvr_));
        }
        else
        {
          STDEXEC::set_value(static_cast<Receiver&&>(op.rcvr_));
        }
      }

      _static_thread_pool& pool_;
      _remote_queue*       queue_;
      Receiver             rcvr_;
      std::size_t          thread_index_{};
      nodemask             constraints_{};
    };

  }  // namespace _pool

  struct static_thread_pool : _pool::_static_thread_pool
  {
    static_thread_pool() = default;

    explicit static_thread_pool(std::uint32_t thread_count,
                                bwos_params   params = {},
                                numa_policy   numa   = get_numa_policy())
      : _pool::_static_thread_pool(thread_count, params, std::move(numa))
    {}
  };
}  // namespace experimental::execution

STDEXEC_MODULE_EXPORT
namespace exec = experimental::execution;

STDEXEC_PRAGMA_POP()
