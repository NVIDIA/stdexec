/*
 * Copyright (c) 2023 Ben FrantzDale
 * Copyright (c) 2021-2023 Facebook, Inc. and its affiliates.
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
#  include "../stdexec/__detail/__execution_fwd.hpp"

#  include "../stdexec/__detail/__bulk.hpp"
#  include "../stdexec/__detail/__connect.hpp"
#  include "../stdexec/__detail/__env.hpp"
#  include "../stdexec/__detail/__intrusive_queue.hpp"
#  include "../stdexec/__detail/__manual_lifetime.hpp"
#  include "../stdexec/__detail/__meta.hpp"
#  include "../stdexec/__detail/__operation_states.hpp"
#  include "../stdexec/__detail/__receivers.hpp"
#  include "../stdexec/__detail/__schedulers.hpp"
#  include "../stdexec/__detail/__stop_token.hpp"
#  include "../stdexec/__detail/__type_traits.hpp"
#  include "../stdexec/__detail/__utility.hpp"
#  include "../stdexec/__detail/__variant.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <algorithm>
#    include <atomic>
#    include <concepts>
#    include <cstdint>
#    include <exception>
#    include <functional>
#    include <memory>
#    include <mutex>
#    include <ranges>
#    include <span>
#    include <type_traits>
#    include <utility>
#    include <vector>
#  endif
#endif

#include "completion_behavior.hpp"
#include "completion_signatures.hpp"
#include "sender_for.hpp"
#include "sequence_senders.hpp"

namespace experimental::execution
{
  struct CANNOT_DISPATCH_BULK_ALGORITHM_TO_THE_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  struct schedule_all_t;

  template <class Pool>
  struct thread_pool_base;

  namespace _pool
  {
    struct _task_base;

    template <class>
    struct _domain;

    template <class, class>
    struct _opstate;

    template <class, class, class, class, class>
    struct _bulk_receiver;

    template <class, class, class, class, class>
    struct _bulk_state;

    template <class Pool>
    struct _sender;

    template <class Pool>
    struct _scheduler;

    namespace _schedule_all
    {
      template <class, class, class>
      struct _item_opstate;
    }  // namespace _schedule_all
  }  // namespace _pool

  struct bwos_params
  {
    std::size_t numBlocks{32};
    std::size_t blockSize{8};
  };

  struct thread_pool_access
  {
    //! Pools whose queues are intrusive (a task can be in at most one queue at a time)
    //! opt in to per-agent bulk tasks by providing `bulk_enqueue(std::span<Task>)`.
    //! Task `i` of the span runs with `tid == i`.
    template <class Pool>
    static constexpr bool has_bulk_enqueue_for_tasks =
      requires(Pool& pool, std::span<_pool::_task_base> tasks) { pool.bulk_enqueue(tasks); };

    //! Gets the pool from a pool-provided scheduler type. The scheduler must befriend
    //! `thread_pool_access` and have a `pool_` member convertible to a pointer to `Pool`.
    template <class Pool, class Scheduler>
    static auto get_pool(Scheduler const & sch) noexcept -> Pool&
    {
      return static_cast<Pool&>(*sch.pool_);
    }

    template <class Pool>
    static constexpr auto forward_progress_guarantee(Pool& pool) noexcept  //
      -> STDEXEC::forward_progress_guarantee
    {
      return pool.forward_progress_guarantee();
    }

    template <class Pool>
    static void enqueue(Pool& pool, _pool::_task_base* task) noexcept
    {
      static_assert(noexcept(pool.enqueue(task)));
      pool.enqueue(task);
    }

    template <class Pool>
    static void enqueue(Pool& pool, _pool::_task_base* task, std::uint32_t tid) noexcept
    {
      static_assert(noexcept(pool.enqueue(task, tid)));
      pool.enqueue(task, tid);
    }

    template <class Pool>
    static void bulk_enqueue(Pool& pool, _pool::_task_base* task, std::uint32_t n_threads) noexcept
    {
      static_assert(noexcept(pool.bulk_enqueue(task, n_threads)));
      pool.bulk_enqueue(task, n_threads);
    }

    template <class Pool, class Task>
    static void bulk_enqueue(Pool& pool, std::span<Task> tasks) noexcept
    {
      static_assert(noexcept(pool.bulk_enqueue(tasks)));
      pool.bulk_enqueue(tasks);
    }
  };

  //! This is a P2300-style thread pool wrapping base class, which its docs
  //! describe as "A class that represents an explicit, user-managed task
  //! scheduler arena."
  //!
  //! * template<F> void enqueue(F &&f)
  //! and
  //! * template<F> auto execute(F &&f) -> decltype(f())
  //!
  namespace _pool
  {
    using namespace STDEXEC;

    template <class Pool, class Receiver>
    struct _opstate;

    struct _task_base
    {
      void (*execute_)(_task_base*, std::uint32_t tid) noexcept = nullptr;
      _task_base* next_                                         = nullptr;
    };

    //! @brief Splits `n` into `size` chunks distributing `n % size` evenly
    //! between ranks.
    //!
    //! @returns `[begin, end)` range in `n` for a given `rank`.
    //!
    //! @par Example:
    //!
    //! @code{cpp}
    //! //         n_items  thread  n_threads
    //! _even_share(     11,      0,         3); // -> [0,  4) -> 4 items
    //! _even_share(     11,      1,         3); // -> [4,  8) -> 4 items
    //! _even_share(     11,      2,         3); // -> [8, 11) -> 3 items
    //! @endcode
    template <class Shape>
    constexpr auto _even_share(Shape n, std::size_t rank, std::size_t size) noexcept  //
      -> std::pair<Shape, Shape>
    {
      STDEXEC_ASSERT(n >= 0);
      using ushape_t            = std::make_unsigned_t<Shape>;
      auto const avg_per_thread = static_cast<ushape_t>(n) / size;
      auto const n_big_share    = avg_per_thread + 1;
      auto const big_shares     = static_cast<ushape_t>(n) % size;
      auto const is_big_share   = rank < big_shares;
      auto const begin          = is_big_share
                                  ? n_big_share * rank
                                  : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
      auto const end            = begin + (is_big_share ? n_big_share : avg_per_thread);

      return std::make_pair(static_cast<Shape>(begin), static_cast<Shape>(end));
    }

    template <class Pool>
    struct _sender
    {
      using sender_concept = STDEXEC::sender_tag;

      template <class Self, class Env>
      static consteval auto get_completion_signatures() noexcept
      {
        if constexpr (STDEXEC::unstoppable_token<STDEXEC::stop_token_of_t<Env>>)
        {
          return STDEXEC::completion_signatures<STDEXEC::set_value_t()>();
        }
        else
        {
          return STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>();
        }
      }

      template <class CPO>
      auto query(STDEXEC::get_completion_scheduler_t<CPO>) const noexcept  //
        -> Pool::scheduler
      {
        return pool_.get_scheduler();
      }

      template <class CPO>
      auto query(STDEXEC::get_completion_domain_t<CPO>) const noexcept -> Pool::domain
      {
        return {};
      }

      auto get_env() const noexcept -> _sender const &
      {
        return *this;
      }

      template <class Receiver>
      auto connect(Receiver rcvr) const -> _opstate<Pool, Receiver>
      {
        return _opstate<Pool, Receiver>{pool_, std::move(rcvr)};
      }

     private:
      friend struct _scheduler<Pool>;

      explicit _sender(Pool& pool) noexcept
        : pool_(pool)
      {}

      Pool& pool_;
    };

    template <class Pool>
    struct _scheduler
    {
      using scheduler_concept = scheduler_tag;

      auto operator==(_scheduler const &) const -> bool = default;

      [[nodiscard]]
      constexpr auto
      query(get_forward_progress_guarantee_t) const noexcept -> forward_progress_guarantee
      {
        return thread_pool_access::forward_progress_guarantee(*pool_);
      }

      template <__one_of<set_value_t, set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(get_completion_scheduler_t<Tag>) const noexcept -> _scheduler
      {
        return *this;
      }

      template <__one_of<set_value_t, set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(get_completion_domain_t<Tag>) const noexcept -> Pool::domain
      {
        return {};
      }

      template <__one_of<set_value_t, set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(__get_completion_behavior_t<Tag>) const noexcept
      {
        return exec::completion_behavior::asynchronous;
      }

      [[nodiscard]]
      auto schedule() const noexcept -> _sender<Pool>
      {
        return _sender{*pool_};
      }

     private:
      friend thread_pool_access;
      friend struct _domain<Pool>;           // needs access to pool_
      friend struct thread_pool_base<Pool>;  // needs access to ctor

      explicit _scheduler(Pool& pool) noexcept
        : pool_(std::addressof(pool))
      {}

      Pool* pool_;
    };

    template <class State>
    struct _bulk_task : _task_base
    {
      explicit _bulk_task(State* state) noexcept
        : _task_base{&_bulk_task::do_execute_}
        , state_(state)
      {}

     private:
      static void do_execute_(_task_base* task, std::uint32_t tid) noexcept
      {
        static_cast<_bulk_task*>(task)->state_->run(tid);
      }

      State* state_;
    };

    template <class Pool, class Receiver, class Shape, class Fun, class Variant>
    struct _bulk_state
    {
      using task_t = _bulk_task<_bulk_state>;

      //! If true, the pool wants one task node per agent (e.g., because its queues are
      //! intrusive); otherwise a single node is enqueued once per agent.
      static constexpr bool per_agent_tasks = thread_pool_access::has_bulk_enqueue_for_tasks<Pool>;

      explicit _bulk_state(Pool& pool, Receiver rcvr, Shape shape, Fun fun, bool parallelize)
        noexcept(__nothrow_decay_copyable<Fun> && !per_agent_tasks)
        : pool_(pool)
        , rcvr_{std::move(rcvr)}
        , shape_{shape}
        , fun_{std::move(fun)}
        , parallelize_{parallelize}
        , thread_with_exception_{num_agents_required()}
        , tasks_(make_tasks())
      {}

      [[nodiscard]]
      auto num_agents_required() const noexcept -> std::uint32_t
      {
        using shape_t          = std::common_type_t<Shape, std::uint32_t>;
        auto const parallelism = parallelize_ ? pool_.available_parallelism() : std::uint32_t(1);

        // With work stealing, is std::min necessary, or can we feel free to
        // ask for more agents (tasks) than we can actually deal with at one
        // time?
        return Shape{} < shape_ ? std::uint32_t(std::min<shape_t>(shape_, parallelism))
                                : std::uint32_t(0);
      }

      void enqueue() noexcept
      {
        if constexpr (per_agent_tasks)
        {
          STDEXEC_ASSERT(tasks_.size() == num_agents_required());
          thread_pool_access::bulk_enqueue(pool_, std::span{tasks_});
        }
        else
        {
          thread_pool_access::bulk_enqueue(pool_, &tasks_, num_agents_required());
        }
      }

      void run(std::uint32_t const tid) noexcept
      {
        auto& self          = *this;
        auto  total_threads = self.num_agents_required();

        // Each computation does one or more call to the the bulk function. In
        // the case that the shape is much larger than the total number of
        // threads, then each call to computation will call the function many
        // times.
        auto const [begin, end] = _pool::_even_share(self.shape_, tid, total_threads);
        auto const applicator   = std::bind_front(std::ref(self.fun_), begin, end);
        auto const computation  = std::bind_front(__apply, applicator);
        auto const completion   = STDEXEC::__flip(std::bind_front(__apply, STDEXEC::set_value));

        if constexpr (noexcept(__visit(computation, self.data_)))
        {
          __visit(computation, self.data_);
          if (self.finished_threads_.fetch_add(1) + 1 == total_threads)  // last thread?
            __visit(completion, std::move(self.data_), std::move(self.rcvr_));
        }
        else
        {
          STDEXEC_TRY
          {
            __visit(computation, self.data_);
          }
          STDEXEC_CATCH_ALL
          {
            std::uint32_t expected = total_threads;
            if (self.thread_with_exception_.compare_exchange_strong(expected,
                                                                    tid,
                                                                    std::memory_order_relaxed,
                                                                    std::memory_order_relaxed))
            {
              self.exception_ = std::current_exception();
            }
          }

          if (self.finished_threads_.fetch_add(1) + 1 == total_threads)  // last thread?
          {
            if (self.exception_)
              STDEXEC::set_error(std::move(self.rcvr_), std::move(self.exception_));
            else
              __visit(completion, std::move(self.data_), std::move(self.rcvr_));
          }
        }
      }

      Variant  data_{__no_init};
      Pool&    pool_;
      Receiver rcvr_;
      Shape    shape_;
      Fun      fun_;
      bool     parallelize_;

      std::atomic<std::uint32_t> finished_threads_{0};
      std::atomic<std::uint32_t> thread_with_exception_{0};
      std::exception_ptr         exception_;

      // Either one task per agent, or a single task enqueued once per agent.
      // NB: the tasks point back at this object, which must not move after construction.
      using tasks_t = std::conditional_t<per_agent_tasks, std::vector<task_t>, task_t>;
      tasks_t tasks_;

     private:
      auto make_tasks() -> tasks_t
      {
        if constexpr (per_agent_tasks)
          return tasks_t(num_agents_required(), task_t{this});
        else
          return task_t{this};
      }
    };

    template <class Pool, class Receiver, class Shape, class Fun, class Variant>
    struct _bulk_receiver
    {
      using receiver_concept = receiver_tag;

      template <class... As>
      void set_value(As&&... as) noexcept
      {
        STDEXEC_TRY
        {
          state_->data_.template emplace<__decayed_tuple<As...>>(static_cast<As&&>(as)...);

          if (Shape{} < state_->shape_)
          {
            state_->enqueue();
          }
          else
          {
            __visit(STDEXEC::__flip(std::bind_front(__apply, STDEXEC::set_value)),
                    std::move(state_->data_),
                    std::move(state_->rcvr_));
          }
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (__nothrow_decay_copyable<As...>)
            __std::unreachable();
          else
            STDEXEC::set_error(std::move(state_->rcvr_), std::current_exception());
        }
      }

      template <class Error>
      void set_error(Error&& err) noexcept
      {
        STDEXEC::set_error(std::move(state_->rcvr_), static_cast<Error&&>(err));
      }

      void set_stopped() noexcept
      {
        STDEXEC::set_stopped(std::move(state_->rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<Receiver>
      {
        return STDEXEC::get_env(state_->rcvr_);
      }

      _bulk_state<Pool, Receiver, Shape, Fun, Variant>* state_;
    };

    template <class Pool, class CvSender, class Receiver, std::integral Shape, class Fun>
    struct _bulk_opstate
    {
      using variant_t  = value_types_of_t<CvSender, env_of_t<Receiver>, __decayed_tuple, __variant>;
      using receiver_t = _bulk_receiver<Pool, Receiver, Shape, Fun, variant_t>;

      explicit _bulk_opstate(Pool&      pool,
                             Shape      shape,
                             Fun        fun,
                             bool       parallelize,
                             CvSender&& sndr,
                             Receiver   rcvr)
        noexcept(__nothrow_decay_copyable<Fun> && __nothrow_connectable<CvSender, receiver_t>)
        : state_(pool, std::move(rcvr), shape, std::move(fun), parallelize)
        , inner_op_{STDEXEC::connect(static_cast<CvSender&&>(sndr), receiver_t{&state_})}
      {}

      void start() & noexcept
      {
        STDEXEC::start(inner_op_);
      }

     private:
      _bulk_state<Pool, Receiver, Shape, Fun, variant_t> state_;
      connect_result_t<CvSender, receiver_t>             inner_op_;
    };

    template <class CvSender, class Shape, class Fun, class... Env>
    constexpr auto _get_bulk_sender_cmplsigs_transform()
    {
      return []<class... Args>()
      {
        if constexpr (!__nothrow_decay_copyable<Args...>)
        {
          return exec::throw_compile_time_error<  //
            _WHAT_(_PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE_),
            _WHERE_(_IN_ALGORITHM_, bulk_t),
            _WITH_SENDER_<CvSender>,
            __fn_t<_WITH_ENVIRONMENT_, Env>...>();
        }
        else if constexpr (!__callable<Fun&, Shape, Shape, __decay_t<Args>&...>)
        {
          return exec::throw_compile_time_error<
            _WHAT_(_FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_),
            _WHERE_(_IN_ALGORITHM_, bulk_t),
            _WITH_FUNCTION_(Fun&),
            _WITH_ARGUMENTS_(Shape, Shape, __decay_t<Args> & ...)>();
        }
        else if constexpr (__nothrow_callable<Fun&, Shape, Shape, __decay_t<Args>&...>)
        {
          return completion_signatures<set_value_t(__decay_t<Args>...)>();
        }
        else
        {
          return completion_signatures<set_value_t(__decay_t<Args>...),
                                       set_error_t(std::exception_ptr)>();
        }
      };
    }

    template <class Pool, class Sender, class Shape, class Fun>
    struct _bulk_sender
    {
      using sender_concept = sender_tag;

      template <class Self, class Receiver>
      using bulk_opstate_t =
        _bulk_opstate<Pool, __copy_cvref_t<Self, Sender>, Receiver, Shape, Fun>;

      explicit _bulk_sender(Pool& pool, Sender sndr, Shape shape, Fun fun, bool parallelize)
        : pool_(pool)
        , sndr_(std::move(sndr))
        , shape_(shape)
        , fun_(std::move(fun))
        , parallelize_(parallelize)
      {}

      // There is no ';' after STDEXEC_AUTO_RETURN(...), because -Wpedantic flags it
      // as extra. Without it, clang-format runs connect into the next member.
      // clang-format off
      template <__decays_to<_bulk_sender> Self, receiver Receiver>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
        STDEXEC_AUTO_RETURN(bulk_opstate_t<Self, Receiver>(self.pool_,
                                                           Shape(self.shape_),
                                                           static_cast<Self&&>(self).fun_,
                                                           self.parallelize_,
                                                           static_cast<Self&&>(self).sndr_,
                                                           std::move(rcvr)))
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <__decays_to<_bulk_sender> Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        using sender_t = __copy_cvref_t<Self, Sender>;
        return exec::transform_completion_signatures(
          STDEXEC::get_completion_signatures<sender_t, Env...>(),
          _pool::_get_bulk_sender_cmplsigs_transform<sender_t, Shape, Fun, Env...>());
      }
      // clang-format on

      struct attrs
      {
        template <class _Env2 = env_of_t<Sender>, __forwarding_query Tag, class... As>
          requires __queryable_with<_Env2, Tag, As...>
        auto query(Tag, As&&... as) const noexcept(__nothrow_queryable_with<_Env2, Tag, As...>)
          -> decltype(auto)
        {
          return STDEXEC::get_env(sndr_.sndr_).query(Tag(), static_cast<As&&>(as)...);
        }

        _bulk_sender const & sndr_;
      };

      [[nodiscard]]
      auto get_env() const noexcept -> attrs
      {
        return {*this};
      }

     private:
      Pool&  pool_;
      Sender sndr_;
      Shape  shape_;
      Fun    fun_;
      bool   parallelize_;
    };

    template <class Pool>
    struct _domain : default_domain
    {
     protected:
      //! Lets pool-specific derived domains reach the pool without befriending
      //! `thread_pool_access` (friendship is not inherited).
      template <class Scheduler>
      static auto _get_pool(Scheduler const & sch) noexcept -> Pool&
      {
        return thread_pool_access::get_pool<Pool>(sch);
      }

     public:
      template <class Sender, class Env>
        requires sender_for<Sender, bulk_chunked_t> || sender_for<Sender, bulk_unchunked_t>
      static constexpr auto transform_sender(set_value_t, Sender&& sndr, Env const & env)
      {
        auto& [tag, data, child] = sndr;
        auto& [pol, shape, fun]  = data;
        using tag_t              = decltype(tag);
        using child_t            = decltype(child);
        using fun_t              = decltype(fun);

        if constexpr (__completes_on<child_t, typename Pool::scheduler, Env>)
        {
          auto sch = STDEXEC::get_completion_scheduler<set_value_t>(STDEXEC::get_env(child), env);
          static_assert(std::is_same_v<decltype(sch), typename Pool::scheduler>);
          auto& pool = thread_pool_access::get_pool<Pool>(sch);

          using policy_type = __decay_t<decltype(pol.__get())>;
          constexpr bool parallelize =
            __one_of<policy_type, parallel_policy, parallel_unsequenced_policy>;

          if constexpr (sender_for<Sender, bulk_chunked_t>)
          {
            return _bulk_sender{pool,
                                STDEXEC::__forward_like<Sender>(child),
                                shape,
                                STDEXEC::__forward_like<Sender>(fun),
                                parallelize};
          }
          else
          {
            using chunked_fun_t = __bulk::__as_bulk_chunked_fn<fun_t>;
            return _bulk_sender{pool,
                                STDEXEC::__forward_like<Sender>(child),
                                shape,
                                chunked_fun_t{STDEXEC::__forward_like<Sender>(fun)},
                                parallelize};
          }
        }
        else
        {
          return __not_a_sender<
            _WHAT_(CANNOT_DISPATCH_BULK_ALGORITHM_TO_THE_POOL_SCHEDULER),
            _WHY_(BECAUSE_THERE_IS_NO_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
            _WHERE_(_IN_ALGORITHM_, tag_t),
            _TO_FIX_THIS_ERROR_(
              ADD_A_CONTINUES_ON_TRANSITION_TO_THE_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
            _WITH_SENDER_<Sender>,
            _WITH_ENVIRONMENT_(Env)>();
        }
      }
    };

    namespace _schedule_all
    {
      template <class Receiver>
      auto _get_allocator(Receiver const & rcvr)
      {
        if constexpr (__callable<get_allocator_t, env_of_t<Receiver>>)
        {
          return STDEXEC::get_allocator(STDEXEC::get_env(rcvr));
        }
        else
        {
          return std::allocator<char>{};
        }
      }

      template <class Receiver>
      using _allocator_of_t = decltype(_schedule_all::_get_allocator(__declval<Receiver>()));

      template <class Range, class Pool>
      struct _state_base
      {
        Range                                 range_;
        Pool&                                 pool_;
        std::mutex                            start_mutex_{};
        bool                                  has_started_{false};
        __intrusive_queue<&_task_base::next_> tasks_{};
        std::size_t                           tasks_size_{};
        __std::atomic<std::size_t>            countdown_{std::ranges::size(range_)};
      };

      template <class Range, class Receiver, class Pool>
      struct _state : _state_base<Range, Pool>
      {
        explicit _state(Range range, Pool& pool, Receiver rcvr)
          : _state_base<Range, Pool>{std::move(range), pool}
          , rcvr_(static_cast<Receiver&&>(rcvr))
        {}

        Receiver rcvr_;
      };

      template <class Range, class ItemReceiver, class Pool>
      struct _item_opstate : _task_base
      {
        explicit _item_opstate(ItemReceiver&&                 item_receiver,
                               std::ranges::iterator_t<Range> it,
                               _state_base<Range, Pool>*      parent)
          : _task_base{&_item_opstate::do_execute_}
          , item_rcvr_(static_cast<ItemReceiver&&>(item_receiver))
          , it_(it)
          , parent_(parent)
        {}

        void start() & noexcept
        {
          std::unique_lock lock{parent_->start_mutex_};
          if (!parent_->has_started_)
          {
            parent_->tasks_.push_back(static_cast<_task_base*>(this));
            parent_->tasks_size_ += 1;
          }
          else
          {
            lock.unlock();
            thread_pool_access::enqueue(parent_->pool_, static_cast<_task_base*>(this));
          }
        }

       private:
        static void do_execute_(_task_base* base, std::uint32_t /* tid */) noexcept
        {
          auto op = static_cast<_item_opstate*>(base);
          STDEXEC::set_value(static_cast<ItemReceiver&&>(op->item_rcvr_), *op->it_);
        }

        ItemReceiver                   item_rcvr_;
        std::ranges::iterator_t<Range> it_;
        _state_base<Range, Pool>*      parent_;
      };

      template <class Range, class Pool>
      struct _item_sender
      {
        using sender_concept = sender_tag;
        using completion_signatures =
          STDEXEC::completion_signatures<set_value_t(std::ranges::range_reference_t<Range>)>;

        struct attrs
        {
          Pool* pool_;

          auto
          query(get_completion_scheduler_t<set_value_t>, __ignore = {}) noexcept -> Pool::scheduler
          {
            return pool_->get_scheduler();
          }
        };

        auto get_env() const noexcept -> attrs
        {
          return {&op_->pool_};
        }

        template <receiver ItemReceiver>
        auto connect(ItemReceiver rcvr) const noexcept
        {
          using item_opstate_t = _item_opstate<Range, ItemReceiver, Pool>;
          return item_opstate_t{static_cast<ItemReceiver&&>(rcvr), it_, op_};
        }

        _state_base<Range, Pool>*      op_;
        std::ranges::iterator_t<Range> it_;
      };

      template <class Range, class Receiver, class Pool>
      struct _next_receiver
      {
        using receiver_concept = receiver_tag;

        void set_value() noexcept
        {
          std::size_t countdown = op_->countdown_.fetch_sub(1, __std::memory_order_relaxed);
          if (countdown == 1)
          {
            STDEXEC::set_value(static_cast<Receiver&&>(op_->rcvr_));
          }
        }

        void set_stopped() noexcept
        {
          std::size_t countdown = op_->countdown_.fetch_sub(1, __std::memory_order_relaxed);
          if (countdown == 1)
          {
            STDEXEC::set_value(static_cast<Receiver&&>(op_->rcvr_));
          }
        }

        auto get_env() const noexcept -> env_of_t<Receiver>
        {
          return STDEXEC::get_env(op_->rcvr_);
        }

        _state<Range, Receiver, Pool>* op_;
      };

      template <class Range, class Receiver, class Pool>
      class _opstate : _state<Range, Receiver, Pool>
      {
       public:
        explicit _opstate(Range range, Pool& pool, Receiver rcvr)
          : _state<Range, Receiver, Pool>{std::move(range), pool, static_cast<Receiver&&>(rcvr)}
          , items_(std::ranges::size(this->range_),
                   item_allocator_t(_schedule_all::_get_allocator(this->rcvr_)))
        {}

        ~_opstate()
        {
          for (std::size_t i = 0; i < items_constructed_; ++i)
          {
            items_[i].__destroy();
          }
        }

        void start() & noexcept
        {
          std::size_t size = items_.size();
          if (size == 0)
          {
            STDEXEC::set_value(std::move(this->rcvr_));
            return;
          }

          std::size_t nthreads = this->pool_.available_parallelism();
          STDEXEC_ASSERT(nthreads > 0);
          bwos_params params     = this->pool_.params();
          std::size_t local_size = params.blockSize * params.numBlocks;
          std::size_t chunk_size = __umax(
            {std::size_t{1}, __umin({size / nthreads, local_size * nthreads})});
          auto&       remote_queue = *this->pool_.get_remote_queue();
          auto        it           = std::ranges::begin(this->range_);
          std::size_t i0           = 0;

          STDEXEC_TRY
          {
            for (std::size_t i = 0; i < size; ++i)
            {
              items_[i].__construct_from(STDEXEC::connect,
                                         set_next(this->rcvr_, item_sender_t{this, it + i}),
                                         next_receiver_t{this});
              ++items_constructed_;
            }
          }
          STDEXEC_CATCH_ALL
          {
            STDEXEC::set_error(std::move(this->rcvr_), std::current_exception());
            return;
          }

          while (i0 + chunk_size < size)
          {
            for (std::size_t i = i0; i < i0 + chunk_size; ++i)
            {
              STDEXEC::start(items_[i].__get());
            }

            std::unique_lock lock{this->start_mutex_};
            this->pool_.bulk_enqueue(remote_queue, std::move(this->tasks_), this->tasks_size_);
            this->tasks_size_ = 0;
            lock.unlock();
            i0 += chunk_size;
          }

          for (std::size_t i = i0; i < size; ++i)
          {
            STDEXEC::start(items_[i].__get());
          }

          std::unique_lock lock{this->start_mutex_};
          this->has_started_ = true;
          this->pool_.bulk_enqueue(remote_queue, std::move(this->tasks_), this->tasks_size_);
        }

       private:
        using allocator_t     = _allocator_of_t<Receiver const &>;
        using item_sender_t   = _item_sender<Range, Pool>;
        using next_sender_t   = next_sender_of_t<Receiver, item_sender_t>;
        using next_receiver_t = _next_receiver<Range, Receiver, Pool>;
        using item_opstate_t  = connect_result_t<next_sender_t, next_receiver_t>;

        using item_allocator_t = std::allocator_traits<allocator_t>::template rebind_alloc<
          __manual_lifetime<item_opstate_t>>;

        std::vector<__manual_lifetime<item_opstate_t>, item_allocator_t> items_;
        std::size_t                                                      items_constructed_{};
      };

      template <class Range, class Pool>
      struct _sequence
      {
        using sender_concept = sequence_sender_tag;
        using item_types     = exec::item_types<_item_sender<Range, Pool>>;
        using completion_signatures =
          STDEXEC::completion_signatures<set_value_t(),
                                         set_error_t(std::exception_ptr),
                                         set_stopped_t()>;

        explicit _sequence(Range range, Pool& pool)
          : range_(static_cast<Range&&>(range))
          , pool_(std::addressof(pool))
        {}

        template <exec::sequence_receiver_of<item_types> Receiver>
        auto subscribe(Receiver rcvr) && noexcept -> _opstate<Range, Receiver, Pool>
        {
          return _opstate<Range, Receiver, Pool>{std::move(range_), *pool_, std::move(rcvr)};
        }

        template <exec::sequence_receiver_of<item_types> Receiver>
          requires __decay_copyable<Range const &>
        auto subscribe(Receiver rcvr) const & noexcept -> _opstate<Range, Receiver, Pool>
        {
          return _opstate<Range, Receiver, Pool>{range_, *pool_, std::move(rcvr)};
        }

       private:
        Range range_;
        Pool* pool_;
      };
    }  // namespace _schedule_all
  }  // namespace _pool

  template <class Pool>  // CRTP
  struct thread_pool_base
  {
    using domain    = _pool::_domain<Pool>;
    using scheduler = _pool::_scheduler<Pool>;

    [[nodiscard]]
    auto get_scheduler() noexcept -> scheduler
    {
      return scheduler{static_cast<Pool&>(*this)};
    }

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t
    {
      return static_cast<Pool const &>(*this).available_parallelism();
    }

   protected:
    // Protected (rather than private + `friend Pool`) so that `Pool` may derive from
    // `thread_pool_base` indirectly, through an implementation class.
    thread_pool_base() = default;

   private:
    friend Pool;
    friend thread_pool_access;

    void enqueue(_pool::_task_base* task) noexcept
    {
      thread_pool_access::enqueue(static_cast<Pool&>(*this), task);
    }

    void enqueue(_pool::_task_base* task, std::uint32_t tid) noexcept
    {
      thread_pool_access::enqueue(static_cast<Pool&>(*this), task, tid);
    }

    void bulk_enqueue(_pool::_task_base* task, std::uint32_t n_threads) noexcept
    {
      for (std::uint32_t tid = 0; tid < n_threads; ++tid)
      {
        enqueue(task, tid);
      }
    }
  };

  namespace _pool
  {
    template <class Pool, class Receiver>
    struct _opstate : _task_base
    {
      explicit _opstate(Pool& pool, Receiver rcvr) noexcept
        : _task_base{&do_execute_}
        , pool_(pool)
        , rcvr_(std::move(rcvr))
      {}

      void start() & noexcept
      {
        thread_pool_access::enqueue(pool_, this);
      }

     private:
      static void do_execute_(_task_base* task, std::uint32_t) noexcept
      {
        auto& self   = *static_cast<_opstate*>(task);
        auto  stoken = STDEXEC::get_stop_token(STDEXEC::get_env(self.rcvr_));

        if constexpr (unstoppable_token<decltype(stoken)>)
        {  // NOLINT(bugprone-branch-clone)
          STDEXEC::set_value(std::move(self.rcvr_));
        }
        else if (stoken.stop_requested())
        {
          STDEXEC::set_stopped(std::move(self.rcvr_));
        }
        else
        {
          STDEXEC::set_value(std::move(self.rcvr_));
        }
      }

      Pool&    pool_;
      Receiver rcvr_;
    };
  }  // namespace _pool

  struct schedule_all_t
  {
    template <class Pool, class Range>
    auto operator()(Pool& pool, Range&& range) const
    {
      return _pool::_schedule_all::_sequence{static_cast<Range&&>(range), pool};
    }
  };

  inline constexpr schedule_all_t schedule_all{};
}  // namespace experimental::execution

namespace exec = experimental::execution;
