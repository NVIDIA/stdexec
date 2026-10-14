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

#if STDEXEC_USE_MODULES()
import std;
import stdexec;
#else
#  include "../stdexec/__detail/__execution_fwd.hpp"

#  include "../stdexec/__detail/__bulk.hpp"
#  include "../stdexec/__detail/__connect.hpp"
#  include "../stdexec/__detail/__env.hpp"
#  include "../stdexec/__detail/__meta.hpp"
#  include "../stdexec/__detail/__operation_states.hpp"
#  include "../stdexec/__detail/__receivers.hpp"
#  include "../stdexec/__detail/__schedulers.hpp"
#  include "../stdexec/__detail/__transform_completion_signatures.hpp"
#  include "../stdexec/__detail/__type_traits.hpp"
#  include "../stdexec/__detail/__utility.hpp"

#  include <algorithm>
#  include <atomic>
#  include <concepts>
#  include <cstdint>
#  include <exception>
#  include <tuple>
#  include <type_traits>
#  include <utility>
#  include <variant>
#endif

#include "completion_signatures.hpp"
#include "sender_for.hpp"
#include "static_thread_pool.hpp"

namespace experimental::execution
{
  struct CANNOT_DISPATCH_BULK_ALGORITHM_TO_THE_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  //! This is a P2300-style thread pool wrapping base class, which its docs
  //! describe as "A class that represents an explicit, user-managed task
  //! scheduler arena."
  //!
  //! * template<F> void enqueue(F &&f)
  //! and
  //! * template<F> auto execute(F &&f) -> decltype(f())
  //!
  namespace _pool_
  {
    template <class PoolType, class Receiver>
    struct opstate;

    using task_base = exec::static_thread_pool::task_base;
  }  // namespace _pool_

  template <class DerivedPoolType>  // CRTP
  class thread_pool_base
  {
    template <class, class>
    friend struct _pool_::opstate;

    friend DerivedPoolType;

    thread_pool_base() = default;

   public:
    struct scheduler;

    struct domain : STDEXEC::default_domain
    {
      template <class Sender, class Env>
        requires sender_for<Sender, STDEXEC::bulk_chunked_t>
              || sender_for<Sender, STDEXEC::bulk_unchunked_t>
      static constexpr auto transform_sender(STDEXEC::set_value_t, Sender&& sndr, Env const & env)
      {
        auto& [tag, data, child] = sndr;
        auto& [pol, shape, fun]  = data;
        using tag_t              = decltype(tag);
        using child_t            = decltype(child);
        using shape_t            = decltype(shape);
        using fun_t              = decltype(fun);

        if constexpr (STDEXEC::__completes_on<child_t, scheduler, Env>)
        {
          auto sch =
            STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(child), env);

          using policy_type          = STDEXEC::__decay_t<decltype(pol.__get())>;
          constexpr bool parallelize = STDEXEC::__one_of<policy_type,
                                                         STDEXEC::parallel_policy,
                                                         STDEXEC::parallel_unsequenced_policy>;

          if constexpr (sender_for<Sender, STDEXEC::bulk_chunked_t>)
          {
            using sender_t = scheduler::template bulk_sender_t<child_t, shape_t, fun_t>;
            return sender_t{*sch.pool_,
                            STDEXEC::__forward_like<Sender>(child),
                            shape,
                            STDEXEC::__forward_like<Sender>(fun),
                            parallelize};
          }
          else
          {
            using chunked_fun_t = STDEXEC::__bulk::__as_bulk_chunked_fn<fun_t>;
            using sender_t = scheduler::template bulk_sender_t<child_t, shape_t, chunked_fun_t>;
            return sender_t{*sch.pool_,
                            STDEXEC::__forward_like<Sender>(child),
                            shape,
                            chunked_fun_t{STDEXEC::__forward_like<Sender>(fun)},
                            parallelize};
          }
        }
        else
        {
          return STDEXEC::__not_a_sender<
            STDEXEC::_WHAT_(CANNOT_DISPATCH_BULK_ALGORITHM_TO_THE_POOL_SCHEDULER),
            STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
            STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, tag_t),
            STDEXEC::_TO_FIX_THIS_ERROR_(
              ADD_A_CONTINUES_ON_TRANSITION_TO_THE_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
            STDEXEC::_WITH_SENDER_<Sender>,
            STDEXEC::_WITH_ENVIRONMENT_(Env)>();
        }
      }
    };

    struct scheduler
    {
      using scheduler_concept = STDEXEC::scheduler_tag;

     private:
      template <class, class>
      friend struct _pool_::opstate;

      class sender
      {
       public:
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
            return STDEXEC::completion_signatures<STDEXEC::set_value_t(),
                                                  STDEXEC::set_stopped_t()>();
          }
        }

        template <class CPO>
        auto query(STDEXEC::get_completion_scheduler_t<CPO>) const noexcept  //
          -> DerivedPoolType::scheduler
        {
          return pool_.get_scheduler();
        }

        template <class CPO>
        auto query(STDEXEC::get_completion_domain_t<CPO>) const noexcept -> domain
        {
          return {};
        }

        auto get_env() const noexcept -> sender const &
        {
          return *this;
        }

        template <class Receiver>
        auto connect(Receiver rcvr) const -> _pool_::opstate<DerivedPoolType, Receiver>
        {
          return _pool_::opstate<DerivedPoolType, Receiver>{pool_, std::move(rcvr)};
        }

       private:
        using derived_pool_type = DerivedPoolType;
        friend struct derived_pool_type::scheduler;

        explicit sender(DerivedPoolType& pool) noexcept
          : pool_(pool)
        {}

        DerivedPoolType& pool_;
      };

      template <class CvSender, class Receiver, class Shape, class Fun>
      struct bulk_shared_state : _pool_::task_base
      {
        using variant_t = STDEXEC::value_types_of_t<CvSender,
                                                    STDEXEC::env_of_t<Receiver>,
                                                    STDEXEC::__decayed_tuple,
                                                    STDEXEC::__variant>;

        explicit bulk_shared_state(DerivedPoolType& pool,
                                   Receiver         rcvr,
                                   Shape            shape,
                                   Fun              fun,
                                   bool             parallelize)
          noexcept(STDEXEC::__nothrow_decay_copyable<Fun>)
          : _pool_::task_base{&do_execute_}
          , pool_(pool)
          , rcvr_{std::move(rcvr)}
          , shape_{shape}
          , fun_{std::move(fun)}
          , parallelize_{parallelize}
          , thread_with_exception_{num_agents_required()}
        {}

        [[nodiscard]]
        auto num_agents_required() const noexcept -> std::uint32_t
        {
          using common_shape_t   = std::common_type_t<Shape, std::uint32_t>;
          auto const parallelism = parallelize_ ? pool_.available_parallelism() : std::uint32_t(1);

          // With work stealing, is std::min necessary, or can we feel free to
          // ask for more agents (tasks) than we can actually deal with at one
          // time?
          return Shape{} < shape_ ? std::uint32_t(std::min<common_shape_t>(shape_, parallelism))
                                  : std::uint32_t(0);
        }

        static void do_execute_(_pool_::task_base* task, std::uint32_t const tid) noexcept
        {
          using namespace STDEXEC;
          auto& self          = *static_cast<bulk_shared_state*>(task);
          auto  total_threads = self.num_agents_required();

          // Each computation does one or more call to the the bulk function. In
          // the case that the shape is much larger than the total number of
          // threads, then each call to computation will call the function many
          // times.
          auto const [begin, end] = _pool_::even_share(self.shape_, tid, total_threads);
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

        variant_t        data_{STDEXEC::__no_init};
        DerivedPoolType& pool_;
        Receiver         rcvr_;
        Shape            shape_;
        Fun              fun_;
        bool             parallelize_;

        std::atomic<std::uint32_t> finished_threads_{0};
        std::atomic<std::uint32_t> thread_with_exception_{0};
        std::exception_ptr         exception_;
      };

      template <class CvSender, class Receiver, class Shape, class Fun>
      struct bulk_receiver
      {
        using receiver_concept = STDEXEC::receiver_tag;
        using shared_state_t   = bulk_shared_state<CvSender, Receiver, Shape, Fun>;

        template <class... As>
        void set_value(As&&... as) noexcept
        {
          using namespace STDEXEC;
          STDEXEC_TRY
          {
            shared_state_.data_.template emplace<__decayed_tuple<As...>>(static_cast<As&&>(as)...);

            if (Shape{} < shared_state_.shape_)
              shared_state_.pool_.bulk_enqueue(&shared_state_, shared_state_.num_agents_required());
            else
              __visit(STDEXEC::__flip(std::bind_front(__apply, STDEXEC::set_value)),
                      std::move(shared_state_.data_),
                      std::move(shared_state_.rcvr_));
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (__nothrow_decay_copyable<As...>)
              __std::unreachable();
            else
              STDEXEC::set_error(std::move(shared_state_.rcvr_), std::current_exception());
          }
        }

        template <class Error>
        void set_error(Error&& err) noexcept
        {
          STDEXEC::set_error(std::move(shared_state_.rcvr_), static_cast<Error&&>(err));
        }

        void set_stopped() noexcept
        {
          STDEXEC::set_stopped(std::move(shared_state_.rcvr_));
        }

        auto get_env() const noexcept -> STDEXEC::env_of_t<Receiver>
        {
          return STDEXEC::get_env(shared_state_.rcvr_);
        }

        shared_state_t& shared_state_;
      };

      template <class CvSender, class Receiver, std::integral Shape, class Fun>
      struct bulk_opstate
      {
        using receiver_t = bulk_receiver<CvSender, Receiver, Shape, Fun>;

        explicit bulk_opstate(DerivedPoolType& pool,
                              Shape            shape,
                              Fun              fun,
                              bool             parallelize,
                              CvSender&&       sndr,
                              Receiver         rcvr)
          noexcept(STDEXEC::__nothrow_decay_copyable<Fun>
                   && STDEXEC::__nothrow_connectable<CvSender, receiver_t>)
          : shared_state_(pool, std::move(rcvr), shape, std::move(fun), parallelize)
          , inner_op_{STDEXEC::connect(static_cast<CvSender&&>(sndr), receiver_t{shared_state_})}
        {}

        void start() & noexcept
        {
          STDEXEC::start(inner_op_);
        }

       private:
        bulk_shared_state<CvSender, Receiver, Shape, Fun> shared_state_;
        STDEXEC::connect_result_t<CvSender, receiver_t>   inner_op_;
      };

      template <class Sender, std::integral Shape, class Fun>
      struct bulk_sender
      {
        using sender_concept = STDEXEC::sender_tag;

        template <class Self, class Receiver>
        using bulk_opstate_t =
          bulk_opstate<STDEXEC::__copy_cvref_t<Self, Sender>, Receiver, Shape, Fun>;

        explicit bulk_sender(DerivedPoolType& pool,
                             Sender           sndr,
                             Shape            shape,
                             Fun              fun,
                             bool             parallelize)
          : pool_(pool)
          , sndr_(std::move(sndr))
          , shape_(shape)
          , fun_(std::move(fun))
          , parallelize_(parallelize)
        {}

        template <STDEXEC::__decays_to<bulk_sender> Self, STDEXEC::receiver Receiver>
        STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
          STDEXEC_AUTO_RETURN(bulk_opstate_t<Self, Receiver>(self.pool_,
                                                             Shape(self.shape_),
                                                             static_cast<Self&&>(self).fun_,
                                                             self.parallelize_,
                                                             static_cast<Self&&>(self).sndr_,
                                                             std::move(rcvr)));
        STDEXEC_EXPLICIT_THIS_END(connect)

        template <STDEXEC::__decays_to<bulk_sender> Self, class... Env>
        static consteval auto get_completion_signatures()
        {
          using namespace STDEXEC;
          return exec::transform_completion_signatures(
            STDEXEC::get_completion_signatures<STDEXEC::__copy_cvref_t<Self, Sender>, Env...>(),
            []<class... Args>()
            {
              if constexpr (!__nothrow_decay_copyable<Args...>)
              {
                return exec::throw_compile_time_error<
                  _WHAT_(_PREDECESSOR_RESULTS_ARE_NOT_DECAY_COPYABLE_),
                  _WHERE_(_IN_ALGORITHM_, bulk_t),
                  _WITH_SENDER_<__copy_cvref_t<Self, Sender>>,
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
            });
        }

        struct attrs
        {
          template <class _Env2 = STDEXEC::env_of_t<Sender>,
                    STDEXEC::__forwarding_query Tag,
                    class... As>
            requires STDEXEC::__queryable_with<_Env2, Tag, As...>
          auto query(Tag, As&&... as) const
            noexcept(STDEXEC::__nothrow_queryable_with<_Env2, Tag, As...>) -> decltype(auto)
          {
            return STDEXEC::get_env(sndr_.sndr_).query(Tag(), static_cast<As&&>(as)...);
          }

          bulk_sender const & sndr_;
        };

        [[nodiscard]]
        auto get_env() const noexcept -> attrs
        {
          return {*this};
        }

       private:
        DerivedPoolType& pool_;
        Sender           sndr_;
        Shape            shape_;
        Fun              fun_;
        bool             parallelize_;
      };

      template <STDEXEC::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_t = bulk_sender<STDEXEC::__decay_t<Sender>, Shape, Fun>;

      friend thread_pool_base;

      explicit scheduler(DerivedPoolType& pool) noexcept
        : pool_(&pool)
      {}

      DerivedPoolType* pool_;

     public:
      auto operator==(scheduler const &) const -> bool = default;

      [[nodiscard]]
      constexpr auto query(STDEXEC::get_forward_progress_guarantee_t) const noexcept
        -> STDEXEC::forward_progress_guarantee
      {
        return pool_->forward_progress_guarantee();
      }

      template <STDEXEC::__one_of<STDEXEC::set_value_t, STDEXEC::set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(STDEXEC::get_completion_scheduler_t<Tag>) const noexcept -> scheduler
      {
        return *this;
      }

      template <STDEXEC::__one_of<STDEXEC::set_value_t, STDEXEC::set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(STDEXEC::get_completion_domain_t<Tag>) const noexcept -> domain
      {
        return {};
      }

      template <STDEXEC::__one_of<STDEXEC::set_value_t, STDEXEC::set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(STDEXEC::__get_completion_behavior_t<Tag>) const noexcept
      {
        return STDEXEC::__completion_behavior::__asynchronous;
      }

      [[nodiscard]]
      auto schedule() const noexcept -> sender
      {
        return sender{*pool_};
      }
    };

    [[nodiscard]]
    auto get_scheduler() noexcept -> scheduler
    {
      return scheduler{static_cast<DerivedPoolType&>(*this)};
    }

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t
    {
      return static_cast<DerivedPoolType const &>(*this).available_parallelism();
    }

   private:
    void enqueue(_pool_::task_base* task, std::uint32_t tid = 0) noexcept
    {
      auto& self = static_cast<DerivedPoolType&>(*this);
      static_assert(noexcept(self.enqueue(task, tid)));
      self.enqueue(task, tid);
    }

    void bulk_enqueue(_pool_::task_base* task, std::uint32_t n_threads) noexcept
    {
      for (std::uint32_t tid = 0; tid < n_threads; ++tid)
      {
        enqueue(task, tid);
      }
    }
  };

  namespace _pool_
  {
    template <class PoolType, class Receiver>
    struct opstate : _pool_::task_base
    {
      explicit opstate(PoolType& pool, Receiver rcvr) noexcept
        : _pool_::task_base{&do_execute_}
        , pool_(pool)
        , rcvr_(std::move(rcvr))
      {}

      void start() & noexcept
      {
        pool_.enqueue(this);
      }

     private:
      static void do_execute_(_pool_::task_base* task, std::uint32_t) noexcept
      {
        auto& self   = *static_cast<opstate*>(task);
        auto  stoken = STDEXEC::get_stop_token(STDEXEC::get_env(self.rcvr_));

        if constexpr (STDEXEC::unstoppable_token<decltype(stoken)>)
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

      PoolType& pool_;
      Receiver  rcvr_;
    };
  }  // namespace _pool_
}  // namespace experimental::execution

namespace exec = experimental::execution;
