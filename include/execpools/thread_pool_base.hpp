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

#include <exec/static_thread_pool.hpp>

namespace execpools {
  struct CANNOT_DISPATCH_BULK_ALGORITHM_TO_THE_POOL_SCHEDULER;
  struct BECAUSE_THERE_IS_NO_POOL_SCHEDULER_IN_THE_ENVIRONMENT;
  struct ADD_A_CONTINUES_ON_TRANSITION_TO_THE_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM;

  //! This is a P2300-style thread pool wrapping base class, which its docs describe as "A class that represents an
  //! explicit, user-managed task scheduler arena."
  //! Once set up, a task arena and it has
  //! * template<F> void enqueue(F &&f)
  //! and
  //! * template<F> auto execute(F &&f) -> decltype(f())
  //!
  template <class PoolType, class Receiver>
  struct operation;

  using task_base = exec::static_thread_pool::task_base;

  template <class DerivedPoolType> // CRTP
  class thread_pool_base {
    template <class, class>
    friend struct operation;

   public:
    struct scheduler;

    struct domain : STDEXEC::default_domain {
      template <STDEXEC::sender_expr_for<STDEXEC::bulk_chunked_t> Sender, class Env>
      static constexpr auto transform_sender(STDEXEC::set_value_t, Sender&& sndr, const Env& env) {
        auto& [tag, data, child] = sndr;
        auto& [pol, shape, fun] = data;

        if constexpr (STDEXEC::__completes_on<decltype(child), scheduler, Env>) {
          auto sch =
            STDEXEC::get_completion_scheduler<STDEXEC::set_value_t>(STDEXEC::get_env(child), env);
          using sender_t =
            scheduler::template bulk_sender_t<decltype(child), decltype(shape), decltype(fun)>;
          return sender_t{
            *sch.pool_,
            STDEXEC::__forward_like<Sender>(child),
            shape,
            STDEXEC::__forward_like<Sender>(fun)};
        } else {
          return STDEXEC::__not_a_sender<
            STDEXEC::_WHAT_(CANNOT_DISPATCH_BULK_ALGORITHM_TO_THE_POOL_SCHEDULER),
            STDEXEC::_WHY_(BECAUSE_THERE_IS_NO_POOL_SCHEDULER_IN_THE_ENVIRONMENT),
            STDEXEC::_WHERE_(STDEXEC::_IN_ALGORITHM_, STDEXEC::tag_of_t<Sender>),
            STDEXEC::_TO_FIX_THIS_ERROR_(
              ADD_A_CONTINUES_ON_TRANSITION_TO_THE_POOL_SCHEDULER_BEFORE_THE_BULK_ALGORITHM),
            STDEXEC::_WITH_PRETTY_SENDER_<Sender>,
            STDEXEC::_WITH_ENVIRONMENT_(Env)
          >();
        }
      }

      template <STDEXEC::sender_expr_for<STDEXEC::bulk_unchunked_t> Sender, class Env>
      static constexpr auto transform_sender(STDEXEC::set_value_t, Sender&& sndr, const Env& env);
    };

    struct scheduler {
     private:
      template <class DerivedPoolType_, class Receiver>
      friend struct operation;

      class sender {
       public:
        using sender_concept = STDEXEC::sender_t;
        using completion_signatures =
          STDEXEC::completion_signatures<STDEXEC::set_value_t(), STDEXEC::set_stopped_t()>;

        template <class CPO>
        auto query(STDEXEC::get_completion_scheduler_t<CPO>) const noexcept
          -> DerivedPoolType::scheduler {
          return pool_.get_scheduler();
        }

        template <class CPO>
        auto query(STDEXEC::get_completion_domain_t<CPO>) const noexcept -> domain {
          return {};
        }

        auto get_env() const noexcept -> const sender& {
          return *this;
        }

        template <class Receiver>
        auto connect(Receiver rcvr) const -> operation<DerivedPoolType, Receiver> {
          return operation<DerivedPoolType, Receiver>{this->pool_, static_cast<Receiver&&>(rcvr)};
        }

       private:
        friend struct DerivedPoolType::scheduler;

        explicit sender(DerivedPoolType& pool) noexcept
          : pool_(pool) {
        }

        DerivedPoolType& pool_;
      };

      template <class Fun, class Shape, class... Args>
      using bulk_non_throwing = STDEXEC::__mbool<
        // If function invocation doesn't throw
        STDEXEC::__nothrow_callable<Fun, Shape, Args...> &&
        // and emplacing a tuple doesn't throw
        noexcept(STDEXEC::__decayed_std_tuple<Args...>(std::declval<Args>()...))
        // there's no need to advertise completion with `exception_ptr`
      >;

      template <class CvSender, class Receiver, class Shape, class Fun, bool MayThrow>
      struct bulk_shared_state : task_base {
        using variant_t = STDEXEC::__value_types_of_t<
          CvSender,
          STDEXEC::env_of_t<Receiver>,
          STDEXEC::__q<STDEXEC::__decayed_std_tuple>,
          STDEXEC::__q<STDEXEC::__std_variant>
        >;

        variant_t data_;
        DerivedPoolType& pool_;
        Receiver rcvr_;
        Shape shape_;
        Fun fun_;

        std::atomic<std::uint32_t> finished_threads_{0};
        std::atomic<std::uint32_t> thread_with_exception_{0};
        std::exception_ptr exception_;

        [[nodiscard]]
        auto num_agents_required() const -> std::uint32_t {
          // With work stealing, is std::min necessary, or can we feel free to ask for more agents (tasks)
          // than we can actually deal with at one time?
          return static_cast<std::uint32_t>(
            (std::min) (shape_, static_cast<Shape>(pool_.available_parallelism())));
        }

        template <class F>
        void apply(F f) {
          std::visit(
            [&](auto& tupl) -> void {
              std::apply([&](auto&... args) -> void { f(args...); }, tupl);
            },
            data_);
        }

        bulk_shared_state(DerivedPoolType& pool, Receiver rcvr, Shape shape, Fun fun)
          : pool_(pool)
          , rcvr_{static_cast<Receiver&&>(rcvr)}
          , shape_{shape}
          , fun_{fun}
          , thread_with_exception_{num_agents_required()} {
          this->execute_ = [](task_base* t, std::uint32_t tid) noexcept {
            auto& self = *static_cast<bulk_shared_state*>(t);
            auto total_threads = self.num_agents_required();

            auto computation = [&](auto&... args) {
              auto [begin, end] = exec::_pool_::even_share(self.shape_, tid, total_threads);
              self.fun_(begin, end, args...);
            };

            auto completion = [&](auto&... args) {
              STDEXEC::set_value(static_cast<Receiver&&>(self.rcvr_), std::move(args)...);
            };

            if constexpr (MayThrow) {
              STDEXEC_TRY {
                self.apply(computation);
              }
              STDEXEC_CATCH_ALL {
                std::uint32_t expected = total_threads;

                if (self.thread_with_exception_.compare_exchange_strong(
                      expected, tid, std::memory_order_relaxed, std::memory_order_relaxed)) {
                  self.exception_ = std::current_exception();
                }
              }

              const bool is_last_thread = self.finished_threads_.fetch_add(1)
                                       == (total_threads - 1);

              if (is_last_thread) {
                if (self.exception_) {
                  STDEXEC::set_error(
                    static_cast<Receiver&&>(self.rcvr_), std::move(self.exception_));
                } else {
                  self.apply(completion);
                }
              }
            } else {
              self.apply(computation);

              const bool is_last_thread = self.finished_threads_.fetch_add(1)
                                       == (total_threads - 1);

              if (is_last_thread) {
                self.apply(completion);
              }
            }
          };
        }
      };

      template <class CvSender, class Receiver, class Shape, class Fun, bool MayThrow>
      struct bulk_receiver {
        using receiver_concept = STDEXEC::receiver_t;

        using shared_state = bulk_shared_state<CvSender, Receiver, Shape, Fun, MayThrow>;

        void enqueue() noexcept {
          shared_state_.pool_.bulk_enqueue(&shared_state_, shared_state_.num_agents_required());
        }

        template <class... As>
        void set_value(As&&... as) noexcept {
          using tuple_t = STDEXEC::__decayed_std_tuple<As...>;

          shared_state& state = shared_state_;

          if constexpr (MayThrow) {
            STDEXEC_TRY {
              state.data_.template emplace<tuple_t>(static_cast<As&&>(as)...);
            }
            STDEXEC_CATCH_ALL {
              STDEXEC::set_error(std::move(state.rcvr_), std::current_exception());
            }
          } else {
            state.data_.template emplace<tuple_t>(static_cast<As&&>(as)...);
          }

          if (state.shape_) {
            enqueue();
          } else {
            state.apply([&](auto&... args) {
              STDEXEC::set_value(std::move(state.rcvr_), std::move(args)...);
            });
          }
        }

        template <class Error>
        void set_error(Error&& err) noexcept {
          shared_state& state = shared_state_;
          STDEXEC::set_error(static_cast<Receiver&&>(state.rcvr_), static_cast<Error&&>(err));
        }

        void set_stopped() noexcept {
          shared_state& state = shared_state_;
          STDEXEC::set_stopped(static_cast<Receiver&&>(state.rcvr_));
        }

        auto get_env() const noexcept -> STDEXEC::env_of_t<Receiver> {
          return STDEXEC::get_env(shared_state_.rcvr_);
        }

        shared_state& shared_state_;
      };

      template <class CvSender, class Receiver, std::integral Shape, class Fun>
      struct bulk_op_state {
        static constexpr bool may_throw = !STDEXEC::__value_types_of_t<
          CvSender,
          STDEXEC::env_of_t<Receiver>,
          STDEXEC::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
          STDEXEC::__q<STDEXEC::__mand>
        >::value;

        using bulk_rcvr = bulk_receiver<CvSender, Receiver, Shape, Fun, may_throw>;
        using shared_state = bulk_shared_state<CvSender, Receiver, Shape, Fun, may_throw>;
        using inner_op_state = STDEXEC::connect_result_t<CvSender, bulk_rcvr>;

        shared_state shared_state_;

        inner_op_state inner_op_;

        void start() & noexcept {
          STDEXEC::start(inner_op_);
        }

        bulk_op_state(DerivedPoolType& pool, Shape shape, Fun fun, CvSender&& sndr, Receiver rcvr)
          : shared_state_(pool, static_cast<Receiver&&>(rcvr), shape, fun)
          , inner_op_{STDEXEC::connect(static_cast<CvSender&&>(sndr), bulk_rcvr{shared_state_})} {
        }
      };

      template <class _Ty>
      using __decay_ref = STDEXEC::__decay_t<_Ty>&;

      template <class Sender, std::integral Shape, class Fun>
      struct bulk_sender {
        using sender_concept = STDEXEC::sender_t;

        template <class Self, class... Env>
        using _with_error_invoke_t = STDEXEC::__eptr_completion_unless_t<STDEXEC::__value_types_t<
          STDEXEC::__completion_signatures_of_t<STDEXEC::__copy_cvref_t<Self, Sender>, Env...>,
          STDEXEC::__mtransform<
            STDEXEC::__q1<__decay_ref>,
            STDEXEC::__mbind_front_q<bulk_non_throwing, Fun, Shape>
          >,
          STDEXEC::__q<STDEXEC::__mand>
        >>;

        template <class... Tys>
        using _set_value_t =
          STDEXEC::completion_signatures<STDEXEC::set_value_t(STDEXEC::__decay_t<Tys>...)>;

        template <class Self, class... Env>
        using _completion_signatures_t = STDEXEC::transform_completion_signatures<
          STDEXEC::__completion_signatures_of_t<STDEXEC::__copy_cvref_t<Self, Sender>, Env...>,
          _with_error_invoke_t<Self, Env...>,
          _set_value_t
        >;

        template <class Self, class Receiver>
        using bulk_op_state_t =
          bulk_op_state<STDEXEC::__copy_cvref_t<Self, Sender>, Receiver, Shape, Fun>;

        template <STDEXEC::__decays_to<bulk_sender> Self, STDEXEC::receiver Receiver>
          requires STDEXEC::receiver_of<
            Receiver,
            _completion_signatures_t<Self, STDEXEC::env_of_t<Receiver>>
          >
        STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this Self&& self, Receiver rcvr)
          noexcept(STDEXEC::__nothrow_constructible_from<
                   bulk_op_state_t<Self, Receiver>,
                   DerivedPoolType&,
                   Shape,
                   Fun,
                   Sender,
                   Receiver
          >) -> bulk_op_state_t<Self, Receiver> {
          return bulk_op_state_t<Self, Receiver>{
            self.pool_,
            self.shape_,
            static_cast<Self&&>(self).fun_,
            static_cast<Self&&>(self).sndr_,
            static_cast<Receiver&&>(rcvr)};
        }
        STDEXEC_EXPLICIT_THIS_END(connect)

        template <STDEXEC::__decays_to<bulk_sender> Self, class... Env>
        static consteval auto get_completion_signatures() //
          -> _completion_signatures_t<Self, Env...> {
          return {};
        }

        struct attrs {
          template <STDEXEC::__forwarding_query Tag, class... As>
            requires STDEXEC::__queryable_with<STDEXEC::env_of_t<Sender>, Tag, As...>
          auto query(Tag, As&&... as) const
            noexcept(STDEXEC::__nothrow_queryable_with<STDEXEC::env_of_t<Sender>, Tag, As...>)
              -> decltype(auto) {
            return STDEXEC::__query<Tag>()(STDEXEC::get_env(sndr_.sndr_), static_cast<As&&>(as)...);
          }

          const bulk_sender& sndr_;
        };

        [[nodiscard]]
        auto get_env() const noexcept -> attrs {
          return {*this};
        }

        DerivedPoolType& pool_;
        Sender sndr_;
        Shape shape_;
        Fun fun_;
      };

      template <STDEXEC::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_t = bulk_sender<STDEXEC::__decay_t<Sender>, Shape, Fun>;

      friend thread_pool_base;

      explicit scheduler(DerivedPoolType& pool) noexcept
        : pool_(&pool) {
      }

      DerivedPoolType* pool_;

     public:
      auto operator==(const scheduler&) const -> bool = default;


      [[nodiscard]]
      constexpr auto query(STDEXEC::get_forward_progress_guarantee_t) const noexcept
        -> STDEXEC::forward_progress_guarantee {
        return pool_->forward_progress_guarantee();
      }

      template <STDEXEC::__one_of<STDEXEC::set_value_t, STDEXEC::set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(STDEXEC::get_completion_scheduler_t<Tag>) const noexcept -> scheduler {
        return *this;
      }

      template <STDEXEC::__one_of<STDEXEC::set_value_t, STDEXEC::set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(STDEXEC::get_completion_domain_t<Tag>) const noexcept -> domain {
        return {};
      }

      template <STDEXEC::__one_of<STDEXEC::set_value_t, STDEXEC::set_stopped_t> Tag>
      [[nodiscard]]
      constexpr auto query(STDEXEC::get_completion_behavior_t<Tag>) const noexcept {
        return STDEXEC::completion_behavior::asynchronous;
      }

      [[nodiscard]]
      auto schedule() const noexcept -> sender {
        return sender{*pool_};
      }
    };

    [[nodiscard]]
    auto get_scheduler() noexcept -> scheduler {
      return scheduler{static_cast<DerivedPoolType&>(*this)};
    }

    [[nodiscard]]
    auto available_parallelism() const -> std::uint32_t {
      return static_cast<const DerivedPoolType&>(*this).available_parallelism();
    }

   private:
    void enqueue(task_base* task, std::uint32_t tid = 0) noexcept {
      static_cast<DerivedPoolType&>(*this).enqueue(task, tid);
    }

    void bulk_enqueue(task_base* task, std::uint32_t n_threads) noexcept {
      for (std::uint32_t tid = 0; tid < n_threads; ++tid) {
        this->enqueue(task, tid);
      }
    }
  };

  template <class PoolType, class Receiver>
  struct operation : task_base {
    friend class thread_pool_base<PoolType>;

    PoolType& pool_;
    Receiver rcvr_;

    explicit operation(PoolType& pool, Receiver rcvr)
      : pool_(pool)
      , rcvr_(std::move(rcvr)) {
      this->execute_ = [](task_base* t, std::uint32_t /* tid What is this needed for? */) noexcept {
        auto& op = *static_cast<operation*>(t);
        auto stoken = STDEXEC::get_stop_token(STDEXEC::get_env(op.rcvr_));
        if (stoken.stop_requested()) {
          STDEXEC::set_stopped(std::move(op.rcvr_));
        } else {
          STDEXEC::set_value(std::move(op.rcvr_));
        }
      };
    }

    void enqueue() noexcept {
      pool_.enqueue(this);
    }

    void start() & noexcept {
      enqueue();
    }
  };
} // namespace execpools
