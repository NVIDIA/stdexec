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
  //! This is a P2300-style thread pool wrapping base class, which its docs describe as "A class that represents an
  //! explicit, user-managed task scheduler arena."
  //! Once set up, a task arena and it has
  //! * template<F> void enqueue(F &&f)
  //! and
  //! * template<F> auto execute(F &&f) -> decltype(f())
  //!
  using namespace stdexec::tags;

  template <class PoolType, class ReceiverId>
  struct operation {
    using Receiver = stdexec::__t<ReceiverId>;
    struct __t;
  };

  using task_base = exec::static_thread_pool::task_base;

  template <class DerivedPoolType> // CRTP
  class thread_pool_base {
    template <class DerivedPoolType_, class ReceiverId>
    friend struct operation;

   public:
    struct scheduler {
     private:
      template <class DerivedPoolType_, class ReceiverId>
      friend struct operation;

      class sender {
       public:
        using sender_concept = stdexec::sender_t;
        using __t = sender;
        using __id = sender;
        using completion_signatures =
          stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

        template <class CPO>
        auto query(stdexec::get_completion_scheduler_t<CPO>) const noexcept ->
          typename DerivedPoolType::scheduler {
          return pool_.get_scheduler();
        }

        auto get_env() const noexcept -> const sender& {
          return *this;
        }

        template <class Receiver>
        auto connect(Receiver rcvr) const
          -> stdexec::__t<operation<DerivedPoolType, stdexec::__id<Receiver>>> {
          return stdexec::__t<operation<DerivedPoolType, stdexec::__id<Receiver>>>{
            this->pool_, static_cast<Receiver&&>(rcvr)};
        }

       private:
        friend struct DerivedPoolType::scheduler;

        explicit sender(DerivedPoolType& pool) noexcept
          : pool_(pool) {
        }

        DerivedPoolType& pool_;
      };

      template <class Fun, class Shape, class... Args>
      using bulk_non_throwing = stdexec::__mbool<
        // If function invocation doesn't throw
        stdexec::__nothrow_callable<Fun, Shape, Args...> &&
        // and emplacing a tuple doesn't throw
        noexcept(stdexec::__decayed_std_tuple<Args...>(std::declval<Args>()...))
        // there's no need to advertise completion with `exception_ptr`
      >;

      template <class CvrefSender, class Receiver, class Shape, class Fun, bool MayThrow>
      struct bulk_shared_state : task_base {
        using variant_t = stdexec::__value_types_of_t<
          CvrefSender,
          stdexec::env_of_t<Receiver>,
          stdexec::__q<stdexec::__decayed_std_tuple>,
          stdexec::__q<stdexec::__std_variant>
        >;

        variant_t data_;
        DerivedPoolType& pool_;
        Receiver rcvr_;
        Shape shape_;
        Fun fun_;

        std::atomic<std::uint32_t> finished_threads_{0};
        std::atomic<std::uint32_t> thread_with_exception_{0};
        std::exception_ptr exception_;

        // Splits `n` into `size` chunks distributing `n % size` evenly between ranks.
        // Returns `[begin, end)` range in `n` for a given `rank`.
        // Example:
        // ```cpp
        // //         n_items  thread  n_threads
        // even_share(     11,      0,         3); // -> [0,  4) -> 4 items
        // even_share(     11,      1,         3); // -> [4,  8) -> 4 items
        // even_share(     11,      2,         3); // -> [8, 11) -> 3 items
        // ```
        static auto even_share(Shape n, std::size_t rank, std::size_t size) noexcept
          -> std::pair<Shape, Shape> {
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

        [[nodiscard]]
        auto num_agents_required() const -> std::uint32_t {
          // With work stealing, is std::min necessary, or can we feel free to ask for more agents (tasks)
          // than we can actually deal with at one time?
          return static_cast<std::uint32_t>(
            std::min(shape_, static_cast<Shape>(pool_.available_parallelism())));
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
          this->__execute = [](task_base* t, std::uint32_t tid) noexcept {
            auto& self = *static_cast<bulk_shared_state*>(t);
            auto total_threads = self.num_agents_required();

            auto computation = [&](auto&... args) {
              auto [begin, end] = even_share(self.shape_, tid, total_threads);
              for (Shape i = begin; i < end; ++i) {
                self.fun_(i, args...);
              }
            };

            auto completion = [&](auto&... args) {
              stdexec::set_value(static_cast<Receiver&&>(self.rcvr_), std::move(args)...);
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
                  stdexec::set_error(
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

      template <class CvrefSenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
      struct bulk_receiver {
        using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t {
          using __id = bulk_receiver;
          using receiver_concept = stdexec::receiver_t;

          using shared_state = bulk_shared_state<CvrefSender, Receiver, Shape, Fun, MayThrow>;

          shared_state& shared_state_;

          void enqueue() noexcept {
            shared_state_.pool_.bulk_enqueue(&shared_state_, shared_state_.num_agents_required());
          }

          template <class... As>
          void set_value(As&&... as) noexcept {
            using tuple_t = stdexec::__decayed_std_tuple<As...>;

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
              state.apply([&](auto&... args) {
                stdexec::set_value(std::move(state.rcvr_), std::move(args)...);
              });
            }
          }

          template <class Error>
          void set_error(Error&& err) noexcept {
            shared_state& state = shared_state_;
            stdexec::set_error(static_cast<Receiver&&>(state.rcvr_), static_cast<Error&&>(err));
          }

          void set_stopped() noexcept {
            shared_state& state = shared_state_;
            stdexec::set_stopped(static_cast<Receiver&&>(state.rcvr_));
          }

          auto get_env() const noexcept -> stdexec::env_of_t<Receiver> {
            return stdexec::get_env(shared_state_.rcvr_);
          }
        };
      };

      template <class CvrefSenderId, class ReceiverId, std::integral Shape, class Fun>
      struct bulk_op_state {
        using CvrefSender = stdexec::__cvref_t<CvrefSenderId>;
        using Receiver = stdexec::__t<ReceiverId>;

        struct __t {
          using __id = bulk_op_state;
          static constexpr bool may_throw = !stdexec::__v<stdexec::__value_types_of_t<
            CvrefSender,
            stdexec::env_of_t<Receiver>,
            stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
            stdexec::__q<stdexec::__mand>
          >>;

          using bulk_rcvr =
            stdexec::__t<bulk_receiver<CvrefSenderId, ReceiverId, Shape, Fun, may_throw>>;
          using shared_state = bulk_shared_state<CvrefSender, Receiver, Shape, Fun, may_throw>;
          using inner_op_state = stdexec::connect_result_t<CvrefSender, bulk_rcvr>;

          shared_state shared_state_;

          inner_op_state inner_op_;

          void start() & noexcept {
            stdexec::start(inner_op_);
          }

          __t(DerivedPoolType& pool, Shape shape, Fun fun, CvrefSender&& sndr, Receiver rcvr)
            : shared_state_(pool, static_cast<Receiver&&>(rcvr), shape, fun)
            , inner_op_{
                stdexec::connect(static_cast<CvrefSender&&>(sndr), bulk_rcvr{shared_state_})} {
          }
        };
      };

      template <class _Ty>
      using __decay_ref = stdexec::__decay_t<_Ty>&;

      template <class SenderId, std::integral Shape, class Fun>
      struct bulk_sender {
        using Sender = stdexec::__t<SenderId>;

        struct __t {
          using __id = bulk_sender;
          using sender_concept = stdexec::sender_t;

          DerivedPoolType& pool_;
          Sender sndr_;
          Shape shape_;
          Fun fun_;

          template <class Sender, class... Env>
          using _with_error_invoke_t = stdexec::__eptr_completion_if_t<stdexec::__value_types_t<
            stdexec::__completion_signatures_of_t<Sender, Env...>,
            stdexec::__mtransform<
              stdexec::__q1<__decay_ref>,
              stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>
            >,
            stdexec::__q<stdexec::__mand>
          >>;

          template <class... Tys>
          using _set_value_t =
            stdexec::completion_signatures<stdexec::set_value_t(stdexec::__decay_t<Tys>...)>;

          template <class Self, class... Env>
          using completion_signatures = stdexec::transform_completion_signatures<
            stdexec::__completion_signatures_of_t<stdexec::__copy_cvref_t<Self, Sender>, Env...>,
            _with_error_invoke_t<stdexec::__copy_cvref_t<Self, Sender>, Env...>,
            _set_value_t
          >;

          template <class Self, class Receiver>
          using bulk_op_state_t = stdexec::__t<
            bulk_op_state<stdexec::__cvref_id<Self, Sender>, stdexec::__id<Receiver>, Shape, Fun>
          >;

          template <stdexec::__decays_to<__t> Self, stdexec::receiver Receiver>
            requires stdexec::receiver_of<
              Receiver,
              completion_signatures<Self, stdexec::env_of_t<Receiver>>
            >
          static auto
            connect(Self&& self, Receiver rcvr) noexcept(stdexec::__nothrow_constructible_from<
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
              self.fun_,
              static_cast<Self&&>(self).sndr_,
              static_cast<Receiver&&>(rcvr)};
          }

          template <stdexec::__decays_to<__t> Self, class... Env>
          static auto
            get_completion_signatures(Self&&, Env&&...) -> completion_signatures<Self, Env...> {
            return {};
          }

          template <stdexec::__forwarding_query Tag, class... As>
            requires stdexec::__callable<Tag, const Sender&, As...>
          auto query(Tag, As&&... as) const
            noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>) -> decltype(auto) {
            return Tag()(sndr_, static_cast<As&&>(as)...);
          }

          auto get_env() const noexcept -> const __t& {
            return *this;
          }
        };
      };

      template <stdexec::sender Sender, std::integral Shape, class Fun>
      using bulk_sender_t =
        stdexec::__t<bulk_sender<stdexec::__id<stdexec::__decay_t<Sender>>, Shape, Fun>>;

      STDEXEC_MEMFN_FRIEND(bulk);

      template <stdexec::sender S, std::integral Shape, class Fun>
      STDEXEC_MEMFN_DECL(
        auto bulk)(this const scheduler& sch, S&& sndr, Shape shape, Fun fun) noexcept
        -> bulk_sender_t<S, Shape, Fun> {
        return bulk_sender_t<S, Shape, Fun>{
          *sch.pool_, static_cast<S&&>(sndr), shape, static_cast<Fun&&>(fun)};
      }

      [[nodiscard]]
      constexpr auto
        forward_progress_guarantee() const noexcept -> stdexec::forward_progress_guarantee {
        return pool_->forward_progress_guarantee();
      }

      friend thread_pool_base;

      explicit scheduler(DerivedPoolType& pool) noexcept
        : pool_(&pool) {
      }

      DerivedPoolType* pool_;

     public:
      using __t = scheduler;
      using __id = scheduler;
      auto operator==(const scheduler&) const -> bool = default;

      [[nodiscard]]
      constexpr auto query(stdexec::get_forward_progress_guarantee_t) const noexcept
        -> stdexec::forward_progress_guarantee {
        return forward_progress_guarantee();
      }

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
      return static_cast<DerivedPoolType&>(*this).available_parallelism();
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

  template <class PoolType, class ReceiverId>
  struct operation<PoolType, ReceiverId>::__t : task_base {
    using __id = operation;
    friend class thread_pool_base<PoolType>;

    PoolType& pool_;
    Receiver rcvr_;

    explicit __t(PoolType& pool, Receiver rcvr)
      : pool_(pool)
      , rcvr_(std::move(rcvr)) {
      this
        ->__execute = [](task_base* t, std::uint32_t /* tid What is this needed for? */) noexcept {
        auto& op = *static_cast<__t*>(t);
        auto stoken = stdexec::get_stop_token(stdexec::get_env(op.rcvr_));
        if (stoken.stop_requested()) {
          stdexec::set_stopped(std::move(op.rcvr_));
        } else {
          stdexec::set_value(std::move(op.rcvr_));
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
