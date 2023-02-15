#pragma once

#include <tbb/task_arena.h>

#include <exec/static_thread_pool.hpp>


namespace tbbexec {

template <typename ReceiverId>
class operation;

using task_base = exec::task_base;

//! This is a P2300-style thread pool wrapping tbb::task_arena, which its docs describe as "A class that represents an
//! explicit, user-managed task scheduler arena."
//! Once set up, a tbb::task_arena has
//! * template<F> void enqueue(F &&f)
//! and
//! * template<F> auto execute(F &&f) -> decltype(f())
//!
//! See https://spec.oneapi.io/versions/1.0-rev-3/elements/oneTBB/source/task_scheduler/task_arena/task_arena_cls.html
class tbb_thread_pool {
    template <typename ReceiverId>
    friend class operation;

public:
    tbb_thread_pool() : tbb_thread_pool(tbb::task_arena::attach{}) {}

    explicit tbb_thread_pool(std::uint32_t threadCount) : m_arena{int(threadCount)} { m_arena.initialize(); }

    template <typename... Args>
    explicit tbb_thread_pool(Args&&... args) : m_arena{std::forward<Args>(args)...} {
        m_arena.initialize();
    }

    struct scheduler {
        using T = scheduler;
        using ID = scheduler;
        bool operator==(const scheduler&) const = default;

    private:
        template <typename ReceiverId>
        friend class operation;

        class sender {
        public:
            using T = sender;
            using ID = sender;
            using completion_signatures =
                stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

        private:
            template <typename Receiver>
            operation<stdexec::__x<std::decay_t<Receiver>>> make_operation_(Receiver&& r) const {
                return operation<stdexec::__x<std::decay_t<Receiver>>>{this->pool_, (Receiver &&) r};
            }

            template <class Receiver>
            friend operation<stdexec::__x<std::decay_t<Receiver>>> tag_invoke(stdexec::connect_t,
                                                                              sender s,
                                                                              Receiver&& r) {
                return s.make_operation_(std::forward<Receiver>(r));
            }

            template <class CPO>
            friend tbb_thread_pool::scheduler tag_invoke(stdexec::get_completion_scheduler_t<CPO>, sender s) noexcept {
                return tbb_thread_pool::scheduler{s.pool_};
            }

            friend struct tbb_thread_pool::scheduler;

            explicit sender(tbb_thread_pool& pool) noexcept : pool_(pool) {}

            tbb_thread_pool& pool_;
        };

        template <class Fun, class Shape, class... Args>
            requires stdexec::__callable<Fun, Shape, Args...>
        using bulk_non_throwing = stdexec::__bool<
            // If function invocation doesn't throw
            stdexec::__nothrow_callable<Fun, Shape, Args...>&&
            // and emplacing a tuple doesn't throw
            noexcept(stdexec::__decayed_tuple<Args...>(std::declval<Args>()...))
            // there's no need to advertise completion with `exception_ptr`
            >;

        template <class SenderId, class ReceiverId, class Shape, class Fun, bool MayThrow>
        struct bulk_shared_state : exec::task_base {
            using Sender = stdexec::__t<SenderId>;
            using Receiver = stdexec::__t<ReceiverId>;

            using variant_t = stdexec::__value_types_of_t<Sender,
                                                          stdexec::env_of_t<Receiver>,
                                                          stdexec::__q<stdexec::__decayed_tuple>,
                                                          stdexec::__q<stdexec::__variant>>;

            variant_t data_;
            tbb_thread_pool& pool_;
            Receiver receiver_;
            Shape shape_;
            Fun fn_;

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
            static std::pair<Shape, Shape> even_share(Shape n, std::uint32_t rank, std::uint32_t size) noexcept {
                const auto avg_per_thread = n / size;
                const auto n_big_share = avg_per_thread + 1;
                const auto big_shares = n % size;
                const auto is_big_share = rank < big_shares;
                const auto begin =
                    is_big_share ? n_big_share * rank : n_big_share * big_shares + (rank - big_shares) * avg_per_thread;
                const auto end = begin + (is_big_share ? n_big_share : avg_per_thread);

                return std::make_pair(begin, end);
            }

            std::uint32_t num_agents_required() const {
                // With work stealing, is std::min necessary, or can we feel free to ask for more agents (tasks)
                // than we can actually deal with at one time? I think the answer is "no".
                return shape_;
            }

            template <class F>
            void apply(F f) {
                std::visit([&](auto& tupl) -> void { std::apply([&](auto&... args) -> void { f(args...); }, tupl); },
                           data_);
            }

            bulk_shared_state(tbb_thread_pool& pool, Receiver receiver, Shape shape, Fun fn)
                : pool_(pool)
                , receiver_{(Receiver &&) receiver}
                , shape_{shape}
                , fn_{fn}
                , thread_with_exception_{num_agents_required()} {
                this->__execute = [](task_base* t, std::uint32_t tid) noexcept {
                    auto& self = *static_cast<bulk_shared_state*>(t);
                    auto total_threads = self.num_agents_required();

                    auto computation = [&](auto&... args) {
                        auto [begin, end] = even_share(self.shape_, tid, total_threads);
                        for (Shape i = begin; i < end; ++i) {
                            self.fn_(i, args...);
                        }
                    };

                    auto completion = [&](auto&... args) {
                        stdexec::set_value((Receiver &&) self.receiver_, std::move(args)...);
                    };

                    if constexpr (MayThrow) {
                        try {
                            self.apply(computation);
                        } catch (...) {
                            std::uint32_t expected = total_threads;

                            if (self.thread_with_exception_.compare_exchange_strong(
                                    expected, tid, std::memory_order_relaxed, std::memory_order_relaxed)) {
                                self.exception_ = std::current_exception();
                            }
                        }

                        const bool is_last_thread = self.finished_threads_.fetch_add(1) == (total_threads - 1);

                        if (is_last_thread) {
                            if (self.exception_) {
                                stdexec::set_error((Receiver &&) self.receiver_, self.exception_);
                            } else {
                                self.apply(completion);
                            }
                        }
                    } else {
                        self.apply(computation);

                        const bool is_last_thread = self.finished_threads_.fetch_add(1) == (total_threads - 1);

                        if (is_last_thread) {
                            self.apply(completion);
                        }
                    }
                };
            }
        };

        template <class SenderId, class ReceiverId, class Shape, class Fn, bool MayThrow>
        struct bulk_receiver {
            using Sender = stdexec::__t<SenderId>;
            using Receiver = stdexec::__t<ReceiverId>;

            using shared_state = bulk_shared_state<SenderId, ReceiverId, Shape, Fn, MayThrow>;

            shared_state& shared_state_;

            void enqueue() noexcept {
                shared_state_.pool_.bulk_enqueue(&shared_state_, shared_state_.num_agents_required());
            }

            template <class... As>
            friend void tag_invoke(stdexec::set_value_t, bulk_receiver&& self, As&&... as) noexcept {
                using tuple_t = stdexec::__decayed_tuple<As...>;

                shared_state& state = self.shared_state_;

                if constexpr (MayThrow) {
                    try {
                        state.data_.template emplace<tuple_t>((As &&) as...);
                    } catch (...) {
                        stdexec::set_error(std::move(state.receiver_), std::current_exception());
                    }
                } else {
                    state.data_.template emplace<tuple_t>((As &&) as...);
                }

                if (state.shape_) {
                    self.enqueue();
                } else {
                    state.apply(
                        [&](auto&... args) { stdexec::set_value(std::move(state.receiver_), std::move(args)...); });
                }
            }

            template <stdexec::__one_of<stdexec::set_error_t, stdexec::set_stopped_t> Tag, class... As>
            friend void tag_invoke(Tag tag, bulk_receiver&& self, As&&... as) noexcept {
                shared_state& state = self.shared_state_;
                tag((Receiver &&) state.receiver_, (As &&) as...);
            }

            friend auto tag_invoke(stdexec::get_env_t, const bulk_receiver& self) -> stdexec::env_of_t<Receiver> {
                return stdexec::get_env(self.shared_state_.receiver_);
            }
        };

        template <class SenderId, class ReceiverId, std::integral Shape, class Fun>
        struct bulk_op_state {
            using Sender = stdexec::__t<SenderId>;
            using Receiver = stdexec::__t<ReceiverId>;

            static constexpr bool may_throw =
                !stdexec::__v<stdexec::__value_types_of_t<Sender,
                                                          stdexec::env_of_t<Receiver>,
                                                          stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
                                                          stdexec::__q<stdexec::__mand>>>;

            using bulk_rcvr = bulk_receiver<SenderId, ReceiverId, Shape, Fun, may_throw>;
            using shared_state = bulk_shared_state<SenderId, ReceiverId, Shape, Fun, may_throw>;
            using inner_op_state = stdexec::connect_result_t<Sender, bulk_rcvr>;

            shared_state shared_state_;

            inner_op_state inner_op_;

            friend void tag_invoke(stdexec::start_t, bulk_op_state& op) noexcept { stdexec::start(op.inner_op_); }

            bulk_op_state(tbb_thread_pool& pool, Shape shape, Fun fn, Sender&& sender, Receiver receiver)
                : shared_state_(pool, (Receiver &&) receiver, shape, fn)
                , inner_op_{stdexec::connect((Sender &&) sender, bulk_rcvr{shared_state_})} {}
        };

        template <class SenderId, std::integral Shape, class FunId>
        struct bulk_sender {
            using Sender = stdexec::__t<SenderId>;
            using Fun = stdexec::__t<FunId>;

            tbb_thread_pool& pool_;
            Sender sndr_;
            Shape shape_;
            Fun fun_;

            template <class Fun, class Sender, class Env>
            using with_error_invoke_t = stdexec::__if_c<
                stdexec::__v<stdexec::__value_types_of_t<Sender,
                                                         Env,
                                                         stdexec::__mbind_front_q<bulk_non_throwing, Fun, Shape>,
                                                         stdexec::__q<stdexec::__mand>>>,
                stdexec::completion_signatures<>,
                stdexec::__with_exception_ptr>;

            template <class... Tys>
            using set_value_t = stdexec::completion_signatures<stdexec::set_value_t(std::decay_t<Tys>...)>;

            template <class Self, class Env>
            using completion_signatures = stdexec::__make_completion_signatures<
                stdexec::__copy_cvref_t<Self, Sender>,
                Env,
                with_error_invoke_t<Fun, stdexec::__copy_cvref_t<Self, Sender>, Env>,
                stdexec::__q<set_value_t>>;

            template <class Self, class Receiver>
            using bulk_op_state_t = bulk_op_state<stdexec::__x<stdexec::__copy_cvref_t<Self, Sender>>,
                                                  stdexec::__x<std::remove_cvref_t<Receiver>>,
                                                  Shape,
                                                  Fun>;

            template <stdexec::__decays_to<bulk_sender> Self, stdexec::receiver Receiver>
                requires stdexec::receiver_of<Receiver, completion_signatures<Self, stdexec::env_of_t<Receiver>>>
            friend bulk_op_state_t<Self, Receiver> tag_invoke(
                stdexec::connect_t,
                Self&& self,
                Receiver&& rcvr) noexcept(std::is_nothrow_constructible_v<bulk_op_state_t<Self, Receiver>,
                                                                          tbb_thread_pool&,
                                                                          Shape,
                                                                          Fun,
                                                                          Sender,
                                                                          Receiver>) {
                return bulk_op_state_t<Self, Receiver>{self.pool_, self.shape_, self.fun_, ((Self &&) self).sndr_,
                                                       (Receiver &&) rcvr};
            }

            template <stdexec::__decays_to<bulk_sender> Self, class Env>
            friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
                -> stdexec::dependent_completion_signatures<Env>;

            template <stdexec::__decays_to<bulk_sender> Self, class Env>
            friend auto tag_invoke(stdexec::get_completion_signatures_t, Self&&, Env)
                -> completion_signatures<Self, Env>
                requires true
            ;

            template <stdexec::tag_category<stdexec::forwarding_sender_query> Tag, class... As>
                requires stdexec::__callable<Tag, const Sender&, As...>
            friend auto tag_invoke(Tag tag,
                                   const bulk_sender& self,
                                   As&&... as) noexcept(stdexec::__nothrow_callable<Tag, const Sender&, As...>)
                -> stdexec::__call_result_if_t<stdexec::tag_category<Tag, stdexec::forwarding_sender_query>,
                                               Tag,
                                               const Sender&,
                                               As...> {
                return ((Tag &&) tag)(self.sndr_, (As &&) as...);
            }
        };

        friend sender tag_invoke(stdexec::schedule_t, const scheduler& s) noexcept { return sender{*s.pool_}; }

        template <stdexec::sender Sender, std::integral Shape, class Fun>
        using bulk_sender_t =
            bulk_sender<stdexec::__x<std::remove_cvref_t<Sender>>, Shape, stdexec::__x<std::remove_cvref_t<Fun>>>;

        template <stdexec::sender S, std::integral Shape, class Fn>
        friend bulk_sender_t<S, Shape, Fn> tag_invoke(stdexec::bulk_t,
                                                      const scheduler& sch,
                                                      S&& sndr,
                                                      Shape shape,
                                                      Fn fun) noexcept {
            return bulk_sender_t<S, Shape, Fn>{*sch.pool_, (S &&) sndr, shape, (Fn &&) fun};
        }

        friend stdexec::forward_progress_guarantee tag_invoke(stdexec::get_forward_progress_guarantee_t,
                                                              const tbb_thread_pool&) noexcept {
            return stdexec::forward_progress_guarantee::parallel;
        }

        friend class tbb_thread_pool;
        explicit scheduler(tbb_thread_pool& pool) noexcept : pool_(&pool) {}

        tbb_thread_pool* pool_;
    };

    scheduler get_scheduler() noexcept { return scheduler{*this}; }

    void request_stop() noexcept {
        // Should this do anything? TBB supports its own flavor of cancelation at a tbb::task_group level
        // https://spec.oneapi.io/versions/latest/elements/oneTBB/source/task_scheduler/scheduling_controls/task_group_context_cls.html
        // but not at the tbb::task_arena level.
        // https://spec.oneapi.io/versions/latest/elements/oneTBB/source/task_scheduler/task_arena/task_arena_cls.html
    }

    std::uint32_t available_parallelism() const { return m_arena.max_concurrency(); }

private:
    void enqueue(task_base* task) noexcept {
        m_arena.enqueue([task] { task->__execute(task, /*tid=*/0); });
    }
    void bulk_enqueue(task_base* task, std::uint32_t n_threads) noexcept {
        for (std::size_t i = 0; i < n_threads; ++i) {
            // FIXME: Is this right? i is the tid?
            m_arena.enqueue([task, i] { task->__execute(task, /*tid=*/i); });
        }
    }

    tbb::task_arena m_arena;
};

template <typename ReceiverId>
class operation : task_base {
    using Receiver = stdexec::__t<ReceiverId>;
    friend tbb_thread_pool::scheduler::sender;

    tbb_thread_pool& pool_;
    Receiver receiver_;

    explicit operation(tbb_thread_pool& pool, Receiver&& r) : pool_(pool), receiver_(std::move(r)) {
        this->__execute = [](task_base* t, std::uint32_t /* tid What is this needed for? */) noexcept {
            auto& op = *static_cast<operation*>(t);
            auto stoken = stdexec::get_stop_token(stdexec::get_env(op.receiver_));
            if (stoken.stop_requested()) {
                stdexec::set_stopped(std::move(op.receiver_));
            } else {
                stdexec::set_value(std::move(op.receiver_));
            }
        };
    }

    friend void tag_invoke(stdexec::start_t, operation& op) noexcept { op.pool_.enqueue(&op); }
};

} // namespace tbbexec