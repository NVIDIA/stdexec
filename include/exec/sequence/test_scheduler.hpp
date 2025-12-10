/*
 * Copyright (c) 2024 Maikel Nadolski
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "../../stdexec/__detail/__completion_signatures.hpp"
#include "../../stdexec/__detail/__concepts.hpp"
#include "../../stdexec/__detail/__execution_fwd.hpp"
#include "../../stdexec/__detail/__intrusive_mpsc_queue.hpp"
#include "../../stdexec/__detail/__meta.hpp"
#include "../../stdexec/__detail/__receivers.hpp"
#include "../../stdexec/__detail/__senders.hpp"
#include "../../stdexec/__detail/__senders_core.hpp"
#include "../../stdexec/__detail/__stop_token.hpp"
#include "../../stdexec/__detail/__transform_completion_signatures.hpp"
#include "../../stdexec/__detail/__type_traits.hpp"

#include "./marbles.hpp"
#include "../__detail/intrusive_heap.hpp"
#include "../sequence.hpp"
#include "../sequence_senders.hpp"
#include "../timed_scheduler.hpp"
#include "../variant_sender.hpp"

#include <chrono>
#include <cstddef>
#include <deque>

namespace exec {
  class test_scheduler;

  //
  // test_context implements a time-scheduler with additional test features.
  //
  // test_context is used to build tests using marble sequence senders
  // and marble recorders
  //
  // test_clock_context & test_clock are used to represent time for the tests
  //
  // test_scheduler is used to queue tasks on the test_context at virtual-time points
  //
  // One or more __test_sequence(s) are constructed from marble
  //   diagrams and each schedules marbles on the test_scheduler
  //   when connected and started
  //
  // Once one or more __test_sequence(s) have been composed into an expression,
  //   the test_context is used to record the expression results as a
  //   set of marbles. To support the testing of infinite sequences
  //   the recording will be requested to stop at a default of 1000ms from
  //   start() if the expression has not completed.
  //
  // The expression result marbles are then compared to an expected
  //   set of marbles generated from a separate marble diagram
  //
  // An example of usage:
  //
  // TEST_CASE(
  // "test_scheduler - test_context marble-sequence never",
  // "[sequence_senders][test_scheduler]") {
  //   test_context __test{};
  //   auto __clock = __test.get_clock();
  //   CHECK(test_clock::time_point{0ms} == __clock.now());
  //
  //   // a sequence that will produce '0' at 1ms from start()
  //   // and then never complete
  //   auto __sequence = __test.get_marble_sequence_from(
  //     "  -0-"_mstr);
  //
  //   // the set of marbles for a sequence that contains '5'
  //   // at 1ms from start() and then is externally stopped
  //   // after 1000ms have elapsed since start()
  //   auto expected = get_marbles_from(__clock,
  //     "=^-5 998ms $"_mstr);
  //
  //   // record an expression that is expected to turn '0' to '5'
  //   auto actual = __test.get_marbles_from(
  //                        __sequence
  //                        | then_each([](char c){ return c+5; }));
  //
  //   CHECK(test_clock::time_point{1000ms} == __clock.now());
  //   CAPTURE(__sequence.__marbles_);
  //   CHECK(expected == actual);
  // }
  //

  struct test_clock_context;

  //
  // test_clock and test_clock_context implement a
  // manually advanced virtual-time clock
  //
  // this is used to run tests that depend on time
  // without consuming actual real-time
  //
  // So tests using virtual-time will complete
  // fast in real-time
  //

  struct test_clock {
    using duration = std::chrono::milliseconds;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = std::chrono::time_point<test_clock>;
    [[maybe_unused]]
    static const bool is_steady = false;

    const test_clock_context* __context_;

    [[maybe_unused, nodiscard]]
    time_point now() const noexcept;
  };

  struct test_clock_context {
    using duration = typename test_clock::duration;
    using rep = duration::rep;
    using period = duration::period;
    using time_point = typename test_clock::time_point;
    [[maybe_unused]]
    static const bool is_steady = test_clock::is_steady;

    time_point __now_{};

    [[maybe_unused, nodiscard]]
    time_point now() const noexcept {
      return __now_;
    }

    auto advance_now_to(time_point __new_now) noexcept -> time_point {
      time_point __old_now = __now_;
      __now_ = __new_now;
      return __old_now;
    }

    auto advance_now_by(duration __by) noexcept -> time_point {
      time_point __old_now = __now_;
      __now_ += __by;
      return __old_now;
    }
  };

  inline typename test_clock::time_point test_clock::now() const noexcept {
    return __context_->now();
  }

  namespace _tst_sched {
    using namespace stdexec::tags;

    struct test_operation_base {
      enum class command_type {
        uninitialized,
        schedule,
        stop
      };

      enum class location {
        uninitialized,
        inert,
        in_command_queue,
        in_heap
      };

      ~test_operation_base() noexcept {
        STDEXEC_ASSERT(location_ == location::inert);
      }

      test_operation_base(
        void (*set_value)(test_operation_base*) noexcept,
        command_type command = command_type::schedule) noexcept
        : command_{command}
        , set_value_{set_value} {
      }

      stdexec::__std::atomic<void*> next_{nullptr};
      location location_{location::inert};
      command_type command_;
      void (*set_value_)(test_operation_base*) noexcept;
    };

    template <class Tp>
    struct when_type {
      when_type() = default;

      explicit when_type(Tp tp, std::size_t n = 0) noexcept
        : time_point{std::move(tp)}
        , counter{n} {
      }

      Tp time_point{};
      std::size_t counter{};

      friend auto operator<(const when_type& lhs, const when_type& rhs) noexcept -> bool {
        return lhs.time_point < rhs.time_point
            || (!(rhs.time_point < lhs.time_point) && lhs.counter < rhs.counter);
      }
    };

    struct alignas(64) test_schedule_operation_base : test_operation_base {
      using time_point = test_clock::time_point;

      test_schedule_operation_base(
        time_point tp,
        void (*set_stopped)(test_operation_base*) noexcept,
        void (*set_value)(test_operation_base*) noexcept) noexcept
        : test_operation_base{set_value, command_type::schedule}
        , time_point_{tp}
        , set_stopped_{set_stopped} {
      }

      time_point time_point_;
      // we increase the when counter to ensure that the heap is stable
      // when two operations have the same time_point
      // We do so only when the operation is started, not when it is constructed
      when_type<time_point> when_{};
      test_schedule_operation_base* prev_ = nullptr;
      test_schedule_operation_base* left_ = nullptr;
      test_schedule_operation_base* right_ = nullptr;
      void (*set_stopped_)(test_operation_base*) noexcept;
    };

    struct alignas(64) test_stop_operation : test_operation_base {
      test_stop_operation(
        void (*set_value)(test_operation_base*) noexcept,
        test_schedule_operation_base* target) noexcept
        : test_operation_base{set_value, command_type::stop}
        , target_{target} {
      }

      test_schedule_operation_base* target_;
    };

    template <class Rcvr>
    struct test_schedule_at_op {
      class __t;
    };

    struct __recording_receiver;
    struct __test_sequence;
    struct __test_sequence_operation_base;
  } // namespace _tst_sched

  class test_context {
   private:
    static constexpr std::ptrdiff_t context_closed = std::numeric_limits<std::ptrdiff_t>::min() / 2;
   public:
    using duration = test_clock::duration;
    using time_point = test_clock::time_point;

    auto get_scheduler() noexcept -> test_scheduler;
    auto get_clock() const noexcept -> test_clock;
    auto now() const noexcept -> time_point;

    // parse a marble diagram into a set of marbles
    template <std::size_t _Len>
    auto get_marbles_from(stdexec::__mstring<_Len> __diagram) noexcept
      -> std::vector<marble_t<test_clock>> {
      return exec::get_marbles_from(get_clock(), __diagram);
    }

    // record the results of a sequence-sender as a set of marbles
    template <class _Sequence>
    auto get_marbles_from(
      _Sequence&& __sequence,
      typename test_clock::duration __stop_after = std::chrono::milliseconds(1000)) noexcept
      -> std::vector<marble_t<test_clock>>;


    // return a sequence sender that will emit signals specified by the
    // set of marbles provided
    inline auto get_marble_sequence_from(std::vector<marble_t<test_clock>> __marbles) noexcept
      -> _tst_sched::__test_sequence;

    // parse a marble diagram into a set of marbles and return a sequence
    // sender that will emit those marbles
    template <std::size_t _Len>
    auto get_marble_sequence_from(stdexec::__mstring<_Len> __diagram) noexcept
      -> _tst_sched::__test_sequence;

   private:
    template <class Rcvr>
    friend struct _tst_sched::test_schedule_at_op;

    using command_type = _tst_sched::test_operation_base;
    using task_type = _tst_sched::test_schedule_operation_base;
    using stop_type = _tst_sched::test_stop_operation;

    void process_command_queue() {
      while (command_type* op = command_queue_.pop_front()) {
        STDEXEC_ASSERT(op->location_ == command_type::location::in_command_queue);
        std::exchange(op->location_, command_type::location::inert);
        if (op->command_ == command_type::command_type::schedule) {
          auto* task = static_cast<task_type*>(op);
          task->when_ = _tst_sched::when_type{task->time_point_, submission_counter_++};
          STDEXEC_ASSERT(task->location_ == command_type::location::inert);
          std::exchange(task->location_, command_type::location::in_heap);
          heap_.insert(task);
        } else {
          STDEXEC_ASSERT(op->command_ == command_type::command_type::stop);
          auto* stop_op = static_cast<stop_type*>(op);
          STDEXEC_ASSERT(stop_op->target_->location_ == command_type::location::in_heap);
          bool __erased = heap_.erase(stop_op->target_);
          std::exchange(stop_op->target_->location_, command_type::location::inert);
          if (__erased) {
            stop_op->target_->set_stopped_(stop_op->target_);
          }
          stop_op->set_value_(stop_op);
        }
      }
    }

    void clear_pending() {
      STDEXEC_ASSERT(stop_requested_);
      std::ptrdiff_t expected = 0;
      while (!n_submissions_in_flight_
                .compare_exchange_weak(expected, context_closed, std::memory_order_relaxed)
             && expected > 0) {
        expected = 0;
      }
      task_type* op = heap_.front();
      while (op) {
        STDEXEC_ASSERT(op->location_ == command_type::location::in_heap);
        heap_.pop_front();
        std::exchange(op->location_, command_type::location::inert);
        op->set_stopped_(op);
        op = heap_.front();
      }
    }

    void run() {
      while (true) {
        process_command_queue();
        task_type* op = heap_.front();
        if (!!op) {
          STDEXEC_ASSERT(op->location_ == command_type::location::in_heap);
          heap_.pop_front();
          std::exchange(op->location_, command_type::location::inert);
          if (__clock_.now() < op->time_point_) {
            __clock_.advance_now_to(op->time_point_);
          }
          op->set_value_(op);
          std::exchange(op, nullptr);
        }
        bool stop_requested = stop_requested_;
        ready_ = false;
        if (stop_requested) {
          clear_pending();
          break;
        }
      }
    }

    void schedule(command_type* op) {
      STDEXEC_ASSERT(op->location_ == command_type::location::inert);
      std::ptrdiff_t n = n_submissions_in_flight_.fetch_add(1, std::memory_order_relaxed);
      if (n < 0) {
        if (op->command_ == command_type::command_type::schedule) {
          static_cast<task_type*>(op)->set_stopped_(op);
        } else {
          STDEXEC_ASSERT(op->command_ == command_type::command_type::stop);
          static_cast<stop_type*>(op)->set_value_(op);
        }
        n_submissions_in_flight_
          .compare_exchange_strong(n, context_closed, std::memory_order_relaxed);
        return;
      }
      std::exchange(op->location_, command_type::location::in_command_queue);
      if (command_queue_.push_back(op)) {
        ready_ = true;
      }
      n_submissions_in_flight_.fetch_sub(1, std::memory_order_relaxed);
    }

    void request_stop() {
      stop_requested_ = true;
      process_command_queue();
      clear_pending();
    }

    friend struct _tst_sched::__recording_receiver;
    friend struct _tst_sched::__test_sequence_operation_base;

    using heap_t = intrusive_heap<
      task_type,
      _tst_sched::when_type<time_point>,
      &task_type::when_,
      &task_type::prev_,
      &task_type::left_,
      &task_type::right_
    >;

    stdexec::__intrusive_mpsc_queue<&command_type::next_> command_queue_;
    heap_t heap_;
    std::atomic<std::ptrdiff_t> n_submissions_in_flight_{0};
    bool ready_{false};
    bool stop_requested_{false};
    std::size_t submission_counter_{1};
    test_clock_context __clock_;
  };

  namespace _tst_sched {
    template <class Receiver>
    class test_schedule_at_op<Receiver>::__t : _tst_sched::test_schedule_operation_base {
     public:
      using __id = test_schedule_at_op;

      __t(test_context& context, test_clock::time_point time_point, Receiver receiver) noexcept
        : _tst_sched::test_schedule_operation_base{
            time_point,
            [](_tst_sched::test_operation_base* op) noexcept {
              auto* self = static_cast<__t*>(op);
              int counter = self->ref_count_.fetch_sub(1, std::memory_order_relaxed);
              if (counter == 1) {
                self->stop_callback_.reset();
                stdexec::set_stopped(std::move(self->receiver_));
              }
            },
            [](_tst_sched::test_operation_base* op) noexcept {
              auto* self = static_cast<__t*>(op);
              int counter = self->ref_count_.fetch_sub(1, std::memory_order_relaxed);
              if (counter == 1) {
                self->stop_callback_.reset();
                stdexec::set_value(std::move(self->receiver_));
              }
            }}
        , context_{context}
        , receiver_{std::move(receiver)}
        , stop_op_{
            [](_tst_sched::test_operation_base* op) noexcept {
              auto* stop = static_cast<_tst_sched::test_stop_operation*>(op);
              auto* self = static_cast<__t*>(stop->target_);
              int counter = self->ref_count_.fetch_sub(1, std::memory_order_relaxed);
              if (counter == 1) {
                self->stop_callback_.reset();
                stdexec::set_stopped(std::move(self->receiver_));
              }
            },
            this} {
      }

      void start() & noexcept {
        stop_callback_
          .emplace(stdexec::get_stop_token(stdexec::get_env(receiver_)), on_stopped_t{*this});
        int expected = 0;
        if (ref_count_.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) {
          schedule_this();
        } else {
          stop_callback_.reset();
          stdexec::set_stopped(std::move(receiver_));
        }
      }

     private:
      void schedule_this() noexcept {
        context_.schedule(this);
      }

      struct on_stopped_t {
        __t& self_;

        void operator()() const noexcept {
          self_.request_stop();
        }
      };

      using callback_type = typename stdexec::stop_token_of_t<
        stdexec::env_of_t<Receiver>
      >::template callback_type<on_stopped_t>;

      void request_stop() noexcept {
        if (ref_count_.fetch_add(1, std::memory_order_relaxed) == 1) {
          context_.schedule(&stop_op_);
        }
      }

      test_context& context_;
      Receiver receiver_;
      _tst_sched::test_stop_operation stop_op_;
      std::optional<callback_type> stop_callback_;
      std::atomic<int> ref_count_{0};
    };

  } // namespace _tst_sched

  class test_scheduler {
   public:
    using time_point = test_clock::time_point;
    using duration = test_clock::duration;

    class schedule_at_sender {
     public:
      using sender_concept = stdexec::sender_t;
      using completion_signatures =
        stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

      schedule_at_sender(test_context& context, test_clock::time_point time_point) noexcept
        : context_{&context}
        , time_point_{time_point} {
      }

      [[nodiscard]]
      auto get_env() const noexcept {
        return stdexec::prop{
          stdexec::get_completion_scheduler<stdexec::set_value_t>, test_scheduler{*context_}};
      }

      template <class Receiver>
      auto connect(Receiver receiver) const & noexcept ->
        typename _tst_sched::test_schedule_at_op<Receiver>::__t {
        return {*context_, time_point_, std::move(receiver)};
      }

     private:
      [[nodiscard]]
      auto get_scheduler() const noexcept -> test_scheduler;

      test_context* context_;
      test_clock::time_point time_point_;
    };

    explicit test_scheduler(test_context& context) noexcept
      : context_{&context} {
    }

    [[nodiscard]]
    auto now() const noexcept -> time_point {
      return context_->now();
    }

    [[nodiscard]]
    auto schedule_at(time_point tp) const noexcept -> schedule_at_sender {
      return schedule_at_sender{*context_, tp};
    }

    [[nodiscard]]
    auto schedule() const noexcept -> schedule_at_sender {
      return schedule_at(time_point());
    }

    auto operator==(const test_scheduler&) const noexcept -> bool = default;

   private:
    test_context* context_;
  };

  inline auto test_context::get_scheduler() noexcept -> test_scheduler {
    return test_scheduler{*this};
  }

  inline auto test_context::get_clock() const noexcept -> test_clock {
    return test_clock{&this->__clock_};
  }

  inline auto test_context::now() const noexcept -> test_context::time_point {
    return __clock_.now();
  }

  namespace _tst_sched {
    struct __recording_receiver {
      using __t = __recording_receiver;
      using __id = __recording_receiver;
      using receiver_concept = stdexec::receiver_t;

      test_context* __context_;
      stdexec::inplace_stop_source* __stop_source_;

      void set_value() noexcept {
        __stop_source_->request_stop();
        __context_->request_stop();
      }
      void set_error(std::exception_ptr) noexcept {
        __stop_source_->request_stop();
        __context_->request_stop();
      }
      void set_stopped() noexcept {
        __stop_source_->request_stop();
        __context_->request_stop();
      }

      using env_t = decltype(stdexec::__env::__join(
        stdexec::prop{stdexec::get_scheduler, stdexec::__declval<test_scheduler>()},
        stdexec::prop{
          stdexec::get_stop_token,
          stdexec::__declval<stdexec::inplace_stop_token&>()}));
      [[nodiscard]]
      auto get_env() const noexcept -> env_t {
        return stdexec::__env::__join(
          stdexec::prop{stdexec::get_scheduler, __context_->get_scheduler()},
          stdexec::prop{stdexec::get_stop_token, __stop_source_->get_token()});
      }
    };

    template <class _ReceiverId>
    struct __next_receiver {
      using __t = __next_receiver;
      using __id = __next_receiver;
      using receiver_concept = stdexec::receiver_t;

      using _Receiver = stdexec::__t<_ReceiverId>;

      _Receiver* __receiver_;

      void set_value() noexcept {
      }
      template <class _Error>
      void set_error(_Error&&) noexcept {
      }
      void set_stopped() noexcept {
      }

      [[nodiscard]]
      auto get_env() const noexcept -> stdexec::env_of_t<_Receiver> {
        return stdexec::get_env(*__receiver_);
      }
    };

    //
    // __proxy.. are a hammer to workaround a type handling bug in clang 19
    //

    template <class _Base, auto _Fn>
    struct __proxy_fn {
      template <class _Derived, class... _Args>
        requires stdexec::__decays_to_derived_from<_Base, _Derived>
      auto operator()(_Derived&& __derived, _Args&&... __args) const
        noexcept(stdexec::__nothrow_callable<
                 decltype(_Fn),
                 stdexec::__copy_cvref_t<_Derived, _Base>,
                 _Args...
        >)
          -> stdexec::__call_result_t<
            decltype(_Fn),
            stdexec::__copy_cvref_t<_Derived, _Base>,
            _Args...
          > {
        return _Fn(
          static_cast<stdexec::__copy_cvref_t<_Derived, _Base>&&>(__derived),
          static_cast<_Args&&>(__args)...);
      };
    };

    template <class _SenderId, class _Receiver>
    struct __proxy_operation {
      using _Sender = stdexec::__t<_SenderId>;
      static_assert(stdexec::__callable<stdexec::connect_t, _Sender, _Receiver>);
      using __operation_t = stdexec::connect_result_t<_Sender, _Receiver>;
      using __t [[maybe_unused]] = __proxy_operation;
      using __id [[maybe_unused]] = __proxy_operation;

      __operation_t __op_;

      __proxy_operation(
        [[maybe_unused]] _Sender&& __sender,
        [[maybe_unused]]
        _Receiver&& __receiver) noexcept(stdexec::__nothrow_connectable<_Sender, _Receiver>)
        : __op_{stdexec::connect(
            static_cast<_Sender&&>(__sender),
            static_cast<_Receiver&&>(__receiver))} {
      }

      void start() noexcept {
        stdexec::start(__op_);
      }
    };

    template <class _SenderId>
    struct __proxy_sender {
      using _Sender = stdexec::__t<_SenderId>;
      using __t [[maybe_unused]] = __proxy_sender;
      using __id [[maybe_unused]] = __proxy_sender;
      using sender_concept = typename _Sender::sender_concept;

      _Sender __sender_;

      explicit __proxy_sender(_Sender __sender)
        : __sender_{__sender} {
      }

      static constexpr auto get_completion_signatures =
        []<stdexec::__decays_to<__proxy_sender> _Self, class... _Env>(_Self&&, _Env&&...) noexcept(
          stdexec::__nothrow_callable<
            stdexec::get_completion_signatures_t,
            stdexec::__copy_cvref_t<_Self, _Sender>,
            _Env...
          >)
        -> stdexec::completion_signatures_of_t<stdexec::__copy_cvref_t<_Self, _Sender>, _Env...> {
        return {};
      };

      template <class _Receiver>
      using __operation_t = __proxy_operation<_SenderId, _Receiver>;

      static constexpr auto connect =
        []<stdexec::__decays_to<__proxy_sender> _Self, class _Receiver>(
          _Self&& __self,
          _Receiver&& __receiver) noexcept(stdexec::__nothrow_connectable<_Sender, _Receiver>)
        -> __operation_t<_Receiver> {
        return __operation_t<_Receiver>{
          static_cast<stdexec::__copy_cvref_t<_Self, _Sender>&&>(__self.__sender_),
          static_cast<_Receiver&&>(__receiver)};
      };
    };

    //
    // __test_sequence.. is a sequence-sender that produces a set of marbles as signals
    //

    struct __test_sequence_operation_base {
      test_context* __context_;
      stdexec::inplace_stop_source __stop_source_{};

      void request_stop() {
        __stop_source_.request_stop();
      }
      void stop_context() {
        __context_->request_stop();
      }
    };

    template <class _ReceiverId>
    struct __test_sequence_operation_part : __test_sequence_operation_base {
      using _Receiver = stdexec::__t<_ReceiverId>;

      using __marble_t = marble_t<test_clock>;
      using __marble_sender_t = __marble_t::__marble_sender_t;
      using __time_point_t = typename test_scheduler::time_point;
      using __receiver_t = __next_receiver<_ReceiverId>;

      std::vector<__marble_t> __marbles_;
      _Receiver __receiver_;
      __marble_t* __end_marble_{nullptr};
      std::size_t __active_ops_ = 0;

      __marble_t __requested_stop_marble_{__time_point_t{}, sequence_stopped};

      struct __stop_callback_fn_t {
        __test_sequence_operation_part* __self_;
        void operator()() const noexcept {
          auto& __self = *__self_;
          __self.__requested_stop_marble_.set_origin_frame(__self.__context_->now());
          // cancel all pending ops
          __self_->request_stop();
          if (__self.__active_ops_ == 0) {
            if (!__self.__end_marble_) {
              __self.__requested_stop_marble_
                .visit_sequence_receiver(static_cast<_Receiver&&>(__self.__receiver_));
            }
          }
        }
      };

      using __stop_callback_t = stdexec::stop_callback_for_t<
        stdexec::stop_token_of_t<stdexec::env_of_t<_Receiver>>,
        __stop_callback_fn_t
      >;

      std::optional<__stop_callback_t> __on_stop_;

      __test_sequence_operation_part(
        std::vector<__marble_t> __marbles,
        test_context* __context,
        _Receiver&& __receiver) noexcept
        : __test_sequence_operation_base{__context}
        , __marbles_{static_cast<std::vector<__marble_t>&&>(__marbles)}
        , __receiver_{static_cast<_Receiver&&>(__receiver)} {
      }

      template <class _Completion>
      static auto __schedule_at(
        __test_sequence_operation_part& __self,
        __marble_t& __marble,
        _Completion&& __completion) noexcept;
      static auto
        __schedule_marble(__test_sequence_operation_part& __self, __marble_t& __marble) noexcept;
    };

    template <class _ReceiverId>
    struct __test_sequence_operation : __test_sequence_operation_part<_ReceiverId> {
      using __part_t = __test_sequence_operation_part<_ReceiverId>;

      using _Receiver = stdexec::__t<_ReceiverId>;

      using __marble_t = typename __part_t::__marble_t;
      using __marble_sender_t = typename __part_t::__marble_sender_t;
      using __time_point_t = typename __part_t::__time_point_t;
      using __receiver_t = typename __part_t::__receiver_t;

      using __scheduled_marble_t =
        stdexec::__call_result_t<decltype(&__part_t::__schedule_marble), __part_t&, __marble_t&>;
      using __marble_op_t = stdexec::connect_result_t<__scheduled_marble_t, __receiver_t>;

      std::deque<stdexec::__optional<__marble_op_t>> __marble_ops_{};

      __test_sequence_operation(
        std::vector<__marble_t> __marbles,
        test_context* __context,
        _Receiver&& __receiver) noexcept
        : __part_t{
            static_cast<std::vector<__marble_t>&&>(__marbles),
            __context,
            static_cast<_Receiver&&>(__receiver)} {
      }

      void start() noexcept;
    };

    template <class _ReceiverId>
    template <class _Completion>
    auto __test_sequence_operation_part<_ReceiverId>::__schedule_at(
      __test_sequence_operation_part<_ReceiverId>& __self,
      marble_t<test_clock>& __marble,
      _Completion&& __completion) noexcept {
      return stdexec::write_env(
               // schedule the marble completion at the specified frame
               exec::sequence(
                 exec::schedule_at(__self.__context_->get_scheduler(), __marble.frame()),
                 static_cast<_Completion&&>(__completion)),
               stdexec::prop{stdexec::get_stop_token, __self.__stop_source_.get_token()})
           | stdexec::upon_error([](auto&&) noexcept { }) | stdexec::upon_stopped([]() noexcept { })
           | stdexec::then([&__self, &__marble]() noexcept {
               // after each completion, update the __test_sequence_operation_part state
               STDEXEC_ASSERT(__self.__active_ops_ > 0);
               if (
                 __marble.error_notification() || __marble.stopped_notification()
                 || __marble.sequence_error() || __marble.sequence_stopped()
                 || __marble.sequence_end()) {
                 // these marbles trigger the whole sequence
                 // to complete with no more items
                 if (!__self.__end_marble_) {
                   // set as the end marble
                   // this determines the signal that will be used to
                   // complete the sequence after all remaining active
                   // operations have completed
                   __self.__end_marble_ = &__marble;
                 }
                 // cancel all pending ops
                 __self.request_stop();
               }
               if (--__self.__active_ops_ == 0) {
                 // all ops are complete,
                 if (!!__self.__end_marble_) {
                   __self.__on_stop_.reset();
                   __self.__end_marble_
                     ->visit_sequence_receiver(static_cast<_Receiver&&>(__self.__receiver_));
                 }
                 // else this sequence never completes -
                 // this sequence must be stopped externally
               }
             });
    }

    template <class _ReceiverId>
    auto __test_sequence_operation_part<_ReceiverId>::__schedule_marble(
      __test_sequence_operation_part<_ReceiverId>& __self,
      marble_t<test_clock>& __marble) noexcept {

      using __next_t = decltype(exec::set_next(__self.__receiver_, __marble.visit_sender()));
      using __next_sender_t =
        decltype(__schedule_at(__self, __marble, stdexec::__declval<__next_t>()));
      using __end_sender_t = decltype(__schedule_at(__self, __marble, stdexec::just()));
      struct __next_sender_id {
        using __t [[maybe_unused]] = __next_sender_t;
      };
      struct __end_sender_id {
        using __t [[maybe_unused]] = __end_sender_t;
      };

      // WORKAROUND clang 19 would fail to compile the construction of the variant_sender.
      // It was unable to find the matching value in the variant that would be constructed.
      // __proxy_sender is a hammer to force the types to look different enough to
      // distinguish which variant value to construct
      using __next_sender_proxy_t = __proxy_sender<__next_sender_id>;
      using __end_sender_proxy_t = __proxy_sender<__end_sender_id>;

      using __result_t = variant_sender<__next_sender_proxy_t, __end_sender_proxy_t>;
      if (__marble.__notification_.has_value()) {

        auto __next = exec::set_next(__self.__receiver_, __marble.visit_sender());
        __next_sender_proxy_t __scheduled(
          __schedule_at(__self, __marble, static_cast<__next_t&&>(__next)));
        return __result_t{__scheduled};
      } else {
        return __result_t{__end_sender_proxy_t{{__schedule_at(__self, __marble, stdexec::just())}}};
      }
    }

    template <class _ReceiverId>
    void __test_sequence_operation<_ReceiverId>::start() noexcept {
      this->__on_stop_.emplace(
        stdexec::get_stop_token(stdexec::get_env(this->__receiver_)),
        typename __part_t::__stop_callback_fn_t(this));
      for (auto& __marble: this->__marbles_) {
        __marble.set_origin_frame(this->__context_->now());
        auto& __op = __marble_ops_.emplace_back();
        __op.__emplace_from([this, &__marble]() {
          return stdexec::connect(
            __part_t::__schedule_marble(*this, __marble), __receiver_t{&this->__receiver_});
        });
      }

      this->__active_ops_ = this->__marble_ops_.size();
      for (auto& __op: this->__marble_ops_) {
        stdexec::start(__op.value());
      }
    }

    struct __test_sequence {
      using __t = __test_sequence;
      using __id = __test_sequence;
      using sender_concept = exec::sequence_sender_t;

      using __marble_t = marble_t<test_clock>;
      using __marble_sender_t = __marble_t::__marble_sender_t;

      test_context* __context_;
      std::vector<__marble_t> __marbles_;

      template <stdexec::__decays_to<__test_sequence> _Self, class... _Env>
      static auto get_item_types(_Self&&, _Env&&...) noexcept -> item_types<__marble_sender_t> {
        return {};
      }

      template <stdexec::__decays_to<__test_sequence> _Self, class... _Env>
      static auto
        get_completion_signatures(_Self&&, _Env&&...) noexcept -> stdexec::completion_signatures<
          stdexec::set_value_t(),
          stdexec::set_error_t(std::error_code),
          stdexec::set_error_t(std::exception_ptr),
          stdexec::set_stopped_t()
        > {
        return {};
      }

      static constexpr auto subscribe =
        []<stdexec::__decays_to<__test_sequence> _Sequence, stdexec::receiver _Receiver>(
          _Sequence&& __sequence,
          _Receiver __receiver) noexcept -> __test_sequence_operation<stdexec::__id<_Receiver>> {
        return {
          static_cast<_Sequence&&>(__sequence).__marbles_,
          static_cast<_Sequence&&>(__sequence).__context_,
          static_cast<_Receiver&&>(__receiver)};
      };
    };
  } // namespace _tst_sched

  template <class _Sequence>
  inline auto test_context::get_marbles_from(
    _Sequence&& __sequence,
    typename test_clock::duration __stop_after) noexcept -> std::vector<marble_t<test_clock>> {

    std::vector<marble_t<test_clock>> __recording;
    stdexec::inplace_stop_source __source;
    auto __clock = get_clock();

    auto __op = stdexec::connect(
      stdexec::when_all(
        // record the sequence
        exec::sequence(
          // schedule connect and start of the sequence being recorded
          // on the test scheduler
          exec::schedule_at(get_scheduler(), __clock.now()),
          record_marbles(&__recording, __clock, static_cast<_Sequence&&>(__sequence))
          // always complete with set_stopped to prevent the following
          // scheduled request_stop from affecting the clock
          ,
          stdexec::just_stopped())
        // this is used to stop a 'never' sequence
        ,
        exec::schedule_at(get_scheduler(), __clock.now() + __stop_after)
          | stdexec::then([&__source]() noexcept { __source.request_stop(); })),
      _tst_sched::__recording_receiver{this, &__source});
    stdexec::start(__op);

    // dispatch the test context queues
    run();

    return __recording;
  }

  inline auto
    test_context::get_marble_sequence_from(std::vector<marble_t<test_clock>> __marbles) noexcept
    -> _tst_sched::__test_sequence {
    return {this, static_cast<std::vector<marble_t<test_clock>>&&>(__marbles)};
  }

  template <std::size_t _Len>
  inline auto test_context::get_marble_sequence_from(stdexec::__mstring<_Len> __diagram) noexcept
    -> _tst_sched::__test_sequence {
    return get_marble_sequence_from(get_marbles_from(__diagram));
  }
} // namespace exec
