/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include <cctype>
#include <exception>

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "../sequence_senders.hpp"

#include "./notification.hpp"
#include "exec/timed_scheduler.hpp"
#include "stdexec/__detail/__completion_signatures.hpp"
#include "stdexec/__detail/__config.hpp"
#include "stdexec/__detail/__execution_fwd.hpp"
#include "stdexec/__detail/__operation_states.hpp"
#include "stdexec/__detail/__sender_introspection.hpp"
#include "stdexec/__detail/__tuple.hpp"

#include <chrono>
#include <ios>
#include <span>
#include <type_traits>

namespace exec {
  namespace __marbles {
    using namespace stdexec;
    using namespace std::chrono_literals;

    //
    // marble_t<_Clock> represents a signal for a sequence
    // sender and the frame at which the signal occurs/occured
    //
    // a marble diagram is a string that is parsed into a
    // vector<marble_t<_Clock>>
    //
    // Example:
    //  this marble diagram
    //    "--a--b--c|"
    //  is equivalent to this set of marble_t
    //    marble_t{2ms, ex::set_value, 'a'},
    //    marble_t{5ms, ex::set_value, 'b'},
    //    marble_t{8ms, ex::set_value, 'c'},
    //    marble_t{8ms, sequence_end}
    //  which is displayed as
    //    set_value('a')@2ms, set_value('b')@5ms,
    //    set_value('c')@8ms, sequence_end()@8ms
    //
    // Diagram reference:
    // Time:
    //   ' ' indicates that 0ms has elapsed - used to line up diagrams in a visually pleasing manner
    //   '-' indicates that 1ms has elapsed
    //   ' 0-9+(ms|s|m) '
    //       indicates elapsed time at that point in the diagram that is equal to the specified number
    //       of ms - milliseconds,  s - seconds, m - minutes
    //       NOTE: must have a preceding and following space to disambiguate from values
    //   '(' begins a group of signals that all occur on the frame that the group begins on
    //   ')' ends a group of signals that all occur on the frame that the group begins on
    // Value:
    //   '0'-'9' 'a'-'z' 'A'-'Z'
    //       indicates that a value in the sequence completes with set_value( char )
    //   '#' indicates that a value in the sequence completes with set_error(error_code(interrupted))
    //   '.' indicates that a value in the sequence completes with set_stopped()
    // Sequence:
    //   '=' indicates connect() on the sequence sender
    //   '^' indicates start() on the sequence operation
    //   '|' indicates that the sequence completes with set_value()
    //   '$' indicates that the sequence completes with set_stopped()
    //   '?' indicates that request_stop() was sent to the sequence from an external source
    //
    //  record_marbles() will record a set of marbles from the signals of the specified sequence sender
    //


    // __value_t is a hammer to stop char from being treated like an integer

    struct __value_t {
      char __c_;

      operator char() noexcept {
        return __c_;
      }

      friend auto operator==(const __value_t& __lhs, const __value_t& __rhs) noexcept -> bool {
        return __lhs.__c_ == __rhs.__c_;
      }

      friend inline std::string to_string(const __value_t& __self) noexcept {
        const char __result[4] = {'\'', __self.__c_, '\'', '\0'};
        return __result;
      }
    };

    enum class marble_selector_t {
      uninitialized,
      frame_only,
      notification,
      sequence_start,
      sequence_connect,
      sequence_value,
      sequence_error,
      sequence_stopped,
      request_stop
    };

    struct sequence_start_t {
      operator marble_selector_t() const noexcept {
        return marble_selector_t::sequence_start;
      }
      friend inline std::string to_string(sequence_start_t) noexcept {
        return {"sequence_start"};
      }
    };
    static constexpr inline sequence_start_t sequence_start;

    struct sequence_connect_t {
      operator marble_selector_t() const noexcept {
        return marble_selector_t::sequence_connect;
      }
      friend inline std::string to_string(sequence_connect_t) noexcept {
        return {"sequence_connect"};
      }
    };
    static constexpr inline sequence_connect_t sequence_connect;

    struct sequence_end_t {
      operator marble_selector_t() const noexcept {
        return marble_selector_t::sequence_value;
      }
      friend inline std::string to_string(sequence_end_t) noexcept {
        return {"sequence_end"};
      }
    };
    static constexpr inline sequence_end_t sequence_end;

    struct sequence_error_t {
      operator marble_selector_t() const noexcept {
        return marble_selector_t::sequence_error;
      }
      friend inline std::string to_string(sequence_error_t) noexcept {
        return {"sequence_error"};
      }
    };
    static constexpr inline sequence_error_t sequence_error;

    struct sequence_stopped_t {
      operator marble_selector_t() const noexcept {
        return marble_selector_t::sequence_stopped;
      }
      friend inline std::string to_string(sequence_stopped_t) noexcept {
        return {"sequence_stopped"};
      }
    };
    static constexpr inline sequence_stopped_t sequence_stopped;

    struct request_stop_t {
      operator marble_selector_t() const noexcept {
        return marble_selector_t::request_stop;
      }
      friend inline std::string to_string(request_stop_t) noexcept {
        return {"request_stop"};
      }
    };
    static constexpr inline request_stop_t request_stop;

    using __completion_signatures_t = completion_signatures<
      set_value_t(__value_t),
      set_error_t(std::error_code),
      set_error_t(std::exception_ptr),
      set_stopped_t()
    >;

    template <class _Clock>
    struct marble_t {
      using __frame_t = typename _Clock::time_point;
      using __duration_t = typename _Clock::duration;
      using __notification_t = notification_t<__completion_signatures_t>;
      using __marble_sender_t = typename __notification_t::__notification_sender_t;

      using selector = marble_selector_t;

      __frame_t __at_;
      selector __selector_;
      std::optional<__notification_t> __notification_;

      marble_t(__frame_t __at, set_error_t __tag, std::error_code __error) noexcept
        : __at_{__at}
        , __selector_{selector::notification}
        , __notification_{} {
        __notification_.emplace(__tag, __error);
      }
      marble_t(__frame_t __at, set_error_t __tag, std::exception_ptr __ex) noexcept
        : __at_{__at}
        , __selector_{selector::notification}
        , __notification_{} {
        __notification_.emplace(__tag, __ex);
      }
      marble_t(__frame_t __at, set_value_t __tag, char __c) noexcept
        : __at_{__at}
        , __selector_{selector::notification}
        , __notification_{} {
        __notification_.emplace(__tag, __value_t{__c});
      }
      marble_t(__frame_t __at, set_value_t __tag, __value_t __v) noexcept
        : __at_{__at}
        , __selector_{selector::notification}
        , __notification_{} {
        __notification_.emplace(__tag, __v);
      }
      marble_t(__frame_t __at, set_stopped_t __tag) noexcept
        : __at_{__at}
        , __selector_{selector::notification}
        , __notification_{} {
        __notification_.emplace(__tag);
      }
      marble_t(__frame_t __at, sequence_start_t __tag) noexcept
        : __at_{__at}
        , __selector_{__tag}
        , __notification_{} {
      }
      marble_t(__frame_t __at, sequence_connect_t __tag) noexcept
        : __at_{__at}
        , __selector_{__tag}
        , __notification_{} {
      }
      marble_t(__frame_t __at, sequence_end_t __tag) noexcept
        : __at_{__at}
        , __selector_{__tag}
        , __notification_{} {
      }
      marble_t(__frame_t __at, sequence_error_t __tag) noexcept
        : __at_{__at}
        , __selector_{__tag}
        , __notification_{} {
      }
      marble_t(__frame_t __at, sequence_stopped_t __tag) noexcept
        : __at_{__at}
        , __selector_{__tag}
        , __notification_{} {
      }
      marble_t(__frame_t __at, request_stop_t __tag) noexcept
        : __at_{__at}
        , __selector_{__tag}
        , __notification_{} {
      }
      marble_t(__frame_t __at) noexcept
        : __at_{__at}
        , __selector_{selector::frame_only}
        , __notification_{} {
      }

      template <class _Fn>
      void visit(_Fn&& __fn) noexcept {
        if (__notification_.has_value()) {
          __notification_->visit(static_cast<_Fn&&>(__fn));
        }
      }

      template <class _Receiver>
      void visit_receiver(_Receiver&& __receiver) noexcept {
        if (__selector_ == selector::notification) {
          __notification_->visit_receiver(static_cast<_Receiver&&>(__receiver));
        } else {
          stdexec::set_stopped(static_cast<_Receiver&&>(__receiver));
        }
      }

      template <class _Receiver>
      void visit_sequence_receiver(_Receiver&& __receiver) noexcept {
        switch (__selector_) {
        case selector::sequence_value: {
          stdexec::set_value(static_cast<_Receiver&&>(__receiver));
          break;
        }
        case selector::sequence_error: {
          stdexec::set_error(static_cast<_Receiver&&>(__receiver), std::exception_ptr{});
          break;
        }
        case selector::notification: {
          if (value_notification()) {
            stdexec::set_value(static_cast<_Receiver&&>(__receiver));
            break;
          }
        }
          [[fallthrough]];
        case selector::request_stop:
          [[fallthrough]];
        case selector::sequence_stopped:
          [[fallthrough]];
        case selector::sequence_connect:
          [[fallthrough]];
        case selector::sequence_start:
          [[fallthrough]];
        case selector::frame_only:
          [[fallthrough]];
        default: {
          stdexec::set_stopped(static_cast<_Receiver&&>(__receiver));
          break;
        }
        };
      }

      [[nodiscard]]
      auto visit_sender() noexcept -> __marble_sender_t {
        return __notification_->visit_sender();
      }

      [[nodiscard]]
      bool sequence_end() const noexcept {
        return __selector_ == selector::sequence_value;
      }
      [[nodiscard]]
      bool sequence_error() const noexcept {
        return __selector_ == selector::sequence_error;
      }
      [[nodiscard]]
      bool sequence_stopped() const noexcept {
        return __selector_ == selector::sequence_stopped;
      }
      [[nodiscard]]
      bool request_stop() const noexcept {
        return __selector_ == selector::request_stop;
      }
      [[nodiscard]]
      bool value_notification() const noexcept {
        return __notification_.has_value() && __notification_->value();
      }
      [[nodiscard]]
      bool error_notification() const noexcept {
        return __notification_.has_value() && __notification_->error();
      }
      [[nodiscard]]
      bool stopped_notification() const noexcept {
        return __notification_.has_value() && __notification_->stopped();
      }
      [[nodiscard]]
      __frame_t frame() const noexcept {
        return __at_;
      }
      __frame_t shift_frame_by(__duration_t __by) noexcept {
        __frame_t __old_frame = __at_;
        __at_ += __by;
        return __old_frame;
      }
      __frame_t set_origin_frame(__frame_t __origin) noexcept {
        __frame_t __old_frame = __at_;
        __at_ += __origin.time_since_epoch();
        return __old_frame;
      }

      friend auto operator==(const marble_t& __lhs, const marble_t& __rhs) noexcept -> bool {
        return std::chrono::duration_cast<std::chrono::milliseconds>(__lhs.__at_.time_since_epoch())
              == std::chrono::duration_cast<std::chrono::milliseconds>(__rhs.__at_
                                                                         .time_since_epoch())
            && __lhs.__selector_ == __rhs.__selector_
            && __lhs.__notification_.has_value() == __rhs.__notification_.has_value()
            && (__lhs.__notification_.has_value() && __rhs.__notification_.has_value()
                  ? __lhs.__notification_.value() == __rhs.__notification_.value()
                  : true);
      }

      friend std::string to_string(const marble_t& __self) noexcept {
        using std::to_string;
        std::string __result;
        switch (__self.__selector_) {
        case selector::frame_only: {
          __result = "frame";
          break;
        }
        case selector::notification: {
          __result = to_string(__self.__notification_.value());
          break;
        }
        case selector::request_stop: {
          __result = to_string(__marbles::request_stop) + "()";
          break;
        }
        case selector::sequence_start: {
          __result = to_string(__marbles::sequence_start) + "()";
          break;
        }
        case selector::sequence_connect: {
          __result = to_string(__marbles::sequence_connect) + "()";
          break;
        }
        case selector::sequence_value: {
          __result = to_string(__marbles::sequence_end) + "()";
          break;
        }
        case selector::sequence_error: {
          __result = to_string(__marbles::sequence_error) + "()";
          break;
        }
        case selector::sequence_stopped: {
          __result = to_string(__marbles::sequence_stopped) + "()";
          break;
        }
        default: {
          return {"uninitialized-marble"};
        }
        };
        return __result + "@"
             + to_string(
                 std::chrono::duration_cast<std::chrono::milliseconds>(__self.__at_
                                                                         .time_since_epoch())
                   .count())
             + "ms";
      }
    };

    struct get_marbles_from_t {

      template <class _Clock, std::size_t _Len>
      constexpr auto operator()(_Clock __clock, __mstring<_Len> __diagram) const noexcept
        -> std::vector<marble_t<_Clock>> {
        using __frame_t = typename _Clock::time_point;
        using __duration_t = typename _Clock::duration;

        constexpr auto __make_span =
          []<std::size_t _LenB>(const __mstring<_LenB>& __string) noexcept {
            return std::span<const char>{__string.__what_, _LenB - 1};
          };

        std::vector<marble_t<_Clock>> __marbles;
        __frame_t __group_start_frame{-1ms};
        __frame_t __frame = __clock.now();
        auto __whole = __make_span(__diagram);
        auto __remaining = __whole;
        auto __consume_first = [&__remaining](std::size_t __skip) noexcept {
          __remaining = __remaining.subspan(__skip);
        };
        auto __push = [&](auto __tag, auto... __args) noexcept {
          __marbles.emplace_back(
            __group_start_frame == __frame_t{-1ms} ? __frame : __group_start_frame,
            __tag,
            __args...);
        };
        while (!__remaining.empty()) {
          __frame_t __next_frame{__frame};
          auto __advance_frame_by = [&__next_frame,
                                     &__group_start_frame](__duration_t __by) noexcept {
            __next_frame += __group_start_frame == __frame_t{-1ms} ? __by : 0ms;
          };
          switch (__remaining.front()) {
          case '-': {
            __advance_frame_by(1ms);
            __consume_first(1);
            break;
          }
          case '(': {
            __group_start_frame = __frame;
            __consume_first(1);
            break;
          }
          case ')': {
            __group_start_frame = __frame_t{-1ms};
            __advance_frame_by(1ms);
            __consume_first(1);
            break;
          }
          case '|': {
            __push(sequence_end);
            __consume_first(1);
            break;
          }
          case '=': {
            __push(sequence_connect);
            __consume_first(1);
            break;
          }
          case '^': {
            __push(sequence_start);
            __consume_first(1);
            break;
          }
          case '$': {
            __push(sequence_stopped);
            __consume_first(1);
            break;
          }
          case '?': {
            __push(request_stop);
            __consume_first(1);
            break;
          }
          case '#': {
            __push(set_error, std::make_error_code(std::errc::interrupted));
            __consume_first(1);
            break;
          }
          case '.': {
            __push(set_stopped);
            __consume_first(1);
            break;
          }
          default: {
            // use auto and math to derive the difference type
            auto __consumed_in_default = __remaining.begin() - __remaining.begin();
            if (
              std::addressof(*__whole.begin()) == std::addressof(*__remaining.begin())
              || !!std::isspace(__remaining.front())) {
              if (!!std::isspace(__remaining.front())) {
                __consume_first(1);
                ++__consumed_in_default;
              }
              // try to consume a duration at first char or after ' ' char.
              if (!!std::isdigit(__remaining.front())) {
                auto __valid_duration_suffix = [](auto c) noexcept {
                  return c == 'm' || c == 's';
                };
                auto __suffix_begin = std::ranges::find_if(__remaining, __valid_duration_suffix);
                bool __all_digits = std::all_of(__remaining.begin(), __suffix_begin, [](auto c) {
                  return std::isdigit(c);
                });
                if (
                  __suffix_begin != __remaining.end() && __suffix_begin - __remaining.begin() > 0
                  && __all_digits) {
                  auto __to_consume = __suffix_begin - __remaining.begin();
                  long __duration = std::atol(__remaining.data());
                  const auto __ms_str = "ms "_mstr;
                  const auto __ms = __make_span(__ms_str);
                  const auto __s_str = "s "_mstr;
                  const auto __s = __make_span(__s_str);
                  const auto __m_str = "m "_mstr;
                  const auto __m = __make_span(__m_str);
                  if (std::ranges::equal(__remaining.subspan(__to_consume, 3), __ms)) {
                    __to_consume += 2;
                  } else if (std::ranges::equal(__remaining.subspan(__to_consume, 2), __s)) {
                    __duration *= 1000;
                    __to_consume += 1;
                  } else if (std::ranges::equal(__remaining.subspan(__to_consume, 2), __m)) {
                    __duration = __duration * 1000 * 60;
                    __to_consume += 1;
                  } else {
                    __duration = -1;
                    __to_consume = 0;
                    //fallthrough
                  }
                  if (__duration >= 0 && __to_consume > 0) {
                    __advance_frame_by(std::chrono::milliseconds(__duration));
                    __consume_first(__to_consume);
                    __consumed_in_default += __to_consume;
                    break;
                  }
                }
              }
            }
            if (!!std::isalnum(__remaining.front())) {
              __advance_frame_by(1ms);
              __push(set_value, __remaining.front());
              __consume_first(1);
              ++__consumed_in_default;
              break;
            }
            if (__consumed_in_default == 0) {
              // parsing error
              return __marbles;
            }
            break;
          }
          };
          __frame = __next_frame;
        }
        return __marbles;
      }
    };

    template <class _Receiver, class _Clock>
    struct __value_receiver {
      using __t = __value_receiver;
      using __id = __value_receiver;
      using receiver_concept = stdexec::receiver_t;

      _Clock __clock_;
      std::vector<marble_t<_Clock>>* __recording_;
      _Receiver* __receiver_;

      template <class... _Args>
      void set_value(_Args&&... __args) noexcept {
        __recording_
          ->emplace_back(__clock_.now(), stdexec::set_value, static_cast<_Args&&>(__args)...);
        stdexec::set_value(static_cast<_Receiver>(*__receiver_));
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        __recording_
          ->emplace_back(__clock_.now(), stdexec::set_error, static_cast<_Error&&>(__error));
        stdexec::set_stopped(static_cast<_Receiver>(*__receiver_));
      }

      void set_stopped() noexcept {
        __recording_->emplace_back(__clock_.now(), stdexec::set_stopped);
        stdexec::set_stopped(static_cast<_Receiver>(*__receiver_));
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return stdexec::get_env(*__receiver_);
      }
    };

    template <class _Value, class _ReceiverId, class _Clock>
    struct __value_operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __value_operation;

        using __receiver_t = __value_receiver<_Receiver, _Clock>;

        _Receiver __receiver_;
        _Clock __clock_;
        std::vector<marble_t<_Clock>>* __recording_;
        stdexec::connect_result_t<_Value, __receiver_t> __op_;

        __t(
          _Value&& __value,
          _Receiver&& __receiver,
          _Clock __clock,
          std::vector<marble_t<_Clock>>* __recording) noexcept
          : __receiver_{static_cast<_Receiver&&>(__receiver)}
          , __clock_{__clock}
          , __recording_(__recording)
          , __op_{stdexec::connect(
              static_cast<_Value&&>(__value),
              __receiver_t{__clock_, __recording_, &__receiver_})} {
        }

        void start() & noexcept {
          stdexec::start(__op_);
        }
      };
    };

    template <class _Value, class _Clock>
    struct __value_sender {
      struct __t {
        using __id = __value_sender;
        using sender_concept = stdexec::sender_t;

        template <class _Receiver>
        using __value_operation_t =
          stdexec::__t<__value_operation<_Value, stdexec::__id<_Receiver>, _Clock>>;

        _Clock __clock_;
        std::vector<marble_t<_Clock>>* __recording_;
        _Value __value_;

        template <std::same_as<__t> _Self, class... _Env>
        static auto get_completion_signatures(_Self&&, _Env&&...) noexcept
          -> stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()> {
          return {};
        }

        template <std::same_as<__t> _Self, receiver _Receiver>
        static auto connect(_Self&& __self, _Receiver&& __rcvr)
          noexcept(__nothrow_move_constructible<_Receiver>) {
          return __value_operation_t<_Receiver>{
            static_cast<_Value&&>(__self.__value_),
            static_cast<_Receiver&&>(__rcvr),
            __self.__clock_,
            __self.__recording_};
        }
      };
    };

    template <class _Receiver, class _Clock>
    struct __receiver {
      using __t = __receiver;
      using __id = __receiver;
      using receiver_concept = stdexec::receiver_t;

      template <class _Value>
      using __value_sender_t = stdexec::__t<__value_sender<_Value, _Clock>>;

      _Clock __clock_;
      std::vector<marble_t<_Clock>>* __recording_;
      _Receiver* __receiver_;

      using __receiver_t = __value_receiver<_Receiver, _Clock>;

      template <stdexec::sender _Value>
      auto set_next(_Value&& __value) noexcept -> next_sender auto {
        return __value_sender_t<_Value>{__clock_, __recording_, static_cast<_Value&&>(__value)};
      }

      void set_value() noexcept {
        __recording_->emplace_back(__clock_.now(), sequence_end);
        stdexec::set_value(static_cast<_Receiver&&>(*__receiver_));
      }

      template <class _Error>
      void set_error(_Error&&) noexcept {
        __recording_->emplace_back(__clock_.now(), sequence_error);
        stdexec::set_value(static_cast<_Receiver&&>(*__receiver_));
      }

      void set_stopped() noexcept {
        __recording_->emplace_back(__clock_.now(), sequence_stopped);
        stdexec::set_value(static_cast<_Receiver&&>(*__receiver_));
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return stdexec::get_env(*__receiver_);
      }
    };

    template <class _Sequence, class _ReceiverId, class _Clock>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __operation;

        using __receiver_t = __receiver<_Receiver, _Clock>;

        _Receiver __receiver_;
        _Clock __clock_;
        std::vector<marble_t<_Clock>>* __recording_;
        exec::subscribe_result_t<_Sequence, __receiver_t> __op_;

        __t(
          _Sequence&& __sequence,
          _Receiver&& __receiver,
          _Clock __clock,
          std::vector<marble_t<_Clock>>* __recording) noexcept
          : __receiver_{static_cast<_Receiver&&>(__receiver)}
          , __clock_{__clock}
          , __recording_(__recording)
          , __op_{exec::subscribe(
              static_cast<_Sequence&&>(__sequence),
              __receiver_t{__clock_, __recording_, &__receiver_})} {
        }

        void start() & noexcept {
          __recording_->emplace_back(__clock_.now(), sequence_start);
          stdexec::start(__op_);
        }
      };
    };

    template <class _Clock, class _Receiver>
    struct __connect_fn {
      _Receiver __rcvr_;

      using __receiver_id_t = __id<_Receiver>;

      template <class _Child>
      using __operation_t = __t<__operation<_Child, __receiver_id_t, _Clock>>;

      template <class _Data, class _Child>
      auto operator()(__ignore, _Data&& __data, _Child&& __child)
        noexcept(__nothrow_constructible_from<
                 __operation_t<_Child>,
                 _Child,
                 _Receiver,
                 _Clock,
                 std::vector<marble_t<_Clock>>*
        >) -> __operation_t<_Child> {
        auto [__recording, __clock] = static_cast<_Data&&>(__data);
        __recording->emplace_back(__clock.now(), sequence_connect);
        return {
          static_cast<_Child&&>(__child), static_cast<_Receiver&&>(__rcvr_), __clock, __recording};
      }
    };

    struct record_marbles_t {
      template <class _Sequence, class _Clock>
      auto operator()(
        std::vector<marble_t<_Clock>>* __recording,
        _Clock __clock,
        _Sequence&& __sequence) const { //-> __well_formed_sender auto {
        auto __domain = __get_early_domain(static_cast<_Sequence&&>(__sequence));
        return transform_sender(
          __domain,
          __make_sexpr<record_marbles_t>(
            __decayed_tuple<std::vector<marble_t<_Clock>>*, _Clock>{__recording, __clock},
            static_cast<_Sequence&&>(__sequence)));
      }
      template <class _Sequence, class _Clock>
      std::vector<marble_t<_Clock>>
        operator()(_Clock __clock, _Sequence&& __sequence) const noexcept {
        std::vector<marble_t<_Clock>> __recording;
        auto __recorder = (*this)(&__recording, __clock, static_cast<_Sequence&&>(__sequence));
        stdexec::sync_wait(__recorder);
        return __recording;
      }
    };

    struct __record_marbles_impl : __sexpr_defaults {

      template <class _Self>
      using __clock_of_t = __mapply<__q<__mback>, __data_of<STDEXEC_REMOVE_REFERENCE(_Self)>>;

      static constexpr auto get_completion_signatures =
        []<sender_expr_for<record_marbles_t> _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept
        -> completion_signatures<set_value_t()> {
        return {};
      };

      static constexpr auto connect =
        []<sender_expr_for<record_marbles_t> _Self, receiver _Receiver>(_Self&& __self, _Receiver&& __rcvr) noexcept(
          __nothrow_callable<__sexpr_apply_t, _Self, __connect_fn<__clock_of_t<_Self>, _Receiver>>)
        -> __call_result_t<__sexpr_apply_t, _Self, __connect_fn<__clock_of_t<_Self>, _Receiver>> {
        return __sexpr_apply(
          static_cast<_Self&&>(__self),
          __connect_fn<__clock_of_t<_Self>, _Receiver>{static_cast<_Receiver&&>(__rcvr)});
      };
    };

  } // namespace __marbles

  using __value_t = __marbles::__value_t;

  using sequence_start_t = __marbles::sequence_start_t;
  static constexpr inline auto sequence_start = sequence_start_t{};

  using sequence_connect_t = __marbles::sequence_connect_t;
  static constexpr inline auto sequence_connect = sequence_connect_t{};

  using sequence_end_t = __marbles::sequence_end_t;
  static constexpr inline auto sequence_end = sequence_end_t{};

  using sequence_error_t = __marbles::sequence_error_t;
  static constexpr inline auto sequence_error = sequence_error_t{};

  using sequence_stopped_t = __marbles::sequence_stopped_t;
  static constexpr inline auto sequence_stopped = sequence_stopped_t{};

  using request_stop_t = __marbles::request_stop_t;
  static constexpr inline auto request_stop = request_stop_t{};

  template <class _Clock>
  using marble_t = __marbles::marble_t<_Clock>;

  using get_marbles_from_t = __marbles::get_marbles_from_t;

  static constexpr inline auto get_marbles_from = get_marbles_from_t{};

  using record_marbles_t = __marbles::record_marbles_t;

  static constexpr inline auto record_marbles = record_marbles_t{};

} // namespace exec

namespace stdexec {
  template <>
  struct __sexpr_impl<exec::record_marbles_t> : exec::__marbles::__record_marbles_impl { };
} // namespace stdexec
