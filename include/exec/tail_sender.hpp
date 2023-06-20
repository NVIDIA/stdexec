/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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
#include "env.hpp"

namespace exec {
  using namespace stdexec;

  template <class _Ty>
  concept __contextually_convertible_to_bool = requires(const _Ty __val) {
    { (static_cast<const _Ty&&>(__val) ? false : false) } -> same_as<bool>;
  };

  template <class _Ty>
  static constexpr bool __nothrow_contextually_convertible_to_bool_v =
    noexcept((std::declval<const _Ty&&>() ? (void) 0 : (void) 0));

  template <class _Ty>
  concept __nothrow_contextually_convertible_to_bool =
    __contextually_convertible_to_bool<_Ty> && __nothrow_contextually_convertible_to_bool_v<_Ty>;

  namespace __unwind {
    struct unwind_t {
      template <class _Op>
        requires std::tag_invocable<unwind_t, _Op&>
      void operator()(_Op& __op) const noexcept(std::nothrow_tag_invocable<unwind_t, _Op&>) {
        (void) tag_invoke(unwind_t{}, __op);
      }
    };
  }

  using __unwind::unwind_t;
  inline constexpr unwind_t unwind{};

  namespace __sender_queries {
    template <class _Ty>
    const _Ty& __cref_fn(const _Ty&);
    template <class _Ty>
    using __cref_t = decltype(__sender_queries::__cref_fn(__declval<_Ty>()));

    struct always_completes_inline_t {
      template <class _Sender, class _Env = no_env>
        requires std::tag_invocable<always_completes_inline_t, __cref_t<_Sender>, __cref_t<_Env>>
      constexpr bool operator()(_Sender&& __sndr, _Env&& __env) const noexcept {
        static_assert(
          same_as<
            bool,
            tag_invoke_result_t<always_completes_inline_t, __cref_t<_Sender>, __cref_t<_Env>>>);
        static_assert(
          std::nothrow_tag_invocable<always_completes_inline_t, __cref_t<_Sender>, __cref_t<_Env>>);
        return tag_invoke(always_completes_inline_t{}, std::as_const(__sndr), std::as_const(__env));
      }

      template <class _Sender, class _Env = no_env>
        requires std::tag_invocable<
          always_completes_inline_t,
          __mtype<std::decay_t<_Sender>>,
          __mtype<std::decay_t<_Env>>>
      constexpr bool operator()(
        __mtype<std::decay_t<_Sender>> __sndr,
        __mtype<std::decay_t<_Env>> __env) const noexcept {
        static_assert(
          same_as<
            bool,
            tag_invoke_result_t<
              always_completes_inline_t,
              __mtype<std::decay_t<_Sender>>,
              __mtype<std::decay_t<_Env>>>>);
        static_assert(std::nothrow_tag_invocable<
                      always_completes_inline_t,
                      __mtype<std::decay_t<_Sender>>,
                      __mtype<std::decay_t<_Env>>>);
        return tag_invoke(always_completes_inline_t{}, __sndr, __env);
      }

      constexpr bool operator()(auto&&, auto&&) const noexcept {
        return false;
      }
    };
  } // namespace __sender_queries

  using __sender_queries::always_completes_inline_t;
  inline constexpr always_completes_inline_t always_completes_inline{};

  template <class _TailOperationState>
  concept tail_operation_state =
    operation_state<_TailOperationState> && std::is_nothrow_destructible_v<_TailOperationState>
    && std::is_trivially_destructible_v<_TailOperationState>
    && (!std::is_copy_constructible_v<_TailOperationState>) //
    &&(!std::is_move_constructible_v< _TailOperationState>) //
    &&(!std::is_copy_assignable_v< _TailOperationState>)    //
    &&(!std::is_move_assignable_v< _TailOperationState>)    //
    &&requires(_TailOperationState& __op) {
      { unwind(__op) } noexcept;
    };

  template <class _Sender, class _Env = no_env>
  inline constexpr bool always_completes_inline_v =
    always_completes_inline(__mtype<std::decay_t<_Sender>>{}, __mtype<std::decay_t<_Env>>{});

  template <class _TailSender, class _Env = no_env>
  concept tail_sender =
    sender<_TailSender, _Env> && same_as<__single_sender_value_t<_TailSender, _Env>, void>
    && always_completes_inline_v<_TailSender, _Env>
    && std::is_nothrow_move_constructible_v<_TailSender>
    && std::is_nothrow_destructible_v<_TailSender> && std::is_trivially_destructible_v<_TailSender>;

  template <class _TailReceiver>
  concept tail_receiver =
    receiver<_TailReceiver> && std::is_nothrow_copy_constructible_v<_TailReceiver>
    && std::is_nothrow_move_constructible_v<_TailReceiver>
    && std::is_nothrow_copy_assignable_v<_TailReceiver>
    && std::is_nothrow_move_assignable_v<_TailReceiver>
    && std::is_nothrow_destructible_v<_TailReceiver>
    && std::is_trivially_destructible_v<_TailReceiver>;

  struct __null_tail_receiver {
    friend void tag_invoke(set_value_t, __null_tail_receiver&&, auto&&...) noexcept {
    }

    friend void tag_invoke(set_stopped_t, __null_tail_receiver&&) noexcept {
    }

    friend __empty_env tag_invoke(get_env_t, const __null_tail_receiver& __self) {
      return {};
    }
  };

  struct __null_tail_sender {
    struct __operation : __immovable {
      __operation() = default;

      // this is a nullable_tail_sender that always returns false to prevent
      // callers from calling start() and unwind()
      inline constexpr operator bool() const noexcept {
        return false;
      }

      friend void tag_invoke(start_t, __operation& self) noexcept {
        printf("__null_tail_sender start\n");
        fflush(stdout);
        std::terminate();
      }

      friend void tag_invoke(unwind_t, __operation& self) noexcept {
        printf("__null_tail_sender unwind\n");
        fflush(stdout);
        std::terminate();
      }
    };

    using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

    template <class _TailReceiver>
    friend auto tag_invoke(connect_t, __null_tail_sender&&, _TailReceiver&&) noexcept -> op {
      return {};
    }

    template <class _Env>
    friend constexpr bool tag_invoke(
      exec::always_completes_inline_t,
      exec::__mtype<__null_tail_sender>,
      exec::__mtype<_Env>) noexcept {
      return true;
    }
  };

  template <typename... _TailSenderN>
  struct __variant_tail_sender;

  template <tail_operation_state _TailOperationState>
  using __next_tail_from_operation_t = __call_result_t<start_t, _TailOperationState&>;

  template <tail_operation_state _TailOperationState>
  using next_tail_from_operation_t = __if<
    __bool<std::is_void_v<__call_result_t<start_t, _TailOperationState&>>>,
    __null_tail_sender,
    __call_result_t<start_t, _TailOperationState&>>;

  template <tail_sender _TailSender, tail_receiver _TailReceiver>
  using __next_tail_from_sender_to_t =
    __next_tail_from_operation_t<stdexec::connect_result_t<_TailSender, _TailReceiver>>;

  template <tail_sender _TailSender, tail_receiver _TailReceiver>
  using next_tail_from_sender_to_t =
    next_tail_from_operation_t<stdexec::connect_result_t<_TailSender, _TailReceiver>>;

  template <class _TailSender, class _TailReceiver>
  concept tail_sender_to =                                                    //
    tail_sender<_TailSender>                                                  //
    && tail_receiver<_TailReceiver>                                           //
    && requires(_TailSender&& __sndr, _TailReceiver&& __rcvr) {               //
         {                                                                    //
           stdexec::connect((_TailSender&&) __sndr, (_TailReceiver&&) __rcvr) //
         } noexcept -> tail_operation_state;                                  //
       }                                                                      //
    && tail_sender<next_tail_from_sender_to_t<_TailSender, _TailReceiver>>;

  template <class _TailOperationState>
  concept __terminal_tail_operation_state =
    tail_operation_state<_TailOperationState>
    && same_as<__next_tail_from_operation_t<_TailOperationState>, void>;

  template <class _TailSender, class _TailReceiver>
  concept __terminal_tail_sender_to =
    tail_sender_to<_TailSender, _TailReceiver>
    && same_as<__next_tail_from_sender_to_t<_TailSender, _TailReceiver>, void>;

  template <class _TailSender, class _TailReceiver, class... _ValidTailSender>
  concept __recursive_tail_sender_to =
    tail_sender_to<_TailSender, _TailReceiver>
    && tail_operation_state<connect_result_t<_TailSender, _TailReceiver>>
    && __one_of<next_tail_from_sender_to_t<_TailSender, _TailReceiver>, _ValidTailSender...>;

  template <class _TailOperationState>
  concept __nullable_tail_operation_state =
    tail_operation_state<_TailOperationState>
    && __nothrow_contextually_convertible_to_bool<_TailOperationState>;

  template <class _TailSender, class _TailReceiver>
  concept __nullable_tail_sender_to =
    tail_sender_to<_TailSender, _TailReceiver>
    && __nullable_tail_operation_state<connect_result_t<_TailSender, _TailReceiver>>;

} // namespace exec

#include "variant_tail_sender.hpp"

namespace exec {

  template <tail_sender _To, tail_sender _From>
  constexpr std::decay_t<_To> result_from(_From&& __f) noexcept {
    if constexpr (
      __is_instance_of<std::decay_t<_From>, __variant_tail_sender>
      && __is_instance_of<std::decay_t<_To>, __variant_tail_sender>) {
      return variant_cast<std::decay_t<_To>>((_From&&) __f);
    } else if constexpr (
      __is_instance_of<std::decay_t<_From>, __variant_tail_sender>
      && !__is_instance_of<std::decay_t<_To>, __variant_tail_sender>) {
      return get<std::decay_t<_To>>((_From&&) __f);
    } else {
      static_assert(std::is_constructible_v<_To, _From>, "result_from cannot convert");
      return (_From&&) __f;
    }
  }

  template <tail_sender _TailSender>
    requires(!same_as<__null_tail_sender, _TailSender>)
  struct maybe_tail_sender {
    maybe_tail_sender() noexcept = default;

    maybe_tail_sender(__null_tail_sender) noexcept {
    }

    maybe_tail_sender(_TailSender __t) noexcept
      : tail_sender_(__t) {
    }

    template <class _TailReceiver>
    struct op : __immovable {
      using op_t = connect_result_t<_TailSender, _TailReceiver>;
      op() = default;

      explicit op(_TailSender __t, _TailReceiver __rcvr)
        : op_(stdexec::__conv{[&] {
          return stdexec::connect(__t, __rcvr);
        }}) {
      }

      operator bool() const noexcept {
        if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
          return !!op_ && !!*op_;
        } else {
          return !!op_;
        }
      }

      [[nodiscard]] friend auto tag_invoke(start_t, op& __self) noexcept {
        if (!__self.op_) {
          printf("maybe_tail_sender start optional\n");
          fflush(stdout);
          std::terminate();
        }
        if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
          if (!*__self.op_) {
            printf("maybe_tail_sender start nullable\n");
            fflush(stdout);
            std::terminate();
          }
        }
        return stdexec::start(*__self.op_);
      }

      friend void tag_invoke(unwind_t, op& __self) noexcept {
        if (!__self.op_) {
          printf("maybe_tail_sender unwind optional\n");
          fflush(stdout);
          std::terminate();
        }
        if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
          if (!*__self.op_) {
            printf("maybe_tail_sender unwind nullable\n");
            fflush(stdout);
            std::terminate();
          }
        }
        exec::unwind(*__self.op_);
      }

      std::optional<op_t> op_;
    };

    using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

    template <class _TailReceiver>
    [[nodiscard]] friend auto
      tag_invoke(connect_t, maybe_tail_sender&& __self, _TailReceiver&& __rcvr) noexcept
      -> op<std::decay_t<_TailReceiver>> {
      if (!__self.tail_sender_) {
        return {};
      }
      return op<std::decay_t<_TailReceiver>>{*((maybe_tail_sender&&) __self).tail_sender_, __rcvr};
    }

    template <class _Env>
    friend constexpr bool tag_invoke(
      exec::always_completes_inline_t,
      exec::__mtype<maybe_tail_sender>,
      exec::__mtype<_Env>) noexcept {
      return true;
    }

   private:
    std::optional<_TailSender> tail_sender_;
  };

  template <tail_sender _TailSender, tail_receiver _TailReceiver = __null_tail_receiver>
  struct scoped_tail_sender {
    explicit scoped_tail_sender(_TailSender __t, _TailReceiver __rcvr = _TailReceiver{}) noexcept
      : t_(__t)
      , r_(__rcvr)
      , valid_(true) {
    }

    scoped_tail_sender(scoped_tail_sender&& other) noexcept
      : t_(other.s_)
      , r_(other.r_)
      , valid_(std::exchange(other.valid_, false)) {
    }

    ~scoped_tail_sender() {
      if (valid_) {
        auto op = stdexec::connect(t_, r_);
        if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
          if (!!op) {
            exec::unwind(op);
          }
        } else {
          exec::unwind(op);
        }
      }
    }

    _TailSender get() noexcept {
      return t_;
    }

    _TailSender release() noexcept {
      valid_ = false;
      return t_;
    }

   private:
    _TailSender t_;
    _TailReceiver r_;
    bool valid_;
  };

  struct __all_resumed_tail_sender {

    using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

    template <class _TailReceiver>
    friend auto tag_invoke(connect_t, __all_resumed_tail_sender&&, _TailReceiver&& __rcvr) noexcept
      -> __call_result_t<connect_t, __null_tail_sender, _TailReceiver> {
      return stdexec::connect(__null_tail_sender{}, __rcvr);
    }

    template <class _Env>
    friend constexpr bool tag_invoke(
      exec::always_completes_inline_t,
      exec::__mtype<__all_resumed_tail_sender>,
      exec::__mtype<_Env>) noexcept {
      return true;
    }
  };

  namespace __start_until_nullable_ {

    struct __start_until_nullable_t;

    template <tail_sender _TailSender, tail_receiver _TailReceiver>
    struct __start_until_nullable_result;

    template <class _TailSender, class _TailReceiver>
    using __start_until_nullable_result_t =
      typename __start_until_nullable_result<_TailSender, _TailReceiver>::type;

    template <tail_sender _TailSender, tail_receiver _TailReceiver>
    struct __start_until_nullable_result {
      using type = __if<
        __bool<__nullable_tail_sender_to<_TailSender, _TailReceiver>>,
        _TailSender,
        __if<
          __bool<__terminal_tail_sender_to<_TailSender, _TailReceiver>>,
          __all_resumed_tail_sender,
          __minvoke<
            __with_default<__q<__start_until_nullable_result_t>, __all_resumed_tail_sender>,
            __next_tail_from_sender_to_t<_TailSender, _TailReceiver>,
            _TailReceiver> > >;
    };

    struct __start_until_nullable_t {
      template <tail_sender _TailSender, tail_receiver _TailReceiver>
      auto operator()(_TailSender __t, _TailReceiver __rcvr) const noexcept
        -> __start_until_nullable_result_t<_TailSender, _TailReceiver> {
        if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
          return __t;
        } else if constexpr (__terminal_tail_sender_to<_TailSender, _TailReceiver>) {
          // restrict scope of op
          {
            auto op = stdexec::connect(std::move(__t), std::move(__rcvr));
            stdexec::start(op);
          }
          return __all_resumed_tail_sender{};
        } else {
          auto op = stdexec::connect(std::move(__t), __rcvr);
          return __start_until_nullable_t{}(stdexec::start(op), std::move(__rcvr));
        }
      }
    };

  } // namespace __start_until_nullable_

  using __start_until_nullable_::__start_until_nullable_t;
  inline constexpr __start_until_nullable_t __start_until_nullable{};

  template <
    tail_sender _NextTailSender,
    tail_sender _TailSender,
    tail_receiver _TailReceiver,
    tail_sender... _PrevTailSenders>
  auto __start_next(_NextTailSender __next, _TailReceiver __rcvr) noexcept;

  template <tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
  struct __start_sequential_result;

  template <tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
  using __start_sequential_result_t =
    typename __start_sequential_result<_TailSender, _TailReceiver>::type;

  template <tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
  struct __start_sequential_result {
    using next_t = next_tail_from_sender_to_t<_TailSender, _TailReceiver>;
    using start_next_t = __call_result_t<
      decltype(__start_next<next_t, _TailSender, _TailReceiver, _PrevTailSenders...>),
      next_t,
      _TailReceiver>;
    using type = __if<
      __bool<
        __nullable_tail_sender_to<_TailSender, _TailReceiver>
        && __terminal_tail_sender_to<_TailSender, _TailReceiver>>,
      _TailSender,
      __if< // elseif
        __bool<__nullable_tail_sender_to<_TailSender, _TailReceiver>>,
        variant_tail_sender<__all_resumed_tail_sender, start_next_t>,
        __if< // elseif
          __bool<!__terminal_tail_sender_to<_TailSender, _TailReceiver>>,
          start_next_t,
          __all_resumed_tail_sender // else
          > > >;
  };

  template <tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
  auto __start_sequential(_TailSender __sndr, _TailReceiver __rcvr) noexcept
    -> __start_sequential_result_t<_TailSender, _TailReceiver, _PrevTailSenders...>;

  template <
    tail_sender _NextTailSender,
    tail_sender _TailSender,
    tail_receiver _TailReceiver,
    tail_sender... _PrevTailSenders>
  auto __start_next(_NextTailSender __next, _TailReceiver __rcvr) noexcept {
    if constexpr (__one_of<_NextTailSender, _TailSender, _PrevTailSenders...>) {
      static_assert(
        (__nullable_tail_sender_to<_TailSender, _TailReceiver>
         || (__nullable_tail_sender_to<_PrevTailSenders, _TailReceiver> || ...)),
        "At least one tail_sender in a cycle must be nullable to avoid "
        "entering an infinite loop");
      return __start_until_nullable(__next, std::move(__rcvr));
    } else {
      return __start_sequential<_NextTailSender, _TailReceiver, _TailSender, _PrevTailSenders...>(
        __next, std::move(__rcvr));
    }
  }

  template <tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
  auto __start_sequential(_TailSender __sndr, _TailReceiver __rcvr) noexcept
    -> __start_sequential_result_t<_TailSender, _TailReceiver, _PrevTailSenders...> {
    using next_t = next_tail_from_sender_to_t<_TailSender, _TailReceiver>;
    using result_t = __start_sequential_result_t<_TailSender, _TailReceiver, _PrevTailSenders...>;

    if constexpr (
      __nullable_tail_sender_to<_TailSender, _TailReceiver>
      && __terminal_tail_sender_to<_TailSender, _TailReceiver>) {
      // halt the recursion
      return __sndr;
    } else if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
      // recurse if the nullable tail-sender is valid otherwise return
      // a nullable and terminal tail-sender
      auto op = stdexec::connect(std::move(__sndr), __rcvr);
      if (!op) {
        return __all_resumed_tail_sender{};
      }
      return result_from<result_t>(
        __start_next<next_t, _TailSender, _TailReceiver, _PrevTailSenders...>(
          stdexec::start(op), __rcvr));
    } else if constexpr (!__terminal_tail_sender_to<_TailSender, _TailReceiver>) {
      auto op = stdexec::connect(std::move(__sndr), __rcvr);
      return result_from<result_t>(
        __start_next<next_t, _TailSender, _TailReceiver, _PrevTailSenders...>(
          stdexec::start(op), __rcvr));
    } else {
      // run the terminal and not nullable tail-sender and return
      // a nullable and terminal tail-sender
      auto op = stdexec::connect(std::move(__sndr), __rcvr);
      stdexec::start(op);
      return __all_resumed_tail_sender{};
    }
  }

  template <tail_receiver _TailReceiver>
  inline __null_tail_sender resume_tail_senders_until_one_remaining(_TailReceiver&&) noexcept {
    return {};
  }

  template <tail_receiver _TailReceiver, tail_sender _TailSender>
  _TailSender
    resume_tail_senders_until_one_remaining(_TailReceiver&&, _TailSender __sndr) noexcept {
    return __sndr;
  }

  template <tail_receiver _TailReceiver, tail_sender... _TailSenders, std::size_t... _Is>
  auto _resume_tail_senders_until_one_remaining(
    _TailReceiver&& __rcvr,
    std::index_sequence<_Is...>,
    _TailSenders... __sndrs) noexcept {
    using result_type = variant_tail_sender<decltype(__start_sequential(
      __start_sequential(__sndrs, __rcvr), __rcvr))...>;
    result_type result;

    auto cs2_tuple = std::make_tuple(
      variant_tail_sender<
        __all_resumed_tail_sender,
        decltype(__start_sequential(__start_sequential(__sndrs, __rcvr), __rcvr))>{
        __start_sequential(__start_sequential(__sndrs, __rcvr), __rcvr)}...);
    while (true) {
      std::size_t remaining = sizeof...(__sndrs);
      ((remaining > 1 ? (
          !holds_alternative<__all_resumed_tail_sender>(std::get<_Is>(cs2_tuple))
            ? (void) (result = result_from<result_type>(
                        std::get<_Is>(cs2_tuple) = result_from<decltype(std::get<_Is>(cs2_tuple))>(
                          __start_sequential(std::get<_Is>(cs2_tuple), __rcvr))))
            : (void) --remaining)
                      : (void) (result = result_from<result_type>(std::get<_Is>(cs2_tuple)))),
       ...);

      if (remaining <= 1) {
        return result;
      }
    }
  }

  template <tail_receiver _TailReceiver, tail_sender... _TailSenders>
  auto resume_tail_senders_until_one_remaining(
    _TailReceiver&& __rcvr,
    _TailSenders... __sndrs) noexcept {
    return _resume_tail_senders_until_one_remaining(
      __rcvr, std::index_sequence_for<_TailSenders...>{}, __sndrs...);
  }

  template <tail_receiver _TailReceiver, tail_sender... _TailSenders>
  void resume_tail_senders(_TailReceiver&& __rcvr, _TailSenders... __sndrs) noexcept {
    auto __last_tail = _resume_tail_senders_until_one_remaining(
      __rcvr, std::index_sequence_for<_TailSenders...>{}, __sndrs...);
    for (;;) {
      auto __op = connect(__last_tail, __rcvr);
      if (!__op) {
        return;
      }
      if constexpr (__terminal_tail_sender_to<decltype(__last_tail), _TailReceiver>) {
        start(__last_tail);
        return;
      } else {
        __last_tail = __start_sequential(start(__last_tail), __rcvr);
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  namespace __loop {
    class run_loop;

    struct __task : __immovable {
      __task* __next_ = this;

      union {
        void (*__execute_)(__task*) noexcept;
        __task* __tail_;
      };

      void __execute() noexcept {
        (*__execute_)(this);
      }
    };

    template <class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __task {
        using __id = __operation;

        run_loop* __loop_;
        [[no_unique_address]] _Receiver __rcvr_;

        static void __execute_impl(__task* __p) noexcept {
          auto& __rcvr = ((__t*) __p)->__rcvr_;
          try {
            if (get_stop_token(get_env(__rcvr)).stop_requested()) {
              set_stopped((_Receiver&&) __rcvr);
            } else {
              set_value((_Receiver&&) __rcvr);
            }
          } catch (...) {
            set_error((_Receiver&&) __rcvr, std::current_exception());
          }
        }

        explicit __t(__task* __tail) noexcept
          : __task{.__tail_ = __tail} {
        }

        __t(__task* __next, run_loop* __loop, _Receiver __rcvr)
          : __task{{}, __next, {&__execute_impl}}
          , __loop_{__loop}
          , __rcvr_{(_Receiver&&) __rcvr} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__start_();
        }

        void __start_() noexcept;
      };
    };

    template <class _ReceiverId>
    struct __run_operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __run_operation;

        run_loop* __loop_;
        [[no_unique_address]] _Receiver __rcvr_;

        __t(run_loop* __loop, _Receiver __rcvr)
          : __loop_{__loop}
          , __rcvr_{(_Receiver&&) __rcvr} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__start_();
        }

        void __start_() noexcept;
      };
    };

    class run_loop {
      template <class... Ts>
      using __completion_signatures_ = completion_signatures<Ts...>;

      template <class>
      friend struct __operation;
     public:
      struct __scheduler {
        using __t = __scheduler;
        using __id = __scheduler;
        bool operator==(const __scheduler&) const noexcept = default;

       private:
        struct __schedule_task {
          using __t = __schedule_task;
          using __id = __schedule_task;
          using completion_signatures = __completion_signatures_<
            set_value_t(),
            set_error_t(std::exception_ptr),
            set_stopped_t()>;

         private:
          friend __scheduler;

          template <class _Receiver>
          using __operation = stdexec::__t<__operation<stdexec::__id<_Receiver>>>;

          template <class _Receiver>
          friend __operation<_Receiver>
            tag_invoke(connect_t, const __schedule_task& __self, _Receiver __rcvr) {
            return __self.__connect_((_Receiver&&) __rcvr);
          }

          template <class _Receiver>
          __operation<_Receiver> __connect_(_Receiver&& __rcvr) const {
            return {&__loop_->__head_, __loop_, (_Receiver&&) __rcvr};
          }

          template <class _CPO>
          friend __scheduler
            tag_invoke(get_completion_scheduler_t<_CPO>, const __schedule_task& __self) noexcept {
            return __scheduler{__self.__loop_};
          }

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop) {
          }

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        friend __schedule_task tag_invoke(schedule_t, const __scheduler& __self) noexcept {
          return __self.__schedule();
        }

        friend stdexec::forward_progress_guarantee
          tag_invoke(get_forward_progress_guarantee_t, const __scheduler&) noexcept {
          return stdexec::forward_progress_guarantee::parallel;
        }

        // BUGBUG NOT TO SPEC
        friend bool
          tag_invoke(this_thread::execute_may_block_caller_t, const __scheduler&) noexcept {
          return false;
        }

        __schedule_task __schedule() const noexcept {
          return __schedule_task{__loop_};
        }

        run_loop* __loop_;
      };

      __scheduler get_scheduler() noexcept {
        return __scheduler{this};
      }

      struct __run_sender {
        using __t = __run_sender;
        using __id = __run_sender;
        using completion_signatures = __completion_signatures_< set_value_t(), set_stopped_t()>;

       private:
        friend __scheduler;

        template <class _Receiver>
        using __operation = stdexec::__t<__operation<stdexec::__id<_Receiver>>>;

        template <class _Receiver>
        friend __operation<_Receiver>
          tag_invoke(connect_t, const __run_sender& __self, _Receiver __rcvr) {
          return __self.__connect_((_Receiver&&) __rcvr);
        }

        template <class _Receiver>
        __operation<_Receiver> __connect_(_Receiver&& __rcvr) const {
          return {&__loop_->__head_, __loop_, (_Receiver&&) __rcvr};
        }

        template <class _CPO>
        friend __scheduler
          tag_invoke(get_completion_scheduler_t<_CPO>, const __run_sender& __self) noexcept {
          return __scheduler{__self.__loop_};
        }

        explicit __run_sender(run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        run_loop* __loop_;
      };
      template <class _Sender>
      __run_sender run(_Sender&& __sndr);

      void run();

      void finish();

     private:
      void __push_back_(__task* __task);
      __task* __pop_front_();

      std::mutex __mutex_;
      std::condition_variable __cv_;
      __task __head_{.__tail_ = &__head_};
      bool __stop_ = false;
    };

    template <class _ReceiverId>
    inline void __operation<_ReceiverId>::__t::__start_() noexcept try {
      __loop_->__push_back_(this);
    } catch (...) {

      set_error((_Receiver&&) __rcvr_, std::current_exception());
    }

    template <class _Sender>
    run_loop::__run_sender run_loop::run(_Sender&& __sndr) {
      return {this, (_Sender&&) __sndr};
    }

    inline void run_loop::run() {
      for (__task* __task; (__task = __pop_front_()) != &__head_;) {
        __task->__execute();
      }
    }

    inline void run_loop::finish() {
      std::unique_lock __lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::__push_back_(__task* __task) {
      std::unique_lock __lock{__mutex_};
      __task->__next_ = &__head_;
      __head_.__tail_ = __head_.__tail_->__next_ = __task;
      __cv_.notify_one();
    }

    inline __task* run_loop::__pop_front_() {
      std::unique_lock __lock{__mutex_};
      __cv_.wait(__lock, [this] { return __head_.__next_ != &__head_ || __stop_; });
      if (__head_.__tail_ == __head_.__next_)
        __head_.__tail_ = &__head_;
      return std::exchange(__head_.__next_, __head_.__next_->__next_);
    }
  } // namespace __loop

  // NOT TO SPEC
  using run_loop = __loop::run_loop;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    template <class _Sender>
    using __into_variant_result_t = decltype(stdexec::into_variant(__declval<_Sender>()));

    struct __env {
      using __t = __env;
      using __id = __env;
      stdexec::run_loop::__scheduler __sched_;

      friend auto tag_invoke(stdexec::get_scheduler_t, const __env& __self) noexcept
        -> stdexec::run_loop::__scheduler {
        return __self.__sched_;
      }

      friend auto tag_invoke(stdexec::get_delegatee_scheduler_t, const __env& __self) noexcept
        -> stdexec::run_loop::__scheduler {
        return __self.__sched_;
      }
    };

    // What should sync_wait(just_stopped()) return?
    template <class _Sender, class _Continuation>
    using __sync_wait_result_impl =
      __value_types_of_t< _Sender, __env, __transform<__q<decay_t>, _Continuation>, __q<__msingle>>;

    template <stdexec::sender<__env> _Sender>
    using __sync_wait_result_t = __sync_wait_result_impl<_Sender, __q<std::tuple>>;

    template <class _Sender>
    using __sync_wait_with_variant_result_t =
      __sync_wait_result_t<__into_variant_result_t<_Sender>>;

    template <class... _Values>
    struct __state {
      using _Tuple = std::tuple<_Values...>;
      std::variant<std::monostate, _Tuple, std::exception_ptr, set_stopped_t> __data_{};
    };

    template <class... _Values>
    struct __receiver {
      struct __t {
        using __id = __receiver;
        __state<_Values...>* __state_;
        stdexec::run_loop* __loop_;

        template <class _Error>
        void __set_error(_Error __err) noexcept {
          if constexpr (__decays_to<_Error, std::exception_ptr>)
            __state_->__data_.template emplace<2>((_Error&&) __err);
          else if constexpr (__decays_to<_Error, std::error_code>)
            __state_->__data_.template emplace<2>(
              std::make_exception_ptr(std::system_error(__err)));
          else
            __state_->__data_.template emplace<2>(std::make_exception_ptr((_Error&&) __err));
          __loop_->finish();
        }

        template <class... _As>
          requires constructible_from<std::tuple<_Values...>, _As...>
        friend void tag_invoke(stdexec::set_value_t, __t&& __rcvr, _As&&... __as) noexcept try {
          __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
          __rcvr.__loop_->finish();
        } catch (...) {

          __rcvr.__set_error(std::current_exception());
        }

        template <class _Error>
        friend void tag_invoke(stdexec::set_error_t, __t&& __rcvr, _Error __err) noexcept {
          __rcvr.__set_error((_Error&&) __err);
        }

        friend void tag_invoke(stdexec::set_stopped_t __d, __t&& __rcvr) noexcept {
          __rcvr.__state_->__data_.template emplace<3>(__d);
          __rcvr.__loop_->finish();
        }

        friend __env tag_invoke(stdexec::get_env_t, const __t& __rcvr) noexcept {
          return {__rcvr.__loop_->get_scheduler()};
        }
      };
    };

    template <class _Sender>
    using __into_variant_result_t = decltype(stdexec::into_variant(__declval<_Sender>()));

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      template <class _Sender>
      using __receiver_t = __t<__sync_wait_result_impl<_Sender, __q<__receiver>>>;

      // TODO: constrain on return type
      template <__single_value_variant_sender<__env> _Sender> // NOT TO SPEC
        requires __tag_invocable_with_completion_scheduler< sync_wait_t, set_value_t, _Sender>
      tag_invoke_result_t< sync_wait_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender>
        operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<
                 sync_wait_t,
                 __completion_scheduler_for<_Sender, set_value_t>,
                 _Sender>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(sync_wait_t{}, std::move(__sched), (_Sender&&) __sndr);
      }

      // TODO: constrain on return type
      template <__single_value_variant_sender<__env> _Sender> // NOT TO SPEC
        requires(!__tag_invocable_with_completion_scheduler< sync_wait_t, set_value_t, _Sender>)
             && tag_invocable<sync_wait_t, _Sender>
      tag_invoke_result_t<sync_wait_t, _Sender> operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<sync_wait_t, _Sender>) {
        return tag_invoke(sync_wait_t{}, (_Sender&&) __sndr);
      }

      template <__single_value_variant_sender<__env> _Sender>
        requires(!__tag_invocable_with_completion_scheduler< sync_wait_t, set_value_t, _Sender>)
             && (!tag_invocable<sync_wait_t, _Sender>) && sender<_Sender, __env>
             && sender_to<_Sender, __receiver_t<_Sender>>
      auto operator()(_Sender&& __sndr) const -> std::optional<__sync_wait_result_t<_Sender>> {
        using state_t = __sync_wait_result_impl<_Sender, __q<__state>>;
        using tail_t = next_tail_from_sender_to_t<_Sender, __receiver_t<_Sender>>;
        state_t __state{};
        run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state = connect((_Sender&&) __sndr, __receiver_t<_Sender>{&__state, &__loop});
        tail_t __tail = start(__op_state);

        // Wait for the variant to be filled in.
        auto __tail_run = __loop.run(just());

        resume_tail_senders(__null_tail_sender{}, __tail, __tail_run);

        if (__state.__data_.index() == 2)
          std::rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return std::nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    };
  } // namespace __sync_wait

  using __sync_wait::sync_wait_t;
  inline constexpr sync_wait_t sync_wait{};
} // namespace exec
