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

  template <class T>
    concept __contextually_convertible_to_bool =
      requires(const T c) {
        { (static_cast<const T&&>(c) ? false : false) } -> same_as<bool>;
      };

  template <class T>
    static constexpr bool __nothrow_contextually_convertible_to_bool_v =
      noexcept((std::declval<const T&&>() ? (void)0 : (void)0));

  template <class T>
    concept __nothrow_contextually_convertible_to_bool =
      __contextually_convertible_to_bool<T> &&
      __nothrow_contextually_convertible_to_bool_v<T>;

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

  template <class _T>
  struct c_t {
  };
  template <class _T>
  inline constexpr c_t<_T> c_v{};

  namespace __sender_queries {
    template <class _Ty>
    const _Ty& __cref_fn(const _Ty&);
    template <class _Ty>
    using __cref_t =
      decltype(__sender_queries::__cref_fn(__declval<_Ty>()));

    struct always_completes_inline_t {
      template <class _Sender, class _Env = no_env>
          requires std::tag_invocable<always_completes_inline_t, __cref_t<_Sender>, __cref_t<_Env>>
        constexpr bool operator()(_Sender&& __s, _Env&& __e) const noexcept {
          static_assert(same_as<bool, tag_invoke_result_t<always_completes_inline_t, __cref_t<_Sender>, __cref_t<_Env>>>);
          static_assert(std::nothrow_tag_invocable<always_completes_inline_t, __cref_t<_Sender>, __cref_t<_Env>>);
          return tag_invoke(always_completes_inline_t{}, std::as_const(__s), std::as_const(__e));
        }
      template <class _Sender, class _Env = no_env>
          requires std::tag_invocable<always_completes_inline_t, c_t<std::decay_t<_Sender>>, c_t<std::decay_t<_Env>>>
        constexpr bool operator()(c_t<std::decay_t<_Sender>>&& __s, c_t<std::decay_t<_Env>>&& __e) const noexcept {
          static_assert(same_as<bool, tag_invoke_result_t<always_completes_inline_t, c_t<std::decay_t<_Sender>>, c_t<std::decay_t<_Env>>>>);
          static_assert(std::nothrow_tag_invocable<always_completes_inline_t, c_t<std::decay_t<_Sender>>, c_t<std::decay_t<_Env>>>);
          return tag_invoke(always_completes_inline_t{}, __s, __e);
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
      operation_state<_TailOperationState> &&
      std::is_nothrow_destructible_v<_TailOperationState> &&
      std::is_trivially_destructible_v<_TailOperationState> &&
      (!std::is_copy_constructible_v<_TailOperationState>) &&
      (!std::is_move_constructible_v<_TailOperationState>) &&
      (!std::is_copy_assignable_v<_TailOperationState>) &&
      (!std::is_move_assignable_v<_TailOperationState>) &&
      requires (_TailOperationState& __o) {
          { unwind(__o) } noexcept;
      };

  template <class _Sender, class _Env = no_env>
    constexpr bool always_completes_inline_v =
      always_completes_inline(c_v<std::decay_t<_Sender>>, c_v<std::decay_t<_Env>>);

  template <class _TailSender, class _Env = no_env>
    concept tail_sender =
      sender<_TailSender, _Env> &&
      same_as<__single_sender_value_t<_TailSender, _Env>, void> &&
      always_completes_inline_v<_TailSender, _Env> &&
      std::is_nothrow_move_constructible_v<_TailSender> &&
      std::is_nothrow_destructible_v<_TailSender> &&
      std::is_trivially_destructible_v<_TailSender>;

  template <class _TailReceiver>
    concept tail_receiver =
      receiver<_TailReceiver> &&
      std::is_nothrow_copy_constructible_v<_TailReceiver> &&
      std::is_nothrow_move_constructible_v<_TailReceiver> &&
      std::is_nothrow_copy_assignable_v<_TailReceiver> &&
      std::is_nothrow_move_assignable_v<_TailReceiver> &&
      std::is_nothrow_destructible_v<_TailReceiver> &&
      std::is_trivially_destructible_v<_TailReceiver>;

  struct __null_tail_receiver {
    friend void tag_invoke(set_value_t, __null_tail_receiver&&, auto&&...) noexcept {}
    friend void tag_invoke(set_stopped_t, __null_tail_receiver&&) noexcept {}
    friend __empty_env tag_invoke(get_env_t, const __null_tail_receiver& __self) {
      return {};
    }
  };

  struct __null_tail_sender {
    struct op : __immovable {
      op() = default;

      // this is a nullable_tail_sender that always returns false to prevent
      // callers from calling start() and unwind()
      inline constexpr operator bool() const noexcept { return false; }

      friend void tag_invoke(start_t, op& self) noexcept {
        printf("__null_tail_sender start\n"); fflush(stdout);
        std::terminate();
      }

      friend void tag_invoke(unwind_t, op& self) noexcept {
        printf("__null_tail_sender unwind\n"); fflush(stdout);
        std::terminate();
      }
    };

    using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

    template <class _TailReceiver>
      friend auto tag_invoke(connect_t, __null_tail_sender&&, _TailReceiver&&) noexcept
          -> op {
        return {};
      }

    template<class _Env>
      friend constexpr bool tag_invoke(
          exec::always_completes_inline_t, exec::c_t<__null_tail_sender>, exec::c_t<_Env>) noexcept {
        return true;
      }
  };

  template <typename... _TailSenderN>
  struct __variant_tail_sender;

  template <tail_operation_state _TailOperationState>
    using __next_tail_from_operation_t =
      __call_result_t<start_t, _TailOperationState&>;

  template <tail_operation_state _TailOperationState>
    using next_tail_from_operation_t =
      __if<__bool<std::is_void_v<__call_result_t<start_t, _TailOperationState&>>>,
        __null_tail_sender,
      __call_result_t<start_t, _TailOperationState&>>;

  template <tail_sender _TailSender, tail_receiver _TailReceiver>
    using __next_tail_from_sender_to_t =
      __next_tail_from_operation_t<stdexec::connect_result_t<_TailSender, _TailReceiver>>;

  template <tail_sender _TailSender, tail_receiver _TailReceiver>
    using next_tail_from_sender_to_t =
      next_tail_from_operation_t<stdexec::connect_result_t<_TailSender, _TailReceiver>>;

  template <class _TailSender, class _TailReceiver>
    concept tail_sender_to =
      tail_sender<_TailSender> &&
      tail_receiver<_TailReceiver> &&
      requires(_TailSender&& __s, _TailReceiver&& __r) {
        { stdexec::connect((_TailSender&&) __s, (_TailReceiver&&) __r) } noexcept ->
          tail_operation_state;
      } &&
      tail_sender<next_tail_from_sender_to_t<_TailSender, _TailReceiver>>;

  template <class _TailOperationState>
    concept __terminal_tail_operation_state =
      tail_operation_state<_TailOperationState> &&
      same_as<__next_tail_from_operation_t<_TailOperationState>, void>;

  template <class _TailSender, class _TailReceiver>
    concept __terminal_tail_sender_to =
      tail_sender_to<_TailSender, _TailReceiver> &&
      same_as<__next_tail_from_sender_to_t<_TailSender, _TailReceiver>, void>;

  template <class _TailSender, class _TailReceiver, class... _ValidTailSender>
    concept __recursive_tail_sender_to =
      tail_sender_to<_TailSender, _TailReceiver> &&
      tail_operation_state<connect_result_t<_TailSender, _TailReceiver>> &&
      __one_of<next_tail_from_sender_to_t<_TailSender, _TailReceiver>, _ValidTailSender...>;

  template <class _TailOperationState>
    concept __nullable_tail_operation_state =
      tail_operation_state<_TailOperationState> &&
      __nothrow_contextually_convertible_to_bool<_TailOperationState>;

  template <class _TailSender, class _TailReceiver>
    concept __nullable_tail_sender_to =
      tail_sender_to<_TailSender, _TailReceiver> &&
      __nullable_tail_operation_state<connect_result_t<_TailSender, _TailReceiver>>;

} // namespace exec



#include "variant_tail_sender.hpp"



namespace exec {

  template<tail_sender _To, tail_sender _From>
    constexpr std::decay_t<_To> result_from(_From&& __f) noexcept {
      if constexpr (
          __is_instance_of<std::decay_t<_From>, __variant_tail_sender> &&
          __is_instance_of<std::decay_t<_To>, __variant_tail_sender>) {
        return variant_cast<std::decay_t<_To>>((_From&&)__f);
      } else if constexpr (
          __is_instance_of<std::decay_t<_From>, __variant_tail_sender> &&
          !__is_instance_of<std::decay_t<_To>, __variant_tail_sender>) {
        return get<std::decay_t<_To>>((_From&&)__f);
      } else {
        static_assert(std::is_constructible_v<_To, _From>, "result_from cannot convert");
        return (_From&&)__f;
      }
    }

  template<tail_sender _TailSender>
      requires (!same_as<__null_tail_sender, _TailSender>)
    struct maybe_tail_sender {
      maybe_tail_sender() noexcept = default;
      maybe_tail_sender(__null_tail_sender) noexcept {}
      maybe_tail_sender(_TailSender __t) noexcept : tail_sender_(__t) {}
      template <class _TailReceiver>
      struct op : __immovable {
        using op_t = connect_result_t<_TailSender, _TailReceiver>;
        op() = default;

        explicit op(_TailSender __t, _TailReceiver __r)
          : op_(stdexec::__conv{
              [&] {
                return stdexec::connect(__t, __r);
              }
            }) {}
        operator bool() const noexcept {
          if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
            return !!op_ && !!*op_;
          } else {
            return !!op_;
          }
        }

        [[nodiscard]]
        friend auto tag_invoke(start_t, op& __self) noexcept {
          if (!__self.op_) {
            printf("maybe_tail_sender start optional\n"); fflush(stdout);
            std::terminate();
          }
          if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
            if (!*__self.op_) {
              printf("maybe_tail_sender start nullable\n"); fflush(stdout);
              std::terminate();
            }
          }
          return stdexec::start(*__self.op_);
        }

        friend void tag_invoke(unwind_t, op& __self) noexcept {
          if (!__self.op_) {
            printf("maybe_tail_sender unwind optional\n"); fflush(stdout);
            std::terminate();
          }
          if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
            if (!*__self.op_) {
              printf("maybe_tail_sender unwind nullable\n"); fflush(stdout);
              std::terminate();
            }
          }
          exec::unwind(*__self.op_);
        }
        std::optional<op_t> op_;
      };

      using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

      template <class _TailReceiver>
      [[nodiscard]]
        friend auto tag_invoke(connect_t, maybe_tail_sender&& __self, _TailReceiver&& __r) noexcept
            -> op<std::decay_t<_TailReceiver>> {
          if (!__self.tail_sender_) { return {}; }
          return op<std::decay_t<_TailReceiver>>{*((maybe_tail_sender&&)__self).tail_sender_, __r};
        }

      template<class _Env>
        friend constexpr bool tag_invoke(
          exec::always_completes_inline_t, exec::c_t<maybe_tail_sender>, exec::c_t<_Env>) noexcept {
          return true;
        }

    private:
      std::optional<_TailSender> tail_sender_;
    };

  template<tail_sender _TailSender, tail_receiver _TailReceiver = __null_tail_receiver>
    struct scoped_tail_sender {
      explicit scoped_tail_sender(_TailSender __t, _TailReceiver __r = _TailReceiver{}) noexcept
        : t_(__t)
        , r_(__r)
        , valid_(true) {}

      scoped_tail_sender(scoped_tail_sender&& other) noexcept
        : t_(other.s_)
        , r_(other.r_)
        , valid_(std::exchange(other.valid_, false)) {}

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

      _TailSender get() noexcept { return t_; }

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
      friend auto tag_invoke(connect_t, __all_resumed_tail_sender&&, _TailReceiver&& __r) noexcept
          -> __call_result_t<connect_t, __null_tail_sender, _TailReceiver> {
        return stdexec::connect(__null_tail_sender{}, __r);
      }

    template<class _Env>
      friend constexpr bool tag_invoke(
        exec::always_completes_inline_t, exec::c_t<__all_resumed_tail_sender>, exec::c_t<_Env>) noexcept {
        return true;
      }
  };

  namespace __start_until_nullable_ {

    struct __start_until_nullable_t;

    template<tail_sender _TailSender, tail_receiver _TailReceiver>
      struct __start_until_nullable_result;

    template<class _TailSender, class _TailReceiver>
      using __start_until_nullable_result_t =
        typename __start_until_nullable_result<_TailSender, _TailReceiver>::type;

    template<tail_sender _TailSender, tail_receiver _TailReceiver>
      struct __start_until_nullable_result {
        using type =
          __if<
            __bool<__nullable_tail_sender_to<_TailSender, _TailReceiver>>, _TailSender,
            __if<
              __bool<__terminal_tail_sender_to<_TailSender, _TailReceiver>>, __all_resumed_tail_sender,
              __minvoke<__with_default<__q<__start_until_nullable_result_t>, __all_resumed_tail_sender>,
                __next_tail_from_sender_to_t<_TailSender, _TailReceiver>,
                _TailReceiver>
              >
            >;
      };

    struct __start_until_nullable_t {
      template<tail_sender _TailSender, tail_receiver _TailReceiver>
        auto operator()(_TailSender __t, _TailReceiver __r) const noexcept
          -> __start_until_nullable_result_t<_TailSender, _TailReceiver> {
          if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
            return __t;
          } else if constexpr (__terminal_tail_sender_to<_TailSender, _TailReceiver>) {
            // restrict scope of op
            {
              auto op = stdexec::connect(std::move(__t), std::move(__r));
              stdexec::start(op);
            }
            return __all_resumed_tail_sender{};
          } else {
            auto op = stdexec::connect(std::move(__t), __r);
            return __start_until_nullable_t{}(stdexec::start(op), std::move(__r));
          }
        }
    };

  } // namespace __start_until_nullable_
  using __start_until_nullable_::__start_until_nullable_t;
  inline constexpr __start_until_nullable_t __start_until_nullable{};

  template <tail_sender _NextTailSender, tail_sender _TailSender, tail_receiver _TailReceiver, tail_sender... _PrevTailSenders>
    auto __start_next(_NextTailSender __next, _TailReceiver __r) noexcept;

  template<tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
    struct __start_sequential_result;

  template<tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
    using __start_sequential_result_t = typename __start_sequential_result<_TailSender, _TailReceiver>::type;

  template<tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
    struct __start_sequential_result {
        using next_t = next_tail_from_sender_to_t<_TailSender, _TailReceiver>;
        using start_next_t = __call_result_t<
                decltype(__start_next<next_t, _TailSender, _TailReceiver, _PrevTailSenders...>),
                  next_t,
                  _TailReceiver>;
        using type =
          __if<
              __bool<
                __nullable_tail_sender_to<_TailSender, _TailReceiver> &&
                __terminal_tail_sender_to<_TailSender, _TailReceiver>>,
            _TailSender,
            __if< // elseif
                __bool<__nullable_tail_sender_to<_TailSender, _TailReceiver>>,
              variant_tail_sender<__all_resumed_tail_sender, start_next_t>,
              __if< // elseif
                  __bool<!__terminal_tail_sender_to<_TailSender, _TailReceiver>>,
                start_next_t,
                __all_resumed_tail_sender // else
                >
              >
            >;
        };

  template<tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
    auto __start_sequential(_TailSender c, _TailReceiver r) noexcept
        -> __start_sequential_result_t<_TailSender, _TailReceiver, _PrevTailSenders...>;

  template <tail_sender _NextTailSender, tail_sender _TailSender, tail_receiver _TailReceiver, tail_sender... _PrevTailSenders>
    auto __start_next(_NextTailSender __next, _TailReceiver __r) noexcept {
      if constexpr (__one_of<_NextTailSender, _TailSender, _PrevTailSenders...>) {
        static_assert(
            (__nullable_tail_sender_to<_TailSender, _TailReceiver> ||
            (__nullable_tail_sender_to<_PrevTailSenders, _TailReceiver> || ...)),
            "At least one tail_sender in a cycle must be nullable to avoid "
            "entering an infinite loop");
        return __start_until_nullable(__next, std::move(__r));
      } else {
        return __start_sequential<_NextTailSender, _TailReceiver, _TailSender, _PrevTailSenders...>(__next, std::move(__r));
      }
    }

  template<tail_sender _TailSender, tail_receiver _TailReceiver, class... _PrevTailSenders>
    auto __start_sequential(_TailSender c, _TailReceiver r) noexcept
        -> __start_sequential_result_t<_TailSender, _TailReceiver, _PrevTailSenders...> {
      using next_t = next_tail_from_sender_to_t<_TailSender, _TailReceiver>;
      using result_t = __start_sequential_result_t<_TailSender, _TailReceiver, _PrevTailSenders...>;

      if constexpr (
          __nullable_tail_sender_to<_TailSender, _TailReceiver> &&
          __terminal_tail_sender_to<_TailSender, _TailReceiver>) {
        // halt the recursion
        return c;
      } else if constexpr (__nullable_tail_sender_to<_TailSender, _TailReceiver>) {
        // recurse if the nullable tail-sender is valid otherwise return
        // a nullable and terminal tail-sender
        auto op = stdexec::connect(std::move(c), r);
        if (!op) {
          return __all_resumed_tail_sender{};
        }
        return result_from<result_t>(
          __start_next<next_t, _TailSender, _TailReceiver, _PrevTailSenders...>(
            stdexec::start(op), r));
      } else if constexpr (!__terminal_tail_sender_to<_TailSender, _TailReceiver>) {
        auto op = stdexec::connect(std::move(c), r);
        return result_from<result_t>(
          __start_next<next_t, _TailSender, _TailReceiver, _PrevTailSenders...>(
            stdexec::start(op), r));
      } else {
        // run the terminal and not nullable tail-sender and return
        // a nullable and terminal tail-sender
        auto op = stdexec::connect(std::move(c), r);
        stdexec::start(op);
        return __all_resumed_tail_sender{};
      }
    }


  template<tail_receiver _TailReceiver>
    inline __null_tail_sender resume_tail_senders_until_one_remaining(_TailReceiver&&) noexcept {
      return {};
    }

  template<tail_receiver _TailReceiver, tail_sender C>
    C resume_tail_senders_until_one_remaining(_TailReceiver&&, C c) noexcept {
      return c;
    }

  template<tail_receiver _TailReceiver, tail_sender... Cs, std::size_t... Is>
    auto _resume_tail_senders_until_one_remaining(_TailReceiver&& __r, std::index_sequence<Is...>, Cs... cs) noexcept {
      using result_type = variant_tail_sender<decltype(__start_sequential(__start_sequential(cs, __r), __r))...>;
      result_type result;

      auto cs2_tuple = std::make_tuple(
        variant_tail_sender<
          __all_resumed_tail_sender,
          decltype(__start_sequential(__start_sequential(cs, __r), __r))>{
            __start_sequential(__start_sequential(cs, __r), __r)}...);
      while (true) {
        std::size_t remaining = sizeof...(cs);
        ((remaining > 1 ?
          (!holds_alternative<__all_resumed_tail_sender>(std::get<Is>(cs2_tuple)) ?
            (void)(result = result_from<result_type>(
              std::get<Is>(cs2_tuple) = result_from<decltype(std::get<Is>(cs2_tuple))>(
                __start_sequential(std::get<Is>(cs2_tuple), __r)))) :
            (void)--remaining) :
          (void)(result = result_from<result_type>(std::get<Is>(cs2_tuple)))), ...);

        if (remaining <= 1) {
          return result;
        }
      }
    }

  template<tail_receiver _TailReceiver, tail_sender... Cs>
    auto resume_tail_senders_until_one_remaining(_TailReceiver&& __r, Cs... cs) noexcept {
      return _resume_tail_senders_until_one_remaining(__r, std::index_sequence_for<Cs...>{}, cs...);
    }

} // namespace exec
