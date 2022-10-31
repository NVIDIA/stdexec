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

  template <typename T>
    concept __contextually_convertible_to_bool =
      requires(const T c) {
        { (static_cast<const T&&>(c) ? false : false) } -> same_as<bool>;
      };

  template <typename T>
  static constexpr bool __nothrow_contextually_convertible_to_bool_v =
    noexcept((std::declval<const T&&>() ? (void)0 : (void)0));

  template <typename T>
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
        requires std::tag_invocable<always_completes_inline_t, c_t<_Sender>, c_t<_Env>>
      constexpr bool operator()(c_t<_Sender>&& __s, c_t<_Env>&& __e) const noexcept {
        static_assert(same_as<bool, tag_invoke_result_t<always_completes_inline_t, c_t<_Sender>, c_t<_Env>>>);
        static_assert(std::nothrow_tag_invocable<always_completes_inline_t, c_t<_Sender>, c_t<_Env>>);
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
      always_completes_inline(c_v<_Sender>, c_v<_Env>);

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

  template <tail_operation_state _TailOperationState>
    using __next_tail_from_operation_t =
      __call_result_t<start_t, _TailOperationState&>;

  template <tail_sender _TailSender, tail_receiver _TailReceiver>
    using __next_tail_from_sender_to_t =
      __next_tail_from_operation_t<stdexec::connect_result_t<_TailSender, _TailReceiver>>;

  template <class _TailSender, class _Env = no_env>
    concept __tail_sender_or_void =
      same_as<_TailSender, void> || tail_sender<_TailSender, _Env>;

  template <class _TailSender, class _TailReceiver>
    concept tail_sender_to =
      tail_sender<_TailSender> &&
      tail_receiver<_TailReceiver> &&
      requires(_TailSender&& __s, _TailReceiver&& __r) {
        { stdexec::connect((_TailSender&&) __s, (_TailReceiver&&) __r) } noexcept ->
          tail_operation_state;
      } &&
      __tail_sender_or_void<__next_tail_from_sender_to_t<_TailSender, _TailReceiver>>;

  template <class _TailSender, class _TailReceiver>
    concept __terminal_tail_sender_to =
      tail_sender_to<_TailSender, _TailReceiver> &&
      same_as<__next_tail_from_sender_to_t<_TailSender, _TailReceiver>, void>;

  template <class _TailSender, class _TailReceiver, class... _ValidTailSender>
    concept __recursive_tail_sender_to =
      tail_sender_to<_TailSender, _TailReceiver> &&
      tail_operation_state<connect_result_t<_TailSender, _TailReceiver>> &&
      __one_of<__next_tail_from_sender_to_t<_TailSender, _TailReceiver>, _ValidTailSender...>;

  template <class _TailOperationState>
    concept __nullable_tail_operation_state =
      tail_operation_state<_TailOperationState> &&
      __nothrow_contextually_convertible_to_bool<_TailOperationState>;

  template <class _TailSender, class _TailReceiver>
    concept __nullable_tail_sender_to =
      tail_sender_to<_TailSender, _TailReceiver> &&
      __nullable_tail_operation_state<connect_result_t<_TailSender, _TailReceiver>>;


  struct __null_tail_receiver {
    void set_value() noexcept {}
    void set_error(std::exception_ptr) noexcept {}
    void set_done() noexcept {}
  };

  struct __null_tail_sender {
    struct op {
      // this is a nullable_tail_sender that always returns false to prevent
      // callers from calling start() and unwind()
      inline constexpr operator bool() const noexcept { return false; }
      friend void tag_invoke(start_t, op& self) noexcept {
        std::terminate();
      }

      friend void tag_invoke(unwind_t, op& self) noexcept {
        std::terminate();
      }
    };

    using completion_signatures = completion_signatures<set_value_t(), set_stopped_t()>;

    template <class Receiver>
    friend auto tag_invoke(connect_t, __null_tail_sender&&, Receiver&&)
        -> op {
      return {};
    }

    template<class _Env>
    friend constexpr bool tag_invoke(
        exec::always_completes_inline_t, exec::c_t<__null_tail_sender>, exec::c_t<_Env>) noexcept {
      return true;
    }
  };

} // namespace exec
