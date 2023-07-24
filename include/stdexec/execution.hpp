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

#include <atomic>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <variant>

#include "__detail/__execution_fwd.hpp"
#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__scope.hpp"
#include "functional.hpp"
#include "concepts.hpp"
#include "coroutine.hpp"
#include "stop_token.hpp"

#ifdef __EDG__
#pragma diagnostic push
#pragma diag_suppress 1302
#pragma diag_suppress 497
#pragma diag_suppress type_qualifiers_ignored_on_reference
#endif

#ifndef STDEXEC_DISABLE_R5_DEPRECATION_WARNINGS
#define STDEXEC_R5_SENDER_DEPRECATION_WARNING \
  [[deprecated( \
    "Deprecated sender type detected. " \
    "Please give the type a nested `is_sender` type alias, or " \
    "specialize stdexec::enable_sender<your-sender-type> to be `true`. " \
    "To suppress this deprecation warning, define `STDEXEC_DISABLE_R5_DEPRECATIONS`.")]]
#define STDEXEC_R5_RECEIVER_DEPRECATION_WARNING \
  [[deprecated( \
    "Deprecated receiver type detected. " \
    "Please give the type a nested `is_receiver` type alias, or " \
    "specialize stdexec::enable_receiver<your-receiver-type> to be `true`." \
    "To suppress this deprecation warning, define `STDEXEC_DISABLE_R5_DEPRECATIONS`.")]]
#define STDEXEC_R5_DEPENDENT_COMPLETION_SIGNATURES_DEPRECATION_WARNING \
  [[deprecated( \
    "The `dependent_completion_signatures<>` type is deprecated. There is " \
    "no need to define a customization of `get_completion_signatures` for " \
    "sender types whose completions are dependent on the receiver's environment. " \
    "Give your sender type a nested `is_sender` type alias instead, " \
    "specialize stdexec::enable_sender<your-sender-type> to be `true`. " \
    "To suppress this deprecation warning, define `STDEXEC_DISABLE_R5_DEPRECATIONS`.")]]
#define STDEXEC_R5_NO_ENV_DEPRECATION_WARNING \
  [[deprecated( \
    "The `no_env` type is deprecated. The stdexec library no longer needs to use " \
    "it to probe a type for satisfaction of the `sender` concept. " \
    "Give your sender type a nested `is_sender` type alias instead, or " \
    "specialize stdexec::enable_sender<your-sender-type> to be `true`. " \
    "To suppress this deprecation warning, define `STDEXEC_DISABLE_R5_DEPRECATIONS`.")]]
#else
#define STDEXEC_R5_SENDER_DEPRECATION_WARNING
#define STDEXEC_R5_RECEIVER_DEPRECATION_WARNING
#define STDEXEC_R5_DEPENDENT_COMPLETION_SIGNATURES_DEPRECATION_WARNING
#define STDEXEC_R5_NO_ENV_DEPRECATION_WARNING
#endif

#define STDEXEC_LEGACY_R5_CONCEPTS() 1

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE("-Wpragmas")
STDEXEC_PRAGMA_IGNORE("-Wundefined-inline")
STDEXEC_PRAGMA_IGNORE("-Wundefined-internal")

namespace stdexec {
  // [exec.queries.queryable]
  template <class T>
  concept queryable = destructible<T>;

  template <class Tag>
  struct __query {
    template <class Sig>
    static inline constexpr Tag (*signature)(Sig) = nullptr;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // [exec.queries]
  namespace __queries {
    struct forwarding_query_t {
      template <class _Query>
      constexpr bool operator()(_Query __query) const noexcept {
        if constexpr (tag_invocable<forwarding_query_t, _Query>) {
          return tag_invoke(*this, (_Query&&) __query);
        } else if constexpr (std::derived_from<_Query, forwarding_query_t>) {
          return true;
        } else {
          return false;
        }
      }
    };

    struct execute_may_block_caller_t : __query<execute_may_block_caller_t> {
      template <class _Tp>
        requires tag_invocable<execute_may_block_caller_t, __cref_t<_Tp>>
      constexpr bool operator()(_Tp&& __t) const noexcept {
        static_assert(
          same_as<bool, tag_invoke_result_t<execute_may_block_caller_t, __cref_t<_Tp>>>);
        static_assert(nothrow_tag_invocable<execute_may_block_caller_t, __cref_t<_Tp>>);
        return tag_invoke(execute_may_block_caller_t{}, std::as_const(__t));
      }

      constexpr bool operator()(auto&&) const noexcept {
        return true;
      }
    };

    struct get_forward_progress_guarantee_t : __query<get_forward_progress_guarantee_t> {
      template <class _Tp>
        requires tag_invocable<get_forward_progress_guarantee_t, __cref_t<_Tp>>
      constexpr auto operator()(_Tp&& __t) const
        noexcept(nothrow_tag_invocable<get_forward_progress_guarantee_t, __cref_t<_Tp>>)
          -> tag_invoke_result_t<get_forward_progress_guarantee_t, __cref_t<_Tp>> {
        return tag_invoke(get_forward_progress_guarantee_t{}, std::as_const(__t));
      }

      constexpr stdexec::forward_progress_guarantee operator()(auto&&) const noexcept {
        return stdexec::forward_progress_guarantee::weakly_parallel;
      }
    };

    struct __has_algorithm_customizations_t : __query<__has_algorithm_customizations_t> {
      template <class _Tp>
      using __result_t = tag_invoke_result_t<__has_algorithm_customizations_t, __cref_t<_Tp>>;

      template <class _Tp>
        requires tag_invocable<__has_algorithm_customizations_t, __cref_t<_Tp>>
      constexpr __result_t<_Tp> operator()(_Tp&&) const noexcept(noexcept(__result_t<_Tp>{})) {
        using _Boolean = tag_invoke_result_t<__has_algorithm_customizations_t, __cref_t<_Tp>>;
        static_assert(_Boolean{} ? true : true); // must be contextually convertible to bool
        return _Boolean{};
      }

      constexpr std::false_type operator()(auto&&) const noexcept {
        return {};
      }
    };

    // TODO: implement allocator concept
    template <class _T0>
    concept __allocator = true;

    struct get_scheduler_t : __query<get_scheduler_t> {
      friend constexpr bool tag_invoke(forwarding_query_t, const get_scheduler_t&) noexcept {
        return true;
      }

      template <class _Env>
        requires tag_invocable<get_scheduler_t, const _Env&>
      auto operator()(const _Env& __env) const noexcept
        -> tag_invoke_result_t<get_scheduler_t, const _Env&>;

      auto operator()() const noexcept;
    };

    struct get_delegatee_scheduler_t : __query<get_delegatee_scheduler_t> {
      friend constexpr bool
        tag_invoke(forwarding_query_t, const get_delegatee_scheduler_t&) noexcept {
        return true;
      }

      template <class _Env>
        requires tag_invocable<get_delegatee_scheduler_t, const _Env&>
      auto operator()(const _Env& __t) const noexcept
        -> tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&>;

      auto operator()() const noexcept;
    };

    struct get_allocator_t : __query<get_allocator_t> {
      friend constexpr bool tag_invoke(forwarding_query_t, const get_allocator_t&) noexcept {
        return true;
      }

      template <class _Env>
        requires tag_invocable<get_allocator_t, const _Env&>
      auto operator()(const _Env& __env) const noexcept
        -> tag_invoke_result_t<get_allocator_t, const _Env&> {
        static_assert(nothrow_tag_invocable<get_allocator_t, const _Env&>);
        static_assert(__allocator<tag_invoke_result_t<get_allocator_t, const _Env&>>);
        return tag_invoke(get_allocator_t{}, __env);
      }

      auto operator()() const noexcept;
    };

    struct get_stop_token_t : __query<get_stop_token_t> {
      friend constexpr bool tag_invoke(forwarding_query_t, const get_stop_token_t&) noexcept {
        return true;
      }

      template <class _Env>
      never_stop_token operator()(const _Env&) const noexcept {
        return {};
      }

      template <class _Env>
        requires tag_invocable<get_stop_token_t, const _Env&>
      auto operator()(const _Env& __env) const noexcept
        -> tag_invoke_result_t<get_stop_token_t, const _Env&> {
        static_assert(nothrow_tag_invocable<get_stop_token_t, const _Env&>);
        static_assert(stoppable_token<tag_invoke_result_t<get_stop_token_t, const _Env&>>);
        return tag_invoke(get_stop_token_t{}, __env);
      }

      auto operator()() const noexcept;
    };

    template <class _Queryable, class _CPO>
    concept __has_completion_scheduler_for =
      queryable<_Queryable> && //
      tag_invocable<get_completion_scheduler_t<_CPO>, const _Queryable&>;

    template <__completion_tag _CPO>
    struct get_completion_scheduler_t : __query<get_completion_scheduler_t<_CPO>> {
      friend constexpr bool
        tag_invoke(forwarding_query_t, const get_completion_scheduler_t<_CPO>&) noexcept {
        return true;
      }

      template <__has_completion_scheduler_for<_CPO> _Queryable>
      auto operator()(const _Queryable& __queryable) const noexcept
        -> tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&>;
    };
  } // namespace __queries

  using __queries::forwarding_query_t;
  using __queries::execute_may_block_caller_t;
  using __queries::__has_algorithm_customizations_t;
  using __queries::get_forward_progress_guarantee_t;
  using __queries::get_allocator_t;
  using __queries::get_scheduler_t;
  using __queries::get_delegatee_scheduler_t;
  using __queries::get_stop_token_t;
  using __queries::get_completion_scheduler_t;

  inline constexpr forwarding_query_t forwarding_query{};
  inline constexpr execute_may_block_caller_t execute_may_block_caller{};
  inline constexpr __has_algorithm_customizations_t __has_algorithm_customizations{};
  inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};
  inline constexpr get_scheduler_t get_scheduler{};
  inline constexpr get_delegatee_scheduler_t get_delegatee_scheduler{};
  inline constexpr get_allocator_t get_allocator{};
  inline constexpr get_stop_token_t get_stop_token{};
#if !STDEXEC_GCC() || defined(__OPTIMIZE__)
  template <__completion_tag _CPO>
  inline constexpr get_completion_scheduler_t<_CPO> get_completion_scheduler{};
#else
  template <>
  inline constexpr get_completion_scheduler_t<set_value_t> get_completion_scheduler<set_value_t>{};
  template <>
  inline constexpr get_completion_scheduler_t<set_error_t> get_completion_scheduler<set_error_t>{};
  template <>
  inline constexpr get_completion_scheduler_t<set_stopped_t>
    get_completion_scheduler<set_stopped_t>{};
#endif

  template <class _Tag>
  concept __forwarding_query = forwarding_query(_Tag{});

  /////////////////////////////////////////////////////////////////////////////
  // env_of
  namespace __env {
    struct no_env {
      using __t = no_env;
      using __id = no_env;
      template <class _Tag, same_as<no_env> _Self, class... _Ts>
      friend void tag_invoke(_Tag, _Self, _Ts&&...) = delete;
    };

    struct empty_env {
      using __t = empty_env;
      using __id = empty_env;
    };

    template <class _Tag>
    struct __deleted { };

    template <__nothrow_move_constructible _Fun>
    struct __env_fn {
      using __t = __env_fn;
      using __id = __env_fn;
      STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

      template <class _Tag>
        requires __callable<const _Fun&, _Tag>
      friend auto tag_invoke(_Tag, const __env_fn& __self) //
        noexcept(__nothrow_callable<const _Fun&, _Tag>) -> __call_result_t<const _Fun&, _Tag> {
        return __self.__fun_(_Tag());
      }
    };

    template <class _Fun>
    __env_fn(_Fun) -> __env_fn<_Fun>;

    template <__nothrow_move_constructible _Env>
    struct __env_fwd {
      using __t = __env_fwd;
      using __id = __env_fwd;
      STDEXEC_NO_UNIQUE_ADDRESS _Env __env_;

      template <__forwarding_query _Tag>
        requires tag_invocable<_Tag, const _Env&>
      friend auto tag_invoke(_Tag, const __env_fwd& __self) //
        noexcept(nothrow_tag_invocable<_Tag, const _Env&>)
          -> tag_invoke_result_t<_Tag, const _Env&> {
        return _Tag()(__self.__env_);
      }
    };

    template <__nothrow_move_constructible _Env, __nothrow_move_constructible _Base = empty_env>
    struct __env_join : __env_fwd<_Base> {
      using __t = __env_join;
      using __id = __env_join;
      STDEXEC_NO_UNIQUE_ADDRESS _Env __env_;

      const _Base& base() const noexcept {
        return this->__env_fwd<_Base>::__env_;
      }

      template <class _Tag>
        requires tag_invocable<_Tag, const _Env&>
      friend auto tag_invoke(_Tag, const __env_join& __self) //
        noexcept(nothrow_tag_invocable<_Tag, const _Env&>)
          -> tag_invoke_result_t<_Tag, const _Env&> {
        return _Tag()(__self.__env_);
      }
    };

    template <class _Tag, class _Base>
    struct __env_join<__env_fn<__deleted<_Tag>>, _Base> : __env_fwd<_Base> {
      using __t = __env_join;
      using __id = __env_join;
      STDEXEC_NO_UNIQUE_ADDRESS __env_fn<__deleted<_Tag>> __env_;

      friend void tag_invoke(_Tag, const __env_join&) noexcept = delete;
    };

    template <class _Env>
    _Env __join_env(_Env&& __env) noexcept {
      return (_Env&&) __env;
    }

    template <class _Env, class _Base>
    __env_join<_Env, _Base> __join_env(_Env&& __env, _Base&& __base) noexcept {
      static_assert(!same_as<__decay_t<_Env>, no_env>);
      return {{{(_Base&&) __base}}, (_Env&&) __env};
    }

    template <class _Base>
    _Base __join_env(empty_env, _Base&& __base) noexcept {
      return (_Base&&) __base;
    }

    template <class _Env>
    _Env __join_env(_Env&& __env, empty_env) noexcept {
      static_assert(!same_as<__decay_t<_Env>, no_env>);
      return (_Env&&) __env;
    }

    inline empty_env __join_env(empty_env, empty_env) noexcept {
      return {};
    }

    template <class _Env>
    no_env __join_env(_Env&&, no_env) noexcept
      requires true
    {
      static_assert(!same_as<__decay_t<_Env>, no_env>);
      return {};
    }

    template <class _Env0, class _Env1, class _Env2, class... _Envs>
    auto __join_env(_Env0&& __env0, _Env1&& __env1, _Env2&& __env2, _Envs&&... __envs) noexcept {
      return __env::__join_env(
        (_Env0&&) __env0,
        __env::__join_env(
          (_Env1&&) __env1, __env::__join_env((_Env2&&) __env2, (_Envs&&) __envs...)));
    }

    template <class... _Envs>
    using __env_join_t = decltype(__env::__join_env(__declval<_Envs>()...));

    // To be kept in sync with the promise type used in __connect_awaitable
    template <class _Env>
    struct __env_promise {
      template <class _Ty>
      _Ty&& await_transform(_Ty&& __value) noexcept {
        return (_Ty&&) __value;
      }

      template <class _Ty>
        requires tag_invocable<as_awaitable_t, _Ty, __env_promise&>
      auto await_transform(_Ty&& __value) //
        noexcept(nothrow_tag_invocable<as_awaitable_t, _Ty, __env_promise&>)
          -> tag_invoke_result_t<as_awaitable_t, _Ty, __env_promise&> {
        return tag_invoke(as_awaitable, (_Ty&&) __value, *this);
      }

      friend auto tag_invoke(get_env_t, const __env_promise&) noexcept -> const _Env&;
    };

    template <class _Tag, __nothrow_move_constructible _Value>
    constexpr auto __with_(_Tag, _Value __val) noexcept {
      return __env_fn{
        [__val = std::move(__val)](_Tag) noexcept(__nothrow_copy_constructible<_Value>) {
          return __val;
        }};
    }

    template <class _Tag>
    __env_fn<__deleted<_Tag>> __with_(_Tag) noexcept {
      return {};
    }

    template <class... _Ts>
    using __with = decltype(__env::__with_(__declval<_Ts>()...));

    // For making an environment from key/value pairs and optionally
    // another environment.
    struct __make_env_t {
      template <__nothrow_move_constructible _Base, __nothrow_move_constructible _Env>
      auto operator()(_Base&& __base, _Env&& __env) const noexcept -> __env_join_t<_Env, _Base> {
        return stdexec::__env::__join_env((_Env&&) __env, (_Base&&) __base);
      }

      template <__nothrow_move_constructible _Env>
      _Env operator()(_Env&& __env) const noexcept {
        return (_Env&&) __env;
      }
    };

    // For getting an evaluation environment from a receiver
    struct get_env_t {
      template <class _EnvProvider>
        requires tag_invocable<get_env_t, const _EnvProvider&>
      constexpr auto operator()(const _EnvProvider& __with_env) const noexcept
        -> tag_invoke_result_t<get_env_t, const _EnvProvider&> {
        static_assert(queryable<tag_invoke_result_t<get_env_t, const _EnvProvider&> >);
        static_assert(nothrow_tag_invocable<get_env_t, const _EnvProvider&>);
        return tag_invoke(*this, __with_env);
      }

      // NOT TO SPEC: The overload below checks the non-standard
      // enable_sender to determine whether to provide backwards
      // compatible behavior for R5 version sender types. When we
      // deprecate R5 support, we can bring this overload in line with
      // P2300R7.
      template <class _EnvProvider>
      constexpr decltype(auto) operator()(const _EnvProvider& __with_env) const noexcept {
        if constexpr (!enable_sender<_EnvProvider>) {
          return __with_env;
        } else {
          return empty_env{};
        }
      }
    };
  } // namespace __env

  using __env::no_env;
  using __env::empty_env;
  using __empty_env [[deprecated("Please use stdexec::empty_env now.")]] = empty_env;

  using __env::__with;
  using __env::__with_;
  using __env::__env_promise;
  using no_env_promise = __env_promise<no_env>;

  inline constexpr __env::__make_env_t __make_env{};
  inline constexpr __env::get_env_t get_env{};

  template <class... _Ts>
  using __make_env_t = decltype(__make_env(__declval<_Ts>()...));

#if STDEXEC_LEGACY_R5_CONCEPTS()
  using __default_env = no_env;
#else
  using __default_env = empty_env;
#endif

  template <class _EnvProvider>
  concept environment_provider = //
    requires(_EnvProvider& __ep) {
      { get_env(std::as_const(__ep)) } -> queryable;
      // NOT TO SPEC: Remove the following line when we deprecate all
      // R5 entities.
      { get_env(std::as_const(__ep)) } -> __none_of<no_env, void>;
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  namespace __receivers {
    struct set_value_t {
      template <class _Fn, class... _Args>
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class... _As>
        requires tag_invocable<set_value_t, _Receiver, _As...>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        void
        operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
        static_assert(nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) tag_invoke(set_value_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
      }
    };

    struct set_error_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 1)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class _Error>
        requires tag_invocable<set_error_t, _Receiver, _Error>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        void
        operator()(_Receiver&& __rcvr, _Error&& __err) const noexcept {
        static_assert(nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
        (void) tag_invoke(set_error_t{}, (_Receiver&&) __rcvr, (_Error&&) __err);
      }
    };

    struct set_stopped_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 0)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver>
        requires tag_invocable<set_stopped_t, _Receiver>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        void
        operator()(_Receiver&& __rcvr) const noexcept {
        static_assert(nothrow_tag_invocable<set_stopped_t, _Receiver>);
        (void) tag_invoke(set_stopped_t{}, (_Receiver&&) __rcvr);
      }
    };
  } // namespace __receivers

  using __receivers::set_value_t;
  using __receivers::set_error_t;
  using __receivers::set_stopped_t;
  inline constexpr set_value_t set_value{};
  inline constexpr set_error_t set_error{};
  inline constexpr set_stopped_t set_stopped{};

  inline constexpr struct __try_call_t {
    template <class _Receiver, class _Fun, class... _Args>
      requires __callable<_Fun, _Args...>
    void operator()(_Receiver&& __rcvr, _Fun __fun, _Args&&... __args) const noexcept {
      if constexpr (__nothrow_callable<_Fun, _Args...>) {
        ((_Fun&&) __fun)((_Args&&) __args...);
      } else {
        try {
          ((_Fun&&) __fun)((_Args&&) __args...);
        } catch (...) {
          set_error((_Receiver&&) __rcvr, std::current_exception());
        }
      }
    }
  } __try_call{};

  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __compl_sigs {
#if STDEXEC_NVHPC()
    template <class _Ty = __q<__types>, class... _Args>
    __types<__minvoke<_Ty, _Args...>> __test(set_value_t (*)(_Args...), set_value_t = {}, _Ty = {});
    template <class _Ty = __q<__types>, class _Error>
    __types<__minvoke<_Ty, _Error>> __test(set_error_t (*)(_Error), set_error_t = {}, _Ty = {});
    template <class _Ty = __q<__types>>
    __types<__minvoke<_Ty>> __test(set_stopped_t (*)(), set_stopped_t = {}, _Ty = {});
    __types<> __test(__ignore, __ignore, __ignore = {});

    template <class _Sig>
    inline constexpr bool __is_compl_sig = false;
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;

#else

    template <same_as<set_value_t> _Tag, class _Ty = __q<__types>, class... _Args>
    __types<__minvoke<_Ty, _Args...>> __test(_Tag (*)(_Args...));
    template <same_as<set_error_t> _Tag, class _Ty = __q<__types>, class _Error>
    __types<__minvoke<_Ty, _Error>> __test(_Tag (*)(_Error));
    template <same_as<set_stopped_t> _Tag, class _Ty = __q<__types>>
    __types<__minvoke<_Ty>> __test(_Tag (*)());
    template <class, class = void>
    __types<> __test(...);
    template <class _Tag, class _Ty = void, class... _Args>
    void __test(_Tag (*)(_Args...) noexcept) = delete;
#endif

    // BUGBUG not to spec!
    struct __dependent {
#if !STDEXEC_STD_NO_COROUTINES_
      bool await_ready();
      template <class _Env>
      void await_suspend(__coro::coroutine_handle<__env_promise<_Env>>);
      __dependent await_resume();
#endif
    };

#if STDEXEC_NVHPC()
    template <class _Sig>
    concept __completion_signature = __compl_sigs::__is_compl_sig<_Sig>;

    template <class _Sig, class _Tag, class _Ty = __q<__types>>
    using __signal_args_t = decltype(__compl_sigs::__test((_Sig*) nullptr, _Tag{}, _Ty{}));
#else
    template <class _Sig>
    concept __completion_signature = __typename<decltype(__compl_sigs::__test((_Sig*) nullptr))>;

    template <class _Sig, class _Tag, class _Ty = __q<__types>>
    using __signal_args_t = decltype(__compl_sigs::__test<_Tag, _Ty>((_Sig*) nullptr));
#endif
  } // namespace __compl_sigs

  using __compl_sigs::__completion_signature;

  template <same_as<no_env>>
  using dependent_completion_signatures = __compl_sigs::__dependent;

  template <__compl_sigs::__completion_signature... _Sigs>
  struct completion_signatures {
    // Uncomment this to see where completion_signatures is
    // erroneously getting instantiated:
    //static_assert(sizeof...(_Sigs) == -1u);
  };

  namespace __compl_sigs {
    template <class _TaggedTuple, __completion_tag _Tag, class... _Ts>
    auto __as_tagged_tuple_(_Tag (*)(_Ts...), _TaggedTuple*)
      -> __mconst<__minvoke<_TaggedTuple, _Tag, _Ts...>>;

    template <class _Sig, class _TaggedTuple>
    using __as_tagged_tuple =
      decltype(__compl_sigs::__as_tagged_tuple_((_Sig*) nullptr, (_TaggedTuple*) nullptr));

    template <class _TaggedTuple, class _Variant, class... _Sigs>
    auto __for_all_sigs_(completion_signatures<_Sigs...>*, _TaggedTuple*, _Variant*)
      -> __mconst< __minvoke< _Variant, __minvoke<__as_tagged_tuple<_Sigs, _TaggedTuple>>...>>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __for_all_sigs =                      //
      __minvoke<                                //
        decltype(__compl_sigs::__for_all_sigs_( //
          (_Completions*) nullptr,
          (_TaggedTuple*) nullptr,
          (_Variant*) nullptr))>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __maybe_for_all_sigs = __meval<__for_all_sigs, _Completions, _TaggedTuple, _Variant>;
  } // namespace __compl_sigs

  template <class _Ty>
  concept __is_completion_signatures = __is_instance_of<_Ty, completion_signatures>;

  template <class...>
  auto __concat_completion_signatures_impl() //
    -> dependent_completion_signatures<no_env>;

  template <__is_completion_signatures... _Completions>
  auto __concat_completion_signatures_impl()
    -> __minvoke< __mconcat<__munique<__q<completion_signatures>>>, _Completions...>;

  template <class... _Completions>
  using __concat_completion_signatures_impl_t = //
    decltype(__concat_completion_signatures_impl<_Completions...>());

  template <class... _Completions>
  struct __concat_completion_signatures_ {
    using __t = __meval<__concat_completion_signatures_impl_t, _Completions...>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_t = __t<__concat_completion_signatures_<_Completions...>>;

  template <class _Completions, class _Env>
  inline constexpr bool __expecting_completion_signatures = false;

  template <class... _Sigs, class _Env>
  inline constexpr bool __expecting_completion_signatures<completion_signatures<_Sigs...>, _Env> =
    true;

  template <>
  inline constexpr bool
    __expecting_completion_signatures<dependent_completion_signatures<no_env>, no_env> = true;

  template <class _Completions, class _Env>
  concept __valid_completion_signatures = __expecting_completion_signatures<_Completions, _Env>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template <class _Receiver>
  struct _WITH_RECEIVER_ { };

  template <class _Sig>
  struct _MISSING_COMPLETION_SIGNAL_ { };

  template <class _Receiver, class _Tag, class... _Args>
  auto __try_completion(_Tag (*)(_Args...))
    -> __mexception<_MISSING_COMPLETION_SIGNAL_<_Tag(_Args...)>, _WITH_RECEIVER_<_Receiver>>;

  template <class _Receiver, class _Tag, class... _Args>
    requires nothrow_tag_invocable<_Tag, _Receiver, _Args...>
  __msuccess __try_completion(_Tag (*)(_Args...));

  template <class _Receiver, class... _Sigs>
  auto __try_completions(completion_signatures<_Sigs...>*)
    -> decltype((__msuccess(), ..., stdexec::__try_completion<_Receiver>((_Sigs*) nullptr)));

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  struct __receiver_base { };

  template <class _Receiver>
  concept __enable_receiver = //
    requires { typename _Receiver::is_receiver; } || STDEXEC_IS_BASE_OF(__receiver_base, _Receiver);

  template <class _Receiver>
  inline constexpr bool enable_receiver = __enable_receiver<_Receiver>; // NOT TO SPEC

  //   NOT TO SPEC:
  //   As we upgrade the receiver related entities from R5 to R7,
  //   we allow types that do not yet satisfy enable_receiver to
  //   still satisfy the receiver concept if the type provides an
  //   explicit get_env. All R5 receivers provided an explicit get_env,
  //   so this is backwards compatible.
  template <class _Receiver>
  concept __receiver_r5_or_r7 = //
    enable_receiver<_Receiver>  //
    || tag_invocable<get_env_t, _Receiver>;

  template <class _Receiver>
  concept __receiver = //
    // Nested requirement here is to make this an atomic constraint
    requires { requires __receiver_r5_or_r7<__decay_t<_Receiver>>; };

  template <class _Receiver>
  concept receiver =
    __receiver<_Receiver> &&                     //
    environment_provider<__cref_t<_Receiver>> && //
    move_constructible<__decay_t<_Receiver>> &&  //
    constructible_from<__decay_t<_Receiver>, _Receiver>;

  template <class _Receiver, class _Completions>
  concept receiver_of =    //
    receiver<_Receiver> && //
    requires(_Completions* __completions) {
      { stdexec::__try_completions<__decay_t<_Receiver>>(__completions) } -> __ok;
    };

  template <class _Receiver, class _Sender>
  concept __receiver_from =
    receiver_of< _Receiver, __completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;

  /////////////////////////////////////////////////////////////////////////////
  // Some utilities for debugging senders
  namespace __debug {
    struct __is_debug_env_t {
      friend constexpr bool tag_invoke(forwarding_query_t, const __is_debug_env_t&) noexcept {
        return true;
      }
      template <class _Env>
        requires tag_invocable<__is_debug_env_t, const _Env&>
      auto operator()(const _Env&) const noexcept
        -> tag_invoke_result_t<__is_debug_env_t, const _Env&>;
    };
    template <class _Env>
    using __debug_env_t = __make_env_t<_Env, __with<__is_debug_env_t, bool>>;

    template <class _Env>
    concept __is_debug_env = tag_invocable<__debug::__is_debug_env_t, _Env>;

    struct __completion_signatures { };

    template <class _Sig>
    extern int __normalize_sig;

    template <class _Tag, class... _Args>
    extern _Tag (*__normalize_sig<_Tag(_Args...)>)(_Args&&...);

    template <class _Sig>
    using __normalize_sig_t = decltype(__normalize_sig<_Sig>);

    template <class... _Sigs>
    struct __valid_completions {
      template <derived_from<__valid_completions> _Self, class _Tag, class... _Args>
        requires __one_of<_Tag (*)(_Args&&...), _Sigs...>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        friend void
        tag_invoke(_Tag, _Self&&, _Args&&...) noexcept {
        STDEXEC_TERMINATE();
      }
    };

    template <class _CvrefSenderId, class _Env, class _Completions>
    struct __debug_receiver {
      using __t = __debug_receiver;
      using __id = __debug_receiver;
      using is_receiver = void;
    };

    template <class _CvrefSenderId, class _Env, class... _Sigs>
    struct __debug_receiver<_CvrefSenderId, _Env, completion_signatures<_Sigs...>> //
      : __valid_completions<__normalize_sig_t<_Sigs>...> {
      using __t = __debug_receiver;
      using __id = __debug_receiver;
      using is_receiver = void;

      template <same_as<get_env_t> _Tag>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        friend __debug_env_t<_Env>
        tag_invoke(_Tag, __debug_receiver) noexcept {
        STDEXEC_TERMINATE();
      }
    };

    struct _COMPLETION_SIGNATURES_MISMATCH_ { };

    template <class _Sig>
    struct _COMPLETION_SIGNATURE_ { };

    template <class... _Sigs>
    struct _IS_NOT_ONE_OF_ { };

    template <class _Sender>
    struct _SIGNAL_SENT_BY_SENDER_ { };

    template <class _Warning>
    [[deprecated(
      "The sender claims to send a particular set of completions,"
      " but in actual fact it completes with a result that is not"
      " one of the declared completion signatures.")]] STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
      void
      _ATTENTION_() noexcept {
    }

    template <class _Sig>
    struct __invalid_completion {
      struct __t {
        template <class _CvrefSenderId, class _Env, class... _Sigs>
        __t(__debug_receiver<_CvrefSenderId, _Env, completion_signatures<_Sigs...>>&&) noexcept {
          using _SenderId = __decay_t<_CvrefSenderId>;
          using _Sender = stdexec::__t<_SenderId>;
          using _What = //
            _WARNING_<  //
              _COMPLETION_SIGNATURES_MISMATCH_,
              _COMPLETION_SIGNATURE_<_Sig>,
              _IS_NOT_ONE_OF_<_Sigs...>,
              _SIGNAL_SENT_BY_SENDER_<_Sender>>;
          __debug::_ATTENTION_<_What>();
        }
      };
    };

    template <__completion_tag _Tag, class... _Args>
    STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
      void
      tag_invoke(_Tag, __t<__invalid_completion<_Tag(_Args...)>>, _Args&&...) noexcept {
    }

    struct __debug_operation {
      template <same_as<start_t> _Tag>
      friend void tag_invoke(_Tag, __debug_operation&) noexcept {
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // `__debug_sender`
    // ===============
    //
    // Understanding why a particular sender doesn't connect to a particular
    // receiver is nigh impossible in the current design due to limitations in
    // how the compiler reports overload resolution failure in the presence of
    // constraints. `__debug_sender` is a utility to assist with the process. It
    // gives you the deep template instantiation backtrace that you need to
    // understand where in a chain of senders the problem is occurring.
    //
    // ```c++
    // template <class _Sigs, class _Env = empty_env, class _Sender>
    //   void __debug_sender(_Sender&& __sndr, _Env = {});
    //
    // template <class _Env = empty_env, class _Sender>
    //   void __debug_sender(_Sender&& __sndr, _Env = {});
    // ```
    //
    // **Usage:**
    //
    // To find out where in a chain of senders a sender is failing to connect
    // to a receiver, pass it to `__debug_sender`, optionally with an
    // environment argument; e.g. `__debug_sender(sndr [, env])`
    //
    // To find out why a sender will not connect to a receiver of a particular
    // signature, specify the set of completion signatures as an explicit template
    // argument that names an instantiation of `completion_signatures`; e.g.:
    // `__debug_sender<completion_signatures<set_value_t(int)>>(sndr [, env])`.
    //
    // **How it works:**
    //
    // The `__debug_sender` function `connect`'s the sender to a
    // `__debug_receiver`, whose environment is augmented with a special
    // `__is_debug_env_t` query. An additional fall-back overload is added to
    // the `connect` CPO that recognizes receivers whose environments respond to
    // that query and lets them through. Then in a non-immediate context, it
    // looks for a `tag_invoke(connect_t...)` overload for the input sender and
    // receiver. This will recurse until it hits the `tag_invoke` call that is
    // causing the failure.
    //
    // At least with clang, this gives me a nice backtrace, at the bottom of
    // which is the faulty `tag_invoke` overload with a mention of the
    // constraint that failed.
    template <class _Sigs, class _Env = empty_env, class _Sender>
    void __debug_sender(_Sender&& __sndr, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env> && !same_as<_Env, no_env>) {
        if (sizeof(_Sender) == ~0) { // never true
          using _Receiver = __debug_receiver<__cvref_id<_Sender>, _Env, _Sigs>;
          using _Operation = connect_result_t<_Sender, _Receiver>;
          //static_assert(receiver_of<_Receiver, _Sigs>);
          if constexpr (!same_as<_Operation, __debug_operation>) {
            auto __op = connect((_Sender&&) __sndr, _Receiver{});
            start(__op);
          }
        }
      }
    }

    template <class _Env = empty_env, class _Sender>
    void __debug_sender(_Sender&& __sndr, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env> && !same_as<_Env, no_env>) {
        if (sizeof(_Sender) == ~0) { // never true
          using _Sigs = __completion_signatures_of_t<_Sender, __debug_env_t<_Env>>;
          if constexpr (!same_as<_Sigs, __debug::__completion_signatures>) {
            using _Receiver = __debug_receiver<__cvref_id<_Sender>, _Env, _Sigs>;
            using _Operation = connect_result_t<_Sender, _Receiver>;
            //static_assert(receiver_of<_Receiver, _Sigs>);
            if constexpr (!same_as<_Operation, __debug_operation>) {
              auto __op = connect((_Sender&&) __sndr, _Receiver{});
              start(__op);
            }
          }
        }
      }
    }
  } // namespace __debug

  using __debug::__is_debug_env;
  using __debug::__debug_sender;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.sndtraits]
  namespace __get_completion_signatures {
    template <class _Sender, class _Env>
    concept __r7_style_sender = same_as<_Env, no_env> && enable_sender<__decay_t<_Sender>>;

    template <class _Sender, class _Env>
    concept __with_tag_invoke =
      __valid<tag_invoke_result_t, get_completion_signatures_t, _Sender, _Env>;

    template <class _Sender, class...>
    using __member_alias_t = typename __decay_t<_Sender>::completion_signatures;

    template <class _Sender>
    concept __with_member_alias = __valid<__member_alias_t, _Sender>;

    struct get_completion_signatures_t {
      template <class _Sender, class _Env>
      static auto __impl() {
        static_assert(STDEXEC_LEGACY_R5_CONCEPTS() || !same_as<_Env, no_env>);
        static_assert(sizeof(_Sender), "Incomplete type used with get_completion_signatures");
        static_assert(sizeof(_Env), "Incomplete type used with get_completion_signatures");
        if constexpr (__with_tag_invoke<_Sender, _Env>) {
          using _Result = tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env>;
          if constexpr (same_as<_Env, no_env> && __merror<_Result>) {
            return (dependent_completion_signatures<no_env>(*)()) nullptr;
          } else {
            return (_Result(*)()) nullptr;
          }
        } else if constexpr (__with_member_alias<_Sender>) {
          return (__member_alias_t<_Sender, _Env>(*)()) nullptr;
        } else if constexpr (__awaitable<_Sender, __env_promise<_Env>>) {
          using _Result = __await_result_t<_Sender, __env_promise<_Env>>;
          if constexpr (same_as<_Result, dependent_completion_signatures<no_env>>) {
            return (dependent_completion_signatures<no_env>(*)()) nullptr;
          } else {
            return (completion_signatures<
                    // set_value_t() or set_value_t(T)
                    __minvoke<__remove<void, __qf<set_value_t>>, _Result>,
                    set_error_t(std::exception_ptr),
                    set_stopped_t()>(*)()) nullptr;
          }
        } else
#if STDEXEC_LEGACY_R5_CONCEPTS()
          if constexpr (__r7_style_sender<_Sender, _Env>) {
          return (dependent_completion_signatures<no_env>(*)()) nullptr;
        } else
#endif
          if constexpr (__is_debug_env<_Env>) {
          using __tag_invoke::tag_invoke;
          // This ought to cause a hard error that indicates where the problem is.
          using _Completions
            [[maybe_unused]] = tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env>;
          return (__debug::__completion_signatures(*)()) nullptr;
        } else {
          return (void (*)()) nullptr;
        }
      }

      template <class _Sender, class _Env = __default_env>
        requires(
          __with_tag_invoke<_Sender, _Env> ||          //
          __with_member_alias<_Sender> ||              //
          __awaitable<_Sender, __env_promise<_Env>> || //
#if STDEXEC_LEGACY_R5_CONCEPTS()                       //
          __r7_style_sender<_Sender, _Env> ||          //
#endif                                                 //
          __is_debug_env<_Env>)                        //
      constexpr auto operator()(_Sender&&, const _Env&) const noexcept
        -> decltype(__impl<_Sender, _Env>()()) {
        return {};
      }
    };
  } // namespace __get_completion_signatures

  using __get_completion_signatures::get_completion_signatures_t;
  inline constexpr get_completion_signatures_t get_completion_signatures{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  template <class _Sender>
  concept __enable_sender =                      //
    requires { typename _Sender::is_sender; } || //
    __awaitable<_Sender, __env_promise<empty_env>>;

  template <class _Sender>
  inline constexpr bool enable_sender = __enable_sender<_Sender>;

  // NOT TO SPEC (YET)
#if !STDEXEC_LEGACY_R5_CONCEPTS()
  // Here is the R7 sender concepts, not yet enabled.
  template <class _Sender>
  concept sender =                             //
    enable_sender<__decay_t<_Sender>> &&       //
    environment_provider<__cref_t<_Sender>> && //
    move_constructible<__decay_t<_Sender>> &&  //
    constructible_from<__decay_t<_Sender>, _Sender>;

  template <class _Sender, class _Env = empty_env>
  concept sender_in =  //
    sender<_Sender> && //
    requires(_Sender&& __sndr, _Env&& __env) {
      {
        get_completion_signatures((_Sender&&) __sndr, (_Env&&) __env)
      } -> __valid_completion_signatures<_Env>;
    };

  template <class _Sender, class _Env = empty_env>
    requires sender_in<_Sender, _Env>
  using completion_signatures_of_t = __completion_signatures_of_t<_Sender, _Env>;

#else

  template <class _Sender, class _Env = no_env>
  concept __sender = //
    requires(_Sender&& __sndr, _Env&& __env) {
      get_completion_signatures((_Sender&&) __sndr, (_Env&&) __env);
    } && //
    __valid_completion_signatures<__completion_signatures_of_t<_Sender, _Env>, _Env>;

  template <class _Sender, class _Env = no_env>
  concept sender =
    // NOT TO SPEC
    // The sender related concepts are temporarily "in flight" being
    // upgraded from P2300R5 to the get_env / enable_sender aware version
    // in P2300R7.
    __sender<_Sender> &&                       //
    __sender<_Sender, _Env> &&                 //
    environment_provider<__cref_t<_Sender>> && //
    move_constructible<__decay_t<_Sender>> &&  //
    constructible_from<__decay_t<_Sender>, _Sender>;

  template <class _Sender, class _Env = empty_env>
  concept sender_in =          //
    __sender<_Sender, _Env> && //
    sender<_Sender, _Env>;

  // __checked_completion_signatures is for catching logic bugs in a typed
  // sender's metadata. If sender<S> and sender_in<S, Ctx> are both true, then they
  // had better report the same metadata. This completion signatures wrapper
  // enforces that at compile time.
  template <class _Sender, class _Env>
  auto __checked_completion_signatures(_Sender&& __sndr, const _Env& __env) noexcept {
    using _WithEnv = __completion_signatures_of_t<_Sender, _Env>;
    using _WithoutEnv = __completion_signatures_of_t<_Sender, no_env>;
    static_assert(__one_of< _WithoutEnv, _WithEnv, dependent_completion_signatures<no_env>>);
    stdexec::__debug_sender<_WithEnv>((_Sender&&) __sndr, __env);
    return _WithEnv{};
  }

  template <class _Sender, class _Env = no_env>
    requires sender_in<_Sender, _Env>
  using completion_signatures_of_t =
    decltype(stdexec::__checked_completion_signatures(__declval<_Sender>(), __declval<_Env>()));
#endif

  struct __not_a_variant {
    __not_a_variant() = delete;
  };
  template <class... _Ts>
  using __variant = //
    __minvoke<
      __if_c<
        sizeof...(_Ts) != 0,
        __transform<__q<__decay_t>, __munique<__q<std::variant>>>,
        __mconst<__not_a_variant>>,
      _Ts...>;

  using __nullable_variant_t = __munique<__mbind_front_q<std::variant, std::monostate>>;

  template <class... _Ts>
  using __decayed_tuple = __meval<std::tuple, __decay_t<_Ts>...>;

  template <class _Tag, class _Tuple>
  struct __select_completions_for {
    template <same_as<_Tag> _Tag2, class... _Args>
    using __f = __minvoke<_Tag2, _Tuple, _Args...>;
  };

  template <class _Tuple>
  struct __invoke_completions {
    template <class _Tag, class... _Args>
    using __f = __minvoke<_Tag, _Tuple, _Args...>;
  };

  template <class _Tag, class _Tuple>
  using __select_completions_for_or = //
    __with_default< __select_completions_for<_Tag, _Tuple>, __>;

  template <class _Tag, class _Completions>
  using __only_gather_signal = //
    __compl_sigs::__maybe_for_all_sigs<
      _Completions,
      __select_completions_for_or<_Tag, __qf<_Tag>>,
      __remove<__, __q<completion_signatures>>>;

  template <class _Tag, class _Completions, class _Tuple, class _Variant>
  using __gather_signal = //
    __compl_sigs::__maybe_for_all_sigs<
      __only_gather_signal<_Tag, _Completions>,
      __invoke_completions<_Tuple>,
      _Variant>;

  template <class _Tag, class _Sender, class _Env, class _Tuple, class _Variant>
  using __gather_completions_for = //
    __meval<                       //
      __gather_signal,
      _Tag,
      __completion_signatures_of_t<_Sender, _Env>,
      _Tuple,
      _Variant>;

  template <                             //
    class _Sender,                       //
    class _Env = __default_env,          //
    class _Tuple = __q<__decayed_tuple>, //
    class _Variant = __q<__variant>>
  using __try_value_types_of_t = //
    __gather_completions_for<set_value_t, _Sender, _Env, _Tuple, _Variant>;

  template <                             //
    class _Sender,                       //
    class _Env = __default_env,          //
    class _Tuple = __q<__decayed_tuple>, //
    class _Variant = __q<__variant>>
    requires sender_in<_Sender, _Env>
  using __value_types_of_t = //
    __msuccess_or_t<__try_value_types_of_t<_Sender, _Env, _Tuple, _Variant>>;

  template <class _Sender, class _Env = __default_env, class _Variant = __q<__variant>>
  using __try_error_types_of_t =
    __gather_completions_for<set_error_t, _Sender, _Env, __q<__midentity>, _Variant>;

  template <class _Sender, class _Env = __default_env, class _Variant = __q<__variant>>
    requires sender_in<_Sender, _Env>
  using __error_types_of_t = __msuccess_or_t<__try_error_types_of_t<_Sender, _Env, _Variant>>;

  template <                                            //
    class _Sender,                                      //
    class _Env = __default_env,                         //
    template <class...> class _Tuple = __decayed_tuple, //
    template <class...> class _Variant = __variant>
    requires sender_in<_Sender, _Env>
  using value_types_of_t = __value_types_of_t<_Sender, _Env, __q<_Tuple>, __q<_Variant>>;

  template <class _Sender, class _Env = __default_env, template <class...> class _Variant = __variant>
    requires sender_in<_Sender, _Env>
  using error_types_of_t = __error_types_of_t<_Sender, _Env, __q<_Variant>>;

  template <class _Tag, class _Sender, class _Env = __default_env>
  using __try_count_of = //
    __compl_sigs::__maybe_for_all_sigs<
      __completion_signatures_of_t<_Sender, _Env>,
      __q<__mfront>,
      __mcount<_Tag>>;

  template <class _Tag, class _Sender, class _Env = __default_env>
    requires sender_in<_Sender, _Env>
  using __count_of = __msuccess_or_t<__try_count_of<_Tag, _Sender, _Env>>;

  template <class _Tag, class _Sender, class _Env = __default_env>
    requires __valid<__count_of, _Tag, _Sender, _Env>
  inline constexpr bool __sends = (__v<__count_of<_Tag, _Sender, _Env>> != 0);

  template <class _Sender, class _Env = __default_env>
    requires __valid<__count_of, set_stopped_t, _Sender, _Env>
  inline constexpr bool sends_stopped = __sends<set_stopped_t, _Sender, _Env>;

  template <class _Sender, class _Env = __default_env>
  using __single_sender_value_t =
    __value_types_of_t<_Sender, _Env, __msingle_or<void>, __q<__msingle>>;

  template <class _Sender, class _Env = __default_env>
  using __single_value_variant_sender_t = value_types_of_t<_Sender, _Env, __types, __msingle>;

  template <class _Sender, class _Env = __default_env>
  concept __single_typed_sender =
    sender_in<_Sender, _Env> && __valid<__single_sender_value_t, _Sender, _Env>;

  template <class _Sender, class _Env = __default_env>
  concept __single_value_variant_sender =
    sender_in<_Sender, _Env> && __valid<__single_value_variant_sender_t, _Sender, _Env>;

  template <class... Errs>
  using __nofail = __mbool<sizeof...(Errs) == 0>;

  template <class _Sender, class _Env = __default_env>
  concept __nofail_sender =
    sender_in<_Sender, _Env> && (__v<error_types_of_t<_Sender, _Env, __nofail>>);

  /////////////////////////////////////////////////////////////////////////////
  namespace __compl_sigs {
    template <class... _Args>
    using __default_set_value = completion_signatures<set_value_t(_Args...)>;

    template <class _Error>
    using __default_set_error = completion_signatures<set_error_t(_Error)>;

    template <__is_completion_signatures... _Sigs>
    using __ensure_concat_ = __minvoke<__mconcat<__q<completion_signatures>>, _Sigs...>;

    template <class... _Sigs>
    using __ensure_concat = __mtry_eval<__ensure_concat_, _Sigs...>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __compl_sigs_impl = //
      __concat_completion_signatures_t<
        _Sigs,
        __mtry_eval<__try_value_types_of_t, _Sender, _Env, _SetVal, __q<__ensure_concat>>,
        __mtry_eval<__try_error_types_of_t, _Sender, _Env, __transform<_SetErr, __q<__ensure_concat>>>,
        __if<__try_count_of<set_stopped_t, _Sender, _Env>, _SetStp, completion_signatures<>>>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      requires __valid<__compl_sigs_impl, _Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>
    extern __compl_sigs_impl<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp> __compl_sigs_v;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __compl_sigs_t =
      decltype(__compl_sigs_v<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>);

    template <bool>
    struct __make_compl_sigs {
      template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      using __f = __compl_sigs_t<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>;
    };

    template <>
    struct __make_compl_sigs<true> {
      template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      using __f = //
        __msuccess_or_t<
          __compl_sigs_t<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>,
          dependent_completion_signatures<_Env>>;
    };

    template <                                                    //
      class _Sender,                                              //
      class _Env = __default_env,                                 //
      class _Sigs = completion_signatures<>,                      //
      class _SetValue = __q<__default_set_value>,                 //
      class _SetError = __q<__default_set_error>,                 //
      class _SetStopped = completion_signatures<set_stopped_t()>> //
    using __try_make_completion_signatures =                      //
      __minvoke<
        __make_compl_sigs<same_as<_Env, no_env>>,
        _Sender,
        _Env,
        _Sigs,
        _SetValue,
        _SetError,
        _SetStopped>;
  } // namespace __compl_sigs

  using __compl_sigs::__try_make_completion_signatures;

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC
  //
  // make_completion_signatures
  // ==========================
  //
  // `make_completion_signatures` takes a sender, and environment, and a bunch
  // of other template arguments for munging the completion signatures of a
  // sender in interesting ways.
  //
  //  ```c++
  //  template <class... Args>
  //    using __default_set_value = completion_signatures<set_value_t(Args...)>;
  //
  //  template <class Err>
  //    using __default_set_error = completion_signatures<set_error_t(Err)>;
  //
  //  template <
  //    sender Sndr,
  //    class Env = __default_env,
  //    class AddlSigs = completion_signatures<>,
  //    template <class...> class SetValue = __default_set_value,
  //    template <class> class SetError = __default_set_error,
  //    class SetStopped = completion_signatures<set_stopped_t()>>
  //      requires sender_in<Sndr, Env>
  //  using make_completion_signatures =
  //    completion_signatures< ... >;
  //  ```
  //
  //  * `SetValue` : an alias template that accepts a set of value types and
  //    returns an instance of `completion_signatures`.
  //  * `SetError` : an alias template that accepts an error types and returns a
  //    an instance of `completion_signatures`.
  //  * `SetStopped` : an instantiation of `completion_signatures` with a list
  //    of completion signatures `Sigs...` to the added to the list if the
  //    sender can complete with a stopped signal.
  //  * `AddlSigs` : an instantiation of `completion_signatures` with a list of
  //    completion signatures `Sigs...` to the added to the list
  //    unconditionally.
  //
  //  `make_completion_signatures` does the following:
  //  * Let `VCs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `value_types_of_t<Sndr, Env, SetValue,
  //    __typelist>`, and let `Vs...` be the concatenation of the packs that are
  //    template arguments to each `completion_signature` in `VCs...`.
  //  * Let `ECs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `error_types_of_t<Sndr, Env, __errorlist>`, where
  //    `__errorlist` is an alias template such that `__errorlist<Ts...>` names
  //    `__typelist<SetError<Ts>...>`, and let `Es...` by the concatenation of
  //    the packs that are the template arguments to each `completion_signature`
  //    in `ECs...`.
  //  * Let `Ss...` be an empty pack if `sends_stopped<Sndr, Env>` is
  //    `false`; otherwise, a pack containing the template arguments of the
  //    `completion_signatures` instantiation named by `SetStopped`.
  //  * Let `MoreSigs...` be a pack of the template arguments of the
  //    `completion_signatures` instantiation named by `AddlSigs`.
  //
  //  Then `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` names the type `completion_signatures< Sigs... >` where
  //  `Sigs...` is the unique set of types in `[Vs..., Es..., Ss...,
  //  MoreSigs...]`.
  //
  //  If any of the above type computations are ill-formed,
  //  `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` is an alias for an empty struct
  template <                                                                 //
    class _Sender,                                                           //
    class _Env = __default_env,                                              //
    __valid_completion_signatures<_Env> _Sigs = completion_signatures<>,     //
    template <class...> class _SetValue = __compl_sigs::__default_set_value, //
    template <class> class _SetError = __compl_sigs::__default_set_error,    //
    __valid_completion_signatures<_Env> _SetStopped = completion_signatures<set_stopped_t()>>
    requires sender_in<_Sender, _Env>
  using make_completion_signatures = //
    __msuccess_or_t<                 //
      __try_make_completion_signatures<
        _Sender,
        _Env,
        _Sigs,
        __q<_SetValue>,
        __q<_SetError>,
        _SetStopped>>;

  // Needed fairly often
  using __with_exception_ptr = completion_signatures<set_error_t(std::exception_ptr)>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  namespace __schedule {
    struct schedule_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_t, _Scheduler>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        auto
        operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>) {
        static_assert(sender<tag_invoke_result_t<schedule_t, _Scheduler>>);
        return tag_invoke(schedule_t{}, (_Scheduler&&) __sched);
      }

      friend constexpr bool tag_invoke(forwarding_query_t, schedule_t) {
        return false;
      }
    };
  }

  using __schedule::schedule_t;
  inline constexpr schedule_t schedule{};

  // NOT TO SPEC
  template <class _Tag, const auto& _Predicate>
  concept tag_category = //
    requires {
      typename __mbool<bool{_Predicate(_Tag{})}>;
      requires bool{_Predicate(_Tag{})};
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.schedulers]
  template <class _Scheduler>
  concept __has_schedule = //
    requires(_Scheduler&& __sched) {
      { schedule((_Scheduler&&) __sched) } -> sender;
    };

  template <class _Scheduler>
  concept __sender_has_completion_scheduler =
    requires(_Scheduler&& __sched, const get_completion_scheduler_t<set_value_t>&& __tag) {
      {
        tag_invoke(std::move(__tag), get_env(schedule((_Scheduler&&) __sched)))
      } -> same_as<__decay_t<_Scheduler>>;
    };

  template <class _Scheduler>
  concept scheduler =                                //
    __has_schedule<_Scheduler> &&                    //
    __sender_has_completion_scheduler<_Scheduler> && //
    equality_comparable<__decay_t<_Scheduler>> &&    //
    copy_constructible<__decay_t<_Scheduler>>;

  template <scheduler _Scheduler>
  using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  template <receiver _Receiver>
  using __current_scheduler_t = __call_result_t<get_scheduler_t, env_of_t<_Receiver>>;

  template <class _SchedulerProvider>
  concept __scheduler_provider = //
    requires(const _SchedulerProvider& __sp) {
      { get_scheduler(__sp) } -> scheduler;
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  namespace __start {
    struct start_t {
      template <class _Op>
        requires tag_invocable<start_t, _Op&>
      void operator()(_Op& __op) const noexcept {
        static_assert(nothrow_tag_invocable<start_t, _Op&>);
        (void) tag_invoke(start_t{}, __op);
      }
    };
  }

  using __start::start_t;
  inline constexpr start_t start{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template <class _Op>
  concept operation_state =  //
    destructible<_Op> &&     //
    std::is_object_v<_Op> && //
    requires(_Op& __op) {    //
      start(__op);
    };

#if !STDEXEC_STD_NO_COROUTINES_
  /////////////////////////////////////////////////////////////////////////////
  // __connect_awaitable_
  namespace __connect_awaitable_ {
    struct __promise_base {
      __coro::suspend_always initial_suspend() noexcept {
        return {};
      }

      [[noreturn]] __coro::suspend_always final_suspend() noexcept {
        std::terminate();
      }

      [[noreturn]] void unhandled_exception() noexcept {
        std::terminate();
      }

      [[noreturn]] void return_void() noexcept {
        std::terminate();
      }
    };

    struct __operation_base {
      __coro::coroutine_handle<> __coro_;

      explicit __operation_base(__coro::coroutine_handle<> __hcoro) noexcept
        : __coro_(__hcoro) {
      }

      __operation_base(__operation_base&& __other) noexcept
        : __coro_(std::exchange(__other.__coro_, {})) {
      }

      ~__operation_base() {
        if (__coro_)
          __coro_.destroy();
      }

      friend void tag_invoke(start_t, __operation_base& __self) noexcept {
        __self.__coro_.resume();
      }
    };

    template <class _ReceiverId>
    struct __promise;

    template <class _ReceiverId>
    struct __operation {
      struct __t : __operation_base {
        using promise_type = stdexec::__t<__promise<_ReceiverId>>;
        using __operation_base::__operation_base;
      };
    };

    template <class _ReceiverId>
    struct __promise {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __promise_base {
        using __id = __promise;

        explicit __t(auto&, _Receiver& __rcvr) noexcept
          : __rcvr_(__rcvr) {
        }

        __coro::coroutine_handle<> unhandled_stopped() noexcept {
          set_stopped((_Receiver&&) __rcvr_);
          // Returning noop_coroutine here causes the __connect_awaitable
          // coroutine to never resume past the point where it co_await's
          // the awaitable.
          return __coro::noop_coroutine();
        }

        stdexec::__t<__operation<_ReceiverId>> get_return_object() noexcept {
          return stdexec::__t<__operation<_ReceiverId>>{
            __coro::coroutine_handle<__t>::from_promise(*this)};
        }

        template <class _Awaitable>
        _Awaitable&& await_transform(_Awaitable&& __await) noexcept {
          return (_Awaitable&&) __await;
        }

        template <class _Awaitable>
          requires tag_invocable<as_awaitable_t, _Awaitable, __t&>
        auto await_transform(_Awaitable&& __await) //
          noexcept(nothrow_tag_invocable<as_awaitable_t, _Awaitable, __t&>)
            -> tag_invoke_result_t<as_awaitable_t, _Awaitable, __t&> {
          return tag_invoke(as_awaitable, (_Awaitable&&) __await, *this);
        }

        // Pass through the get_env receiver query
        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__rcvr_);
        }

        _Receiver& __rcvr_;
      };
    };

    template <receiver _Receiver>
    using __promise_t = __t<__promise<__id<_Receiver>>>;

    template <receiver _Receiver>
    using __operation_t = __t<__operation<__id<_Receiver>>>;

    struct __connect_awaitable_t {
     private:
      template <class _Fun, class... _Ts>
      static auto __co_call(_Fun __fun, _Ts&&... __as) noexcept {
        auto __fn = [&, __fun]() noexcept {
          __fun((_Ts&&) __as...);
        };

        struct __awaiter {
          decltype(__fn) __fn_;

          static constexpr bool await_ready() noexcept {
            return false;
          }

          void await_suspend(__coro::coroutine_handle<>) noexcept {
            __fn_();
          }

          [[noreturn]] void await_resume() noexcept {
            std::terminate();
          }
        };

        return __awaiter{__fn};
      }

      template <class _Awaitable, class _Receiver>
#if STDEXEC_GCC() && (__GNUC__ > 11)
      __attribute__((__used__))
#endif
      static __operation_t<_Receiver>
        __co_impl(_Awaitable __await, _Receiver __rcvr) {
        using __result_t = __await_result_t<_Awaitable, __promise_t<_Receiver>>;
        std::exception_ptr __eptr;
        try {
          if constexpr (same_as<__result_t, void>)
            co_await (co_await (_Awaitable&&) __await, __co_call(set_value, (_Receiver&&) __rcvr));
          else
            co_await __co_call(set_value, (_Receiver&&) __rcvr, co_await (_Awaitable&&) __await);
        } catch (...) {
          __eptr = std::current_exception();
        }
        co_await __co_call(set_error, (_Receiver&&) __rcvr, (std::exception_ptr&&) __eptr);
      }

      template <receiver _Receiver, class _Awaitable>
      using __completions_t = //
        completion_signatures<
          __minvoke< // set_value_t() or set_value_t(T)
            __remove<void, __qf<set_value_t>>,
            __await_result_t<_Awaitable, __promise_t<_Receiver>>>,
          set_error_t(std::exception_ptr),
          set_stopped_t()>;

     public:
      template <class _Receiver, __awaitable<__promise_t<_Receiver>> _Awaitable>
        requires receiver_of<_Receiver, __completions_t<_Receiver, _Awaitable>>
      __operation_t<_Receiver> operator()(_Awaitable&& __await, _Receiver __rcvr) const {
        return __co_impl((_Awaitable&&) __await, (_Receiver&&) __rcvr);
      }
    };
  } // namespace __connect_awaitable_

  using __connect_awaitable_::__connect_awaitable_t;
#else
  struct __connect_awaitable_t { };
#endif
  inline constexpr __connect_awaitable_t __connect_awaitable{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  namespace __connect {
    struct connect_t;

    template <class _Tp>
    STDEXEC_R5_SENDER_DEPRECATION_WARNING //
      void
      _PLEASE_UPDATE_YOUR_SENDER_TYPE() {
    }

    template <class _Tp>
    STDEXEC_R5_RECEIVER_DEPRECATION_WARNING //
      void
      _PLEASE_UPDATE_YOUR_RECEIVER_TYPE() {
    }

    template <class _Sender, class _Receiver>
    concept __connectable_with_tag_invoke =
      receiver<_Receiver> &&                     //
      sender_in<_Sender, env_of_t<_Receiver>> && //
      __receiver_from<_Receiver, _Sender> &&     //
      tag_invocable<connect_t, _Sender, _Receiver>;

    struct connect_t {
      template <class _Sender, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        // Report that 2300R5-style senders and receivers are deprecated:
        if constexpr (!enable_sender<__decay_t<_Sender>>)
          _PLEASE_UPDATE_YOUR_SENDER_TYPE<__decay_t<_Sender>>();

        if constexpr (!enable_receiver<__decay_t<_Receiver>>)
          _PLEASE_UPDATE_YOUR_RECEIVER_TYPE< __decay_t<_Receiver>>();

        if constexpr (__connectable_with_tag_invoke<_Sender, _Receiver>) {
          using _Result = tag_invoke_result_t<connect_t, _Sender, _Receiver>;
          constexpr bool _Nothrow = nothrow_tag_invocable<connect_t, _Sender, _Receiver>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__callable<__connect_awaitable_t, _Sender, _Receiver>) {
          using _Result = __call_result_t<__connect_awaitable_t, _Sender, _Receiver>;
          return static_cast<_Result (*)()>(nullptr);
        } else {
          return static_cast<__debug::__debug_operation (*)() noexcept>(nullptr);
        }
      }

      template <class _Sender, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sender, _Receiver>());

      template <sender _Sender, receiver _Receiver>
        requires __connectable_with_tag_invoke<_Sender, _Receiver>
              || __callable<__connect_awaitable_t, _Sender, _Receiver>
              || __is_debug_env<env_of_t<_Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sender, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sender, _Receiver>> {
        if constexpr (__connectable_with_tag_invoke<_Sender, _Receiver>) {
          static_assert(
            operation_state<tag_invoke_result_t<connect_t, _Sender, _Receiver>>,
            "stdexec::connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          return tag_invoke(connect_t{}, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
        } else if constexpr (__callable<__connect_awaitable_t, _Sender, _Receiver>) {
          return __connect_awaitable((_Sender&&) __sndr, (_Receiver&&) __rcvr);
        } else {
          // This should generate an instantiate backtrace that contains useful
          // debugging information.
          using __tag_invoke::tag_invoke;
          tag_invoke(*this, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
        }
      }

      friend constexpr bool tag_invoke(forwarding_query_t, connect_t) noexcept {
        return false;
      }
    };
  } // namespace __connect

  using __connect::connect_t;
  inline constexpr __connect::connect_t connect{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Sender, class _Receiver>
  concept sender_to =
    receiver<_Receiver> &&                     //
    __sender<_Sender, env_of_t<_Receiver>> &&  // NOT TO SPEC: for simpler diagnostics
    sender_in<_Sender, env_of_t<_Receiver>> && //
    __receiver_from<_Receiver, _Sender> &&     //
    requires(_Sender&& __sndr, _Receiver&& __rcvr) {
      connect((_Sender&&) __sndr, (_Receiver&&) __rcvr);
    };

  template <class _Tag, class... _Args>
  _Tag __tag_of_sig_(_Tag (*)(_Args...));
  template <class _Sig>
  using __tag_of_sig_t = decltype(stdexec::__tag_of_sig_((_Sig*) nullptr));

  template <class _Sender, class _SetSig, class _Env = __default_env>
  concept sender_of =
    sender_in<_Sender, _Env>
    && same_as<
      __types<_SetSig>,
      __gather_completions_for<
        __tag_of_sig_t<_SetSig>,
        _Sender,
        _Env,
        __qf<__tag_of_sig_t<_SetSig>>,
        __q<__types>>>;

  template <class _Fun, class _CPO, class _Sender, class... _As>
  concept __tag_invocable_with_completion_scheduler =
    __has_completion_scheduler<_Sender, _CPO>
    && tag_invocable<_Fun, __completion_scheduler_for<_Sender, _CPO>, _Sender, _As...>;

#if !STDEXEC_STD_NO_COROUTINES_
  /////////////////////////////////////////////////////////////////////////////
  // stdexec::as_awaitable [execution.coro_utils.as_awaitable]
  namespace __as_awaitable {
    struct __void { };
    template <class _Value>
    using __value_or_void_t = __if<std::is_same<_Value, void>, __void, _Value>;
    template <class _Value>
    using __expected_t =
      std::variant<std::monostate, __value_or_void_t<_Value>, std::exception_ptr>;

    template <class _Value>
    struct __receiver_base {
      using is_receiver = void;

      template <same_as<set_value_t> _Tag, class... _Us>
        requires constructible_from<__value_or_void_t<_Value>, _Us...>
      friend void tag_invoke(_Tag, __receiver_base&& __self, _Us&&... __us) noexcept {
        try {
          __self.__result_->template emplace<1>((_Us&&) __us...);
          __self.__continuation_.resume();
        } catch (...) {
          set_error((__receiver_base&&) __self, std::current_exception());
        }
      }

      template <same_as<set_error_t> _Tag, class _Error>
      friend void tag_invoke(_Tag, __receiver_base&& __self, _Error&& __err) noexcept {
        if constexpr (__decays_to<_Error, std::exception_ptr>)
          __self.__result_->template emplace<2>((_Error&&) __err);
        else if constexpr (__decays_to<_Error, std::error_code>)
          __self.__result_->template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
        else
          __self.__result_->template emplace<2>(std::make_exception_ptr((_Error&&) __err));
        __self.__continuation_.resume();
      }

      __expected_t<_Value>* __result_;
      __coro::coroutine_handle<> __continuation_;
    };

    template <class _PromiseId, class _Value>
    struct __receiver {
      using _Promise = stdexec::__t<_PromiseId>;

      struct __t : __receiver_base<_Value> {
        using __id = __receiver;

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
            __self.__continuation_.address());
          __coro::coroutine_handle<> __stopped_continuation =
            __continuation.promise().unhandled_stopped();
          __stopped_continuation.resume();
        }

        // Forward get_env query to the coroutine promise
        friend env_of_t<_Promise&> tag_invoke(get_env_t, const __t& __self) noexcept {
          auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
            __self.__continuation_.address());
          return get_env(__continuation.promise());
        }
      };
    };

    template <class _Sender, class _Promise>
    using __value_t = __decay_t<
      __value_types_of_t< _Sender, env_of_t<_Promise&>, __msingle_or<void>, __msingle_or<void>>>;

    template <class _Sender, class _Promise>
    using __receiver_t = __t<__receiver<__id<_Promise>, __value_t<_Sender, _Promise>>>;

    template <class _Value>
    struct __sender_awaitable_base {
      bool await_ready() const noexcept {
        return false;
      }

      _Value await_resume() {
        switch (__result_.index()) {
        case 0: // receiver contract not satisfied
          STDEXEC_ASSERT(!"_Should never get here");
          break;
        case 1: // set_value
          if constexpr (!std::is_void_v<_Value>)
            return (_Value&&) std::get<1>(__result_);
          else
            return;
        case 2: // set_error
          std::rethrow_exception(std::get<2>(__result_));
        }
        std::terminate();
      }

     protected:
      __expected_t<_Value> __result_;
    };

    template <class _PromiseId, class _SenderId>
    struct __sender_awaitable {
      using _Promise = stdexec::__t<_PromiseId>;
      using _Sender = stdexec::__t<_SenderId>;
      using __value = __value_t<_Sender, _Promise>;

      struct __t : __sender_awaitable_base<__value> {
        __t(_Sender&& sndr, __coro::coroutine_handle<_Promise> __hcoro) //
          noexcept(__nothrow_connectable<_Sender, __receiver>)
          : __op_state_(connect(
            (_Sender&&) sndr,
            __receiver{
              {&this->__result_, __hcoro}
        })) {
        }

        void await_suspend(__coro::coroutine_handle<_Promise>) noexcept {
          start(__op_state_);
        }
       private:
        using __receiver = __receiver_t<_Sender, _Promise>;
        connect_result_t<_Sender, __receiver> __op_state_;
      };
    };

    template <class _Promise, class _Sender>
    using __sender_awaitable_t = __t<__sender_awaitable<__id<_Promise>, __id<_Sender>>>;

    template <class _Sender, class _Promise>
    concept __awaitable_sender =
      sender_in<_Sender, env_of_t<_Promise&>> &&             //
      __valid<__value_t, _Sender, _Promise> &&               //
      sender_to<_Sender, __receiver_t<_Sender, _Promise>> && //
      requires(_Promise& __promise) {
        { __promise.unhandled_stopped() } -> convertible_to<__coro::coroutine_handle<>>;
      };

    struct __unspecified {
      __unspecified get_return_object() noexcept;
      __unspecified initial_suspend() noexcept;
      __unspecified final_suspend() noexcept;
      void unhandled_exception() noexcept;
      void return_void() noexcept;
      __coro::coroutine_handle<> unhandled_stopped() noexcept;
    };

    struct as_awaitable_t {
      template <class _Tp, class _Promise>
      static constexpr auto __select_impl_() noexcept {
        if constexpr (tag_invocable<as_awaitable_t, _Tp, _Promise&>) {
          using _Result = tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
          constexpr bool _Nothrow = nothrow_tag_invocable<as_awaitable_t, _Tp, _Promise&>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__awaitable<_Tp, __unspecified>) { // NOT __awaitable<_Tp, _Promise> !!
          return static_cast < _Tp && (*) () noexcept > (nullptr);
        } else if constexpr (__awaitable_sender<_Tp, _Promise>) {
          using _Result = __sender_awaitable_t<_Promise, _Tp>;
          constexpr bool _Nothrow =
            __nothrow_constructible_from<_Result, _Tp, __coro::coroutine_handle<_Promise>>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else {
          return static_cast < _Tp && (*) () noexcept > (nullptr);
        }
      }
      template <class _Tp, class _Promise>
      using __select_impl_t = decltype(__select_impl_<_Tp, _Promise>());

      template <class _Tp, class _Promise>
      auto operator()(_Tp&& __t, _Promise& __promise) const
        noexcept(__nothrow_callable<__select_impl_t<_Tp, _Promise>>)
          -> __call_result_t<__select_impl_t<_Tp, _Promise>> {
        if constexpr (tag_invocable<as_awaitable_t, _Tp, _Promise&>) {
          using _Result = tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
          static_assert(__awaitable<_Result, _Promise>);
          return tag_invoke(*this, (_Tp&&) __t, __promise);
        } else if constexpr (__awaitable<_Tp, __unspecified>) { // NOT __awaitable<_Tp, _Promise> !!
          return (_Tp&&) __t;
        } else if constexpr (__awaitable_sender<_Tp, _Promise>) {
          auto __hcoro = __coro::coroutine_handle<_Promise>::from_promise(__promise);
          return __sender_awaitable_t<_Promise, _Tp>{(_Tp&&) __t, __hcoro};
        } else {
          return (_Tp&&) __t;
        }
      }
    };
  } // namespace __as_awaitable

  using __as_awaitable::as_awaitable_t;
  inline constexpr as_awaitable_t as_awaitable{};

  namespace __with_awaitable_senders {

    template <class _Promise = void>
    class __continuation_handle;

    template <>
    class __continuation_handle<void> {
     public:
      __continuation_handle() = default;

      template <class _Promise>
      __continuation_handle(__coro::coroutine_handle<_Promise> __coro) noexcept
        : __coro_(__coro) {
        if constexpr (requires(_Promise& __promise) { __promise.unhandled_stopped(); }) {
          __stopped_callback_ = [](void* __address) noexcept -> __coro::coroutine_handle<> {
            // This causes the rest of the coroutine (the part after the co_await
            // of the sender) to be skipped and invokes the calling coroutine's
            // stopped handler.
            return __coro::coroutine_handle<_Promise>::from_address(__address)
              .promise()
              .unhandled_stopped();
          };
        }
        // If _Promise doesn't implement unhandled_stopped(), then if a "stopped" unwind
        // reaches this point, it's considered an unhandled exception and terminate()
        // is called.
      }

      __coro::coroutine_handle<> handle() const noexcept {
        return __coro_;
      }

      __coro::coroutine_handle<> unhandled_stopped() const noexcept {
        return __stopped_callback_(__coro_.address());
      }

     private:
      __coro::coroutine_handle<> __coro_{};
      using __stopped_callback_t = __coro::coroutine_handle<> (*)(void*) noexcept;
      __stopped_callback_t __stopped_callback_ = [](void*) noexcept -> __coro::coroutine_handle<> {
        std::terminate();
      };
    };

    template <class _Promise>
    class __continuation_handle {
     public:
      __continuation_handle() = default;

      __continuation_handle(__coro::coroutine_handle<_Promise> __coro) noexcept
        : __continuation_{__coro} {
      }

      __coro::coroutine_handle<_Promise> handle() const noexcept {
        return __coro::coroutine_handle<_Promise>::from_address(__continuation_.handle().address());
      }

      __coro::coroutine_handle<> unhandled_stopped() const noexcept {
        return __continuation_.unhandled_stopped();
      }

     private:
      __continuation_handle<> __continuation_{};
    };

    struct __with_awaitable_senders_base {
      template <class _OtherPromise>
      void set_continuation(__coro::coroutine_handle<_OtherPromise> __hcoro) noexcept {
        static_assert(!std::is_void_v<_OtherPromise>);
        __continuation_ = __hcoro;
      }

      void set_continuation(__continuation_handle<> __continuation) noexcept {
        __continuation_ = __continuation;
      }

      __continuation_handle<> continuation() const noexcept {
        return __continuation_;
      }

      __coro::coroutine_handle<> unhandled_stopped() noexcept {
        return __continuation_.unhandled_stopped();
      }

     private:
      __continuation_handle<> __continuation_{};
    };

    template <class _Promise>
    struct with_awaitable_senders : __with_awaitable_senders_base {
      template <class _Value>
      auto await_transform(_Value&& __val) -> __call_result_t<as_awaitable_t, _Value, _Promise&> {
        static_assert(derived_from<_Promise, with_awaitable_senders>);
        return as_awaitable((_Value&&) __val, static_cast<_Promise&>(*this));
      }
    };
  } // namespace __with_awaitable_senders;

  using __with_awaitable_senders::with_awaitable_senders;
  using __with_awaitable_senders::__continuation_handle;
#endif

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __submit
  namespace __submit_ {
    template <class _ReceiverId>
    struct __operation_base;

    template <class _ReceiverId>
    struct __operation_base {
      using _Receiver = __t<_ReceiverId>;
      _Receiver __rcvr_;

      using __delete_fn_t = void(__operation_base<_ReceiverId>*) noexcept;
      __delete_fn_t* __delete_;
    };

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver;
        __operation_base<_ReceiverId>* __op_state_;

        // Forward all the receiver ops, and delete the operation state.
        template <__completion_tag _Tag, class... _As>
          requires __callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept(
          __nothrow_callable<_Tag, _Receiver, _As...>) {
          // Delete the state as cleanup:
          auto __g = __scope_guard{__self.__op_state_->__delete_, __self.__op_state_};
          return __tag((_Receiver&&) __self.__op_state_->__rcvr_, (_As&&) __as...);
        }

        // Forward all receiever queries.
        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };
    };
    template <class _ReceiverId>
    using __receiver_t = __t<__receiver<_ReceiverId>>;

    template <class _SenderId, class _ReceiverId>
    struct __operation : __operation_base<_ReceiverId> {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      connect_result_t<_Sender, __receiver_t<_ReceiverId>> __op_state_;

      template <__decays_to<_Receiver> _CvrefReceiver>
      __operation(_Sender&& __sndr, _CvrefReceiver&& __rcvr)
        : __operation_base<_ReceiverId>{
            (_CvrefReceiver&&) __rcvr,
            [](__operation_base<_ReceiverId>* __self) noexcept {
              delete static_cast<__operation*>(__self);
            }}
        , __op_state_(connect((_Sender&&) __sndr, __receiver_t<_ReceiverId>{this})) {
      }
    };

    struct __submit_t {
      template <receiver _Receiver, sender_to<_Receiver> _Sender>
      void operator()(_Sender&& __sndr, _Receiver __rcvr) const noexcept(false) {
        start((new __operation<__id<_Sender>, __id<_Receiver>>{
                 (_Sender&&) __sndr, (_Receiver&&) __rcvr})
                ->__op_state_);
      }
    };
  } // namespace __submit_

  using __submit_::__submit_t;
  inline constexpr __submit_t __submit{};

  namespace __inln {
    struct __scheduler {
      using __t = __scheduler;
      using __id = __scheduler;

      template <class _Receiver>
      struct __op : __immovable {
        _Receiver __recv_;

        friend void tag_invoke(start_t, __op& __self) noexcept {
          set_value((_Receiver&&) __self.__recv_);
        }
      };

      struct __sender {
        using __t = __sender;
        using __id = __sender;
        using is_sender = void;
        using completion_signatures = stdexec::completion_signatures<set_value_t()>;

        template <receiver_of<completion_signatures> _Receiver>
        friend __op<_Receiver> tag_invoke(connect_t, __sender, _Receiver __rcvr) {
          return {{}, (_Receiver&&) __rcvr};
        }

        friend auto tag_invoke(get_env_t, __sender) noexcept {
          return __env::__env_fn{[](get_completion_scheduler_t<set_value_t>) noexcept {
            return __scheduler{};
          }};
        }
      };

      friend __sender tag_invoke(schedule_t, __scheduler) {
        return {};
      }

      bool operator==(const __scheduler&) const noexcept = default;
    };
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  namespace __start_detached {
    template <class _EnvId>
    struct __detached_receiver {
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        using is_receiver = void;
        using __id = __detached_receiver;
        STDEXEC_NO_UNIQUE_ADDRESS _Env __env_;

        template <same_as<set_value_t> _Tag, class... _As>
        friend void tag_invoke(_Tag, __t&&, _As&&...) noexcept {
        }

        template <same_as<set_error_t> _Tag, class _Error>
        [[noreturn]] friend void tag_invoke(_Tag, __t&&, _Error&&) noexcept {
          std::terminate();
        }

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&&) noexcept {
        }

        friend const _Env& tag_invoke(get_env_t, const __t& __self) noexcept {
          // BUGBUG NOT TO SPEC
          return __self.__env_;
        }
      };
    };
    template <class _Env>
    using __detached_receiver_t = __t<__detached_receiver<__id<__decay_t<_Env>>>>;

    struct start_detached_t;

    // When looking for user-defined customizations of start_detached, these
    // are the signatures to test against, in order:
    using _Sender = __0;
    using _Env = __1;
    using __cust_sigs = //
      __types<
        tag_invoke_t(start_detached_t, _Sender),
        tag_invoke_t(start_detached_t, _Sender, _Env),
        tag_invoke_t(start_detached_t, get_scheduler_t(_Env&), _Sender),
        tag_invoke_t(start_detached_t, get_scheduler_t(_Env&), _Sender, _Env)>;

    template <class _Sender, class _Env>
    inline constexpr bool __is_start_detached_customized =
      __minvocable<__which<__cust_sigs>, _Sender, _Env>;

    struct __submit_detached {
      template <class _Sender, class _Env>
      void operator()(_Sender&& __sndr, _Env&& __env) const {
        __submit((_Sender&&) __sndr, __detached_receiver_t<_Env>{(_Env&&) __env});
      }
    };

    template <class _Sender, class _Env>
    using __dispatcher_for =
      __make_dispatcher<__cust_sigs, __mconst<__submit_detached>, _Sender, _Env>;

    struct start_detached_t {
      template <sender _Sender, class _Env = empty_env>
        requires sender_to<_Sender, __detached_receiver_t<_Env>>
              || __is_start_detached_customized<_Sender, _Env>
      void operator()(_Sender&& __sndr, _Env&& __env = _Env{}) const
        noexcept(__nothrow_callable<__dispatcher_for<_Sender, _Env>, _Sender, _Env>) {
        using _Dispatcher = __dispatcher_for<_Sender, _Env>;
        // The selected implementation should return void
        static_assert(same_as<void, __call_result_t<_Dispatcher, _Sender, _Env>>);
        _Dispatcher{}((_Sender&&) __sndr, (_Env&&) __env);
      }
    };
  } // namespace __start_detached

  using __start_detached::start_detached_t;
  inline constexpr start_detached_t start_detached{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  namespace __just {
    template <class _Tag, class... _Ts>
    using __completion_signatures_ = completion_signatures<_Tag(_Ts...)>;

    template <class _ReceiverId, class _Tag, class... _Ts>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __immovable {
        using __id = __operation;
        std::tuple<_Ts...> __vals_;
        _Receiver __rcvr_;

        friend void tag_invoke(start_t, __t& __op_state) noexcept {
          std::apply(
            [&__op_state](_Ts&... __ts) {
              _Tag{}((_Receiver&&) __op_state.__rcvr_, (_Ts&&) __ts...);
            },
            __op_state.__vals_);
        }
      };
    };

    template <class _Tag, class... _Ts>
    struct __basic_sender {
      template <class _Receiver>
      using __operation_t = stdexec::__t<__operation<stdexec::__id<_Receiver>, _Tag, _Ts...>>;

      struct __t {
        using __id = __basic_sender;
        using is_sender = void;
        using completion_signatures = __completion_signatures_<_Tag, _Ts...>;

        std::tuple<_Ts...> __vals_;

        template <receiver_of<completion_signatures> _Receiver>
          requires(copy_constructible<_Ts> && ...)
        friend auto tag_invoke(connect_t, const __t& __sndr, _Receiver __rcvr) //
          noexcept((std::is_nothrow_copy_constructible_v<_Ts> && ...)) -> __operation_t<_Receiver> {
          return {{}, __sndr.__vals_, (_Receiver&&) __rcvr};
        }

        template <receiver_of<completion_signatures> _Receiver>
        friend auto tag_invoke(connect_t, __t&& __sndr, _Receiver __rcvr) //
          noexcept((std::is_nothrow_move_constructible_v<_Ts> && ...)) -> __operation_t<_Receiver> {
          return {{}, ((__t&&) __sndr).__vals_, (_Receiver&&) __rcvr};
        }

        friend empty_env tag_invoke(get_env_t, const __t&) noexcept {
          return {};
        }
      };
    };

    template <class... _Values>
    struct __sender {
      using __base = stdexec::__t<__basic_sender<set_value_t, _Values...>>;

      struct __t : __base {
        using __id = __sender;
      };
    };

    template <class _Error>
    struct __error_sender {
      using __base = stdexec::__t<__basic_sender<set_error_t, _Error>>;

      struct __t : __base {
        using __id = __error_sender;
      };
    };

    struct __stopped_sender : __t<__basic_sender<set_stopped_t>> {
      using __id = __stopped_sender;
      using __t = __stopped_sender;
    };

    inline constexpr struct __just_t {
      template <__movable_value... _Ts>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        __t<__sender<__decay_t<_Ts>...>>
        operator()(_Ts&&... __ts) const
        noexcept((__nothrow_constructible_from<__decay_t<_Ts>, _Ts> && ...)) {
        return {{{(_Ts&&) __ts...}}};
      }
    } just{};

    inline constexpr struct __just_error_t {
      template <__movable_value _Error>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        __t<__error_sender<__decay_t<_Error>>>
        operator()(_Error&& __err) const
        noexcept(__nothrow_constructible_from<__decay_t<_Error>, _Error>) {
        return {{{(_Error&&) __err}}};
      }
    } just_error{};

    inline constexpr struct __just_stopped_t {
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        __stopped_sender
        operator()() const noexcept {
        return {{}};
      }
    } just_stopped{};
  }

  using __just::just;
  using __just::just_error;
  using __just::just_stopped;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  namespace __execute_ {
    template <class _Fun>
    struct __as_receiver {
      using is_receiver = void;
      _Fun __fun_;

      template <same_as<set_value_t> _Tag>
      friend void tag_invoke(_Tag, __as_receiver&& __rcvr) noexcept {
        try {
          __rcvr.__fun_();
        } catch (...) {
          set_error((__as_receiver&&) __rcvr, std::exception_ptr());
        }
      }

      template <same_as<set_error_t> _Tag>
      [[noreturn]] friend void tag_invoke(_Tag, __as_receiver&&, std::exception_ptr) noexcept {
        std::terminate();
      }

      template <same_as<set_stopped_t> _Tag>
      friend void tag_invoke(_Tag, __as_receiver&&) noexcept {
      }

      friend empty_env tag_invoke(get_env_t, const __as_receiver&) noexcept {
        return {};
      }
    };

    struct execute_t {
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const //
        noexcept(noexcept(
          __submit(schedule((_Scheduler&&) __sched), __as_receiver<_Fun>{(_Fun&&) __fun}))) {
        (void) __submit(schedule((_Scheduler&&) __sched), __as_receiver<_Fun>{(_Fun&&) __fun});
      }

      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
              && tag_invocable<execute_t, _Scheduler, _Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const
        noexcept(nothrow_tag_invocable<execute_t, _Scheduler, _Fun>) {
        (void) tag_invoke(execute_t{}, (_Scheduler&&) __sched, (_Fun&&) __fun);
      }
    };
  }

  using __execute_::execute_t;
  inline constexpr execute_t execute{};

  // NOT TO SPEC:
  namespace __closure {
    template <__class _Dp>
    struct sender_adaptor_closure;
  }

  using __closure::sender_adaptor_closure;

  template <class _Tp>
  concept __sender_adaptor_closure =
    derived_from<__decay_t<_Tp>, sender_adaptor_closure<__decay_t<_Tp>>>
    && move_constructible<__decay_t<_Tp>> && constructible_from<__decay_t<_Tp>, _Tp>;

  template <class _Tp, class _Sender>
  concept __sender_adaptor_closure_for =
    __sender_adaptor_closure<_Tp> && sender<__decay_t<_Sender>>
    && __callable<_Tp, __decay_t<_Sender>> && sender<__call_result_t<_Tp, __decay_t<_Sender>>>;

  namespace __closure {
    template <class _T0, class _T1>
    struct __compose : sender_adaptor_closure<__compose<_T0, _T1>> {
      STDEXEC_NO_UNIQUE_ADDRESS _T0 __t0_;
      STDEXEC_NO_UNIQUE_ADDRESS _T1 __t1_;

      template <sender _Sender>
        requires __callable<_T0, _Sender> && __callable<_T1, __call_result_t<_T0, _Sender>>
      __call_result_t<_T1, __call_result_t<_T0, _Sender>> operator()(_Sender&& __sndr) && {
        return ((_T1&&) __t1_)(((_T0&&) __t0_)((_Sender&&) __sndr));
      }

      template <sender _Sender>
        requires __callable<const _T0&, _Sender>
              && __callable<const _T1&, __call_result_t<const _T0&, _Sender>>
      __call_result_t<_T1, __call_result_t<_T0, _Sender>> operator()(_Sender&& __sndr) const & {
        return __t1_(__t0_((_Sender&&) __sndr));
      }
    };

    template <__class _Dp>
    struct sender_adaptor_closure { };

    template <sender _Sender, __sender_adaptor_closure_for<_Sender> _Closure>
    __call_result_t<_Closure, _Sender> operator|(_Sender&& __sndr, _Closure&& __clsur) {
      return ((_Closure&&) __clsur)((_Sender&&) __sndr);
    }

    template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
    __compose<__decay_t<_T0>, __decay_t<_T1>> operator|(_T0&& __t0, _T1&& __t1) {
      return {{}, (_T0&&) __t0, (_T1&&) __t1};
    }

    template <class _Fun, class... _As>
    struct __binder_back : sender_adaptor_closure<__binder_back<_Fun, _As...>> {
      STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;
      std::tuple<_As...> __as_;

      template <sender _Sender>
        requires __callable<_Fun, _Sender, _As...>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        __call_result_t<_Fun, _Sender, _As...>
        operator()(_Sender&& __sndr) && noexcept(__nothrow_callable<_Fun, _Sender, _As...>) {
        return std::apply(
          [&__sndr, this](_As&... __as) {
            return ((_Fun&&) __fun_)((_Sender&&) __sndr, (_As&&) __as...);
          },
          __as_);
      }

      template <sender _Sender>
        requires __callable<const _Fun&, _Sender, const _As&...>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        __call_result_t<const _Fun&, _Sender, const _As&...>
        operator()(_Sender&& __sndr) const & //
        noexcept(__nothrow_callable<const _Fun&, _Sender, const _As&...>) {
        return std::apply(
          [&__sndr, this](const _As&... __as) { return __fun_((_Sender&&) __sndr, __as...); },
          __as_);
      }
    };
  } // namespace __closure

  using __closure::__binder_back;

  namespace __adaptors {
    // A derived-to-base cast that works even when the base is not
    // accessible from derived.
    template <class _Tp, class _Up>
    STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
      __copy_cvref_t<_Up&&, _Tp>
      __c_cast(_Up&& u) noexcept
      requires __decays_to<_Tp, _Tp>
    {
      static_assert(std::is_reference_v<__copy_cvref_t<_Up&&, _Tp>>);
      static_assert(STDEXEC_IS_BASE_OF(_Tp, __decay_t<_Up>));
      return (__copy_cvref_t<_Up&&, _Tp>) (_Up&&) u;
    }

    namespace __no {
      struct __nope { };

      struct __receiver : __nope { };

      template <same_as<set_error_t> _Tag>
      void tag_invoke(_Tag, __receiver, std::exception_ptr) noexcept;
      template <same_as<set_stopped_t> _Tag>
      void tag_invoke(_Tag, __receiver) noexcept;
      empty_env tag_invoke(get_env_t, __receiver) noexcept;
    }

    using __not_a_receiver = __no::__receiver;

    template <class _Base>
    struct __adaptor {
      struct __t {
        template <class _T1>
          requires constructible_from<_Base, _T1>
        explicit __t(_T1&& __base)
          : __base_((_T1&&) __base) {
        }

       private:
        STDEXEC_NO_UNIQUE_ADDRESS _Base __base_;

       protected:
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          _Base&
          base() & noexcept {
          return __base_;
        }

        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          const _Base&
          base() const & noexcept {
          return __base_;
        }

        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          _Base&&
          base() && noexcept {
          return (_Base&&) __base_;
        }
      };
    };

    template <derived_from<__no::__nope> _Base>
    struct __adaptor<_Base> {
      struct __t : __no::__nope { };
    };
    template <class _Base>
    using __adaptor_base = typename __adaptor<_Base>::__t;

// BUGBUG Not to spec: on gcc and nvc++, member functions in derived classes
// don't shadow type aliases of the same name in base classes. :-O
// On mingw gcc, 'bool(type::existing_member_function)' evaluates to true,
// but 'int(type::existing_member_function)' is an error (as desired).
#define _DISPATCH_MEMBER(_TAG) \
  template <class _Self, class... _Ts> \
  STDEXEC_DETAIL_CUDACC_HOST_DEVICE static auto __call_##_TAG( \
    _Self&& __self, _Ts&&... __ts) noexcept \
    -> decltype(((_Self&&) __self)._TAG((_Ts&&) __ts...)) { \
    static_assert(noexcept(((_Self&&) __self)._TAG((_Ts&&) __ts...))); \
    return ((_Self&&) __self)._TAG((_Ts&&) __ts...); \
  } /**/
#define _CALL_MEMBER(_TAG, ...) __call_##_TAG(__VA_ARGS__)

#if STDEXEC_CLANG()
// Only clang gets this right.
#define _MISSING_MEMBER(_Dp, _TAG) requires { typename _Dp::_TAG; }
#define _DEFINE_MEMBER(_TAG) _DISPATCH_MEMBER(_TAG) using _TAG = void
#else
#define _MISSING_MEMBER(_Dp, _TAG) (__missing_##_TAG<_Dp>())
#define _DEFINE_MEMBER(_TAG) \
  template <class _Dp> \
  static constexpr bool __missing_##_TAG() noexcept { \
    return requires { requires bool(int(_Dp::_TAG)); }; \
  } \
  _DISPATCH_MEMBER(_TAG) \
  static constexpr int _TAG = 1 /**/
#endif

    template <__class _Derived, class _Base>
    struct receiver_adaptor {
      class __t
        : __adaptor_base<_Base>
        , __receiver_base {
        friend _Derived;
        _DEFINE_MEMBER(set_value);
        _DEFINE_MEMBER(set_error);
        _DEFINE_MEMBER(set_stopped);
        _DEFINE_MEMBER(get_env);

        static constexpr bool __has_base = !derived_from<_Base, __no::__nope>;

        template <class _Dp>
        using __base_from_derived_t = decltype(__declval<_Dp>().base());

        using __get_base_t =
          __if_c< __has_base, __mbind_back_q<__copy_cvref_t, _Base>, __q<__base_from_derived_t>>;

        template <class _Dp>
        using __base_t = __minvoke<__get_base_t, _Dp&&>;

        template <class _Dp>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          static __base_t<_Dp>
          __get_base(_Dp&& __self) noexcept {
          if constexpr (__has_base) {
            return __c_cast<__t>((_Dp&&) __self).base();
          } else {
            return ((_Dp&&) __self).base();
          }
        }

        template <same_as<set_value_t> _SetValue, class... _As>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend auto
          tag_invoke(_SetValue, _Derived&& __self, _As&&... __as) noexcept //
          -> __msecond<                                                    //
            __if_c<same_as<set_value_t, _SetValue>>,
            decltype(_CALL_MEMBER(set_value, (_Derived&&) __self, (_As&&) __as...))> {
          static_assert(noexcept(_CALL_MEMBER(set_value, (_Derived&&) __self, (_As&&) __as...)));
          _CALL_MEMBER(set_value, (_Derived&&) __self, (_As&&) __as...);
        }

        template <same_as<set_value_t> _SetValue, class _Dp = _Derived, class... _As>
          requires _MISSING_MEMBER(_Dp, set_value)
                && tag_invocable<_SetValue, __base_t<_Dp>, _As...>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void tag_invoke(_SetValue, _Derived&& __self, _As&&... __as) noexcept {
          stdexec::set_value(__get_base((_Dp&&) __self), (_As&&) __as...);
        }

        template <same_as<set_error_t> _SetError, class _Error>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend auto
          tag_invoke(_SetError, _Derived&& __self, _Error&& __err) noexcept //
          -> __msecond<                                                     //
            __if_c<same_as<set_error_t, _SetError>>,
            decltype(_CALL_MEMBER(set_error, (_Derived&&) __self, (_Error&&) __err))> {
          static_assert(noexcept(_CALL_MEMBER(set_error, (_Derived&&) __self, (_Error&&) __err)));
          _CALL_MEMBER(set_error, (_Derived&&) __self, (_Error&&) __err);
        }

        template <same_as<set_error_t> _SetError, class _Error, class _Dp = _Derived>
          requires _MISSING_MEMBER(_Dp, set_error)
                && tag_invocable<_SetError, __base_t<_Dp>, _Error>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void tag_invoke(_SetError, _Derived&& __self, _Error&& __err) noexcept {
          stdexec::set_error(__get_base((_Derived&&) __self), (_Error&&) __err);
        }

        template <same_as<set_stopped_t> _SetStopped, class _Dp = _Derived>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend auto
          tag_invoke(_SetStopped, _Derived&& __self) noexcept //
          -> __msecond<                                       //
            __if_c<same_as<set_stopped_t, _SetStopped>>,
            decltype(_CALL_MEMBER(set_stopped, (_Dp&&) __self))> {
          static_assert(noexcept(_CALL_MEMBER(set_stopped, (_Derived&&) __self)));
          _CALL_MEMBER(set_stopped, (_Derived&&) __self);
        }

        template <same_as<set_stopped_t> _SetStopped, class _Dp = _Derived>
          requires _MISSING_MEMBER(_Dp, set_stopped) && tag_invocable<_SetStopped, __base_t<_Dp>>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void tag_invoke(_SetStopped, _Derived&& __self) noexcept {
          stdexec::set_stopped(__get_base((_Derived&&) __self));
        }

        // Pass through the get_env receiver query
        template <same_as<get_env_t> _GetEnv, class _Dp = _Derived>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend auto
          tag_invoke(_GetEnv, const _Derived& __self) noexcept
          -> decltype(_CALL_MEMBER(get_env, (const _Dp&) __self)) {
          static_assert(noexcept(_CALL_MEMBER(get_env, __self)));
          return _CALL_MEMBER(get_env, __self);
        }

        template <same_as<get_env_t> _GetEnv, class _Dp = _Derived>
          requires _MISSING_MEMBER(_Dp, get_env)
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend auto tag_invoke(_GetEnv, const _Derived& __self) noexcept
          -> env_of_t<__base_t<const _Dp&>> {
          return stdexec::get_env(__get_base(__self));
        }

       public:
        __t() = default;
        using __adaptor_base<_Base>::__adaptor_base;

        using is_receiver = void;
      };
    };
  } // namespace __adaptors

  template <__class _Derived, receiver _Base = __adaptors::__not_a_receiver>
  using receiver_adaptor = typename __adaptors::receiver_adaptor<_Derived, _Base>::__t;

  template <class _Receiver, class _Fun, class... _As>
  concept __receiver_of_invoke_result = //
    receiver_of<
      _Receiver,
      completion_signatures<
        __minvoke<__remove<void, __qf<set_value_t>>, std::invoke_result_t<_Fun, _As...>>>>;

  template <bool _CanThrow = false, class _Receiver, class _Fun, class... _As>
  void __set_value_invoke(_Receiver&& __rcvr, _Fun&& __fun, _As&&... __as) noexcept(!_CanThrow) {
    if constexpr (_CanThrow || __nothrow_invocable<_Fun, _As...>) {
      if constexpr (same_as<void, std::invoke_result_t<_Fun, _As...>>) {
        std::invoke((_Fun&&) __fun, (_As&&) __as...);
        set_value((_Receiver&&) __rcvr);
      } else {
        set_value((_Receiver&&) __rcvr, std::invoke((_Fun&&) __fun, (_As&&) __as...));
      }
    } else {
      try {
        stdexec::__set_value_invoke<true>((_Receiver&&) __rcvr, (_Fun&&) __fun, (_As&&) __as...);
      } catch (...) {
        set_error((_Receiver&&) __rcvr, std::current_exception());
      }
    }
  }

  template <class _Fun>
  struct _WITH_FUNCTION_ { };

  template <class... _Args>
  struct _WITH_ARGUMENTS_ { };

  inline constexpr __mstring __not_callable_diag =
    "The specified function is not callable with the arguments provided."__csz;

  template <__mstring _Context, __mstring _Diagnostic = __not_callable_diag>
  struct _NOT_CALLABLE_ { };

  template <__mstring _Context>
  struct __callable_error {
    template <class _Fun, class... _Args>
    using __f =     //
      __mexception< //
        _NOT_CALLABLE_<_Context>,
        _WITH_FUNCTION_<_Fun>,
        _WITH_ARGUMENTS_<_Args...>>;
  };

  template <class _Fun, class... _Args>
    requires invocable<_Fun, _Args...>
  using __non_throwing_ = __mbool<__nothrow_invocable<_Fun, _Args...>>;

  template <class _Tag, class _Fun, class _Sender, class _Env, class _Catch>
  using __with_error_invoke_t = //
    __if<
      __gather_completions_for<
        _Tag,
        _Sender,
        _Env,
        __mbind_front<__mtry_catch_q<__non_throwing_, _Catch>, _Fun>,
        __q<__mand>>,
      completion_signatures<>,
      __with_exception_ptr>;

  template <class _Fun, class... _Args>
    requires invocable<_Fun, _Args...>
  using __set_value_invoke_t = //
    completion_signatures<
      __minvoke< __remove<void, __qf<set_value_t>>, std::invoke_result_t<_Fun, _Args...>>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  namespace __then {
    template <class _ReceiverId, class _Fun>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __data {
        _Receiver __rcvr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;
      };

      struct __t {
        using is_receiver = void;
        using __id = __receiver;
        __data* __op_;

        // Customize set_value by invoking the invocable and passing the result
        // to the downstream receiver
        template <__same_as<set_value_t> _Tag, class... _As>
          requires invocable<_Fun, _As...> && __receiver_of_invoke_result<_Receiver, _Fun, _As...>
        friend void tag_invoke(_Tag, __t&& __self, _As&&... __as) noexcept {
          stdexec::__set_value_invoke(
            (_Receiver&&) __self.__op_->__rcvr_, (_Fun&&) __self.__op_->__fun_, (_As&&) __as...);
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, class... _As>
          requires __callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          __tag((_Receiver&&) __self.__op_->__rcvr_, (_As&&) __as...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept
          -> env_of_t<const _Receiver&> {
          return get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _Fun>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_id = __receiver<_ReceiverId, _Fun>;
      using __receiver_t = stdexec::__t<__receiver_id>;

      struct __t : __immovable {
        using __id = __operation;
        typename __receiver_id::__data __data_;
        connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver __rcvr, _Fun __fun) //
          noexcept(__nothrow_decay_copyable<_Receiver>      //
                     && __nothrow_decay_copyable<_Fun>      //
                       && __nothrow_connectable<_Sender, __receiver_t>)
          : __data_{(_Receiver&&) __rcvr, (_Fun&&) __fun}
          , __op_(connect((_Sender&&) __sndr, __receiver_t{&__data_})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_);
        }
      };
    };

    inline constexpr __mstring __then_context = "In stdexec::then(Sender, Function)..."__csz;
    using __on_not_callable = __callable_error<__then_context>;

    template <class _SenderId, class _Fun>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;
      template <class _Receiver>
      using __receiver = stdexec::__t<__receiver<stdexec::__id<_Receiver>, _Fun>>;
      template <class _Self, class _Receiver>
      using __operation =
        stdexec::__t<__operation<__copy_cvref_t<_Self, _Sender>, stdexec::__id<_Receiver>, _Fun>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <class _Self, class _Env>
        using __completion_signatures = //
          __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __with_error_invoke_t<
              set_value_t,
              _Fun,
              __copy_cvref_t<_Self, _Sender>,
              _Env,
              __on_not_callable>,
            __mbind_front<__mtry_catch_q<__set_value_invoke_t, __on_not_callable>, _Fun>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_constructible_from<
                   __operation<_Self, _Receiver>,
                   __copy_cvref_t<_Self, _Sender>,
                   _Receiver&&,
                   __copy_cvref_t<_Self, _Fun>>) -> __operation<_Self, _Receiver> {
          return {((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr, ((_Self&&) __self).__fun_};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completion_signatures<_Self, _Env>
          requires true;

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }
      };
    };

    struct then_t {
      template <class _Sender, class _Fun>
      using __sender = __t<__sender<stdexec::__id<__decay_t<_Sender>>, _Fun>>;

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>)
             && (!tag_invocable<then_t, _Sender, _Fun>) && sender<__sender<_Sender, _Fun>>
      __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
        return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
      }

      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<
                 then_t,
                 __completion_scheduler_for<_Sender, set_value_t>,
                 _Sender,
                 _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(get_env(__sndr));
        return tag_invoke(then_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>)
             && tag_invocable<then_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_t, _Sender, _Fun>) {
        return tag_invoke(then_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <class _Fun>
      __binder_back<then_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }

  using __then::then_t;
  inline constexpr then_t then{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_error]
  namespace __upon_error {
    template <class _ReceiverId, class _Fun>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver;

        _Receiver __rcvr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <__one_of<set_value_t, set_stopped_t> _Tag, same_as<__t> _Self, class... _Args>
          requires __callable<_Tag, _Receiver, _Args...>
        friend void tag_invoke(_Tag, _Self&& __self, _Args&&... __args) noexcept {
          _Tag{}((_Receiver&&) __self.__rcvr_, (_Args&&) __args...);
        }

        // Customize set_error by invoking the invocable and passing the result
        // to the base class
        template <same_as<set_error_t> _Tag, __decays_to<__t> _Self, class _Error>
          requires invocable<_Fun, _Error> && __receiver_of_invoke_result<_Receiver, _Fun, _Error>
        friend void tag_invoke(_Tag, _Self&& __self, _Error&& __error) noexcept {
          stdexec::__set_value_invoke(
            (_Receiver&&) __self.__rcvr_, (_Fun&&) __self.__fun_, (_Error&&) __error);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return get_env(__self.__rcvr_);
        }
      };
    };

    inline constexpr __mstring __upon_error_context =
      "In stdexec::upon_error(Sender, Function)..."__csz;
    using __on_not_callable = __callable_error<__upon_error_context>;

    template <class _SenderId, class _Fun>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <class _Receiver>
      using __receiver = stdexec::__t<__receiver<__id<_Receiver>, _Fun>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <class _Self, class _Env>
        using __completion_signatures = //
          __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __with_error_invoke_t<
              set_error_t,
              _Fun,
              __copy_cvref_t<_Self, _Sender>,
              _Env,
              __on_not_callable>,
            __q<__compl_sigs::__default_set_value>,
            __mbind_front_q<__set_value_invoke_t, _Fun>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>)
            -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>> {
          return stdexec::connect(
            ((_Self&&) __self).__sndr_,
            __receiver<_Receiver>{(_Receiver&&) __rcvr, ((_Self&&) __self).__fun_});
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;
        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completion_signatures<_Self, _Env>
          requires true;

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }
      };
    };

    struct upon_error_t {
      template <class _Sender, class _Fun>
      using __sender = __t<__sender<stdexec::__id<__decay_t<_Sender>>, _Fun>>;

      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<upon_error_t, set_error_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<
                 upon_error_t,
                 __completion_scheduler_for<_Sender, set_error_t>,
                 _Sender,
                 _Fun>) {
        auto __sched = get_completion_scheduler<set_error_t>(get_env(__sndr)); // TODO ADD TEST!
        return tag_invoke(upon_error_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<upon_error_t, set_error_t, _Sender, _Fun>)
             && tag_invocable<upon_error_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<upon_error_t, _Sender, _Fun>) {
        return tag_invoke(upon_error_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<upon_error_t, set_error_t, _Sender, _Fun>)
             && (!tag_invocable<upon_error_t, _Sender, _Fun>) && sender<__sender<_Sender, _Fun>>
      __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
        return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
      }

      template <class _Fun>
      __binder_back<upon_error_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }

  using __upon_error::upon_error_t;
  inline constexpr upon_error_t upon_error{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_stopped]
  namespace __upon_stopped {
    template <class _ReceiverId, class _Fun>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver;
        _Receiver __rcvr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <__one_of<set_value_t, set_error_t> _Tag, same_as<__t> _Self, class... _Args>
          requires __callable<_Tag, _Receiver, _Args...>
        friend void tag_invoke(_Tag, _Self&& __self, _Args&&... __args) noexcept {
          _Tag{}((_Receiver&&) __self.__rcvr_, (_Args&&) __args...);
        }

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          stdexec::__set_value_invoke((_Receiver&&) __self.__rcvr_, (_Fun&&) __self.__fun_);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return get_env(__self.__rcvr_);
        }
      };
    };

    inline constexpr __mstring __upon_stopped_context =
      "In stdexec::upon_stopped(Sender, Function)..."__csz;
    using __on_not_callable = __callable_error<__upon_stopped_context>;

    template <class _SenderId, class _Fun>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <class _Receiver>
      using __receiver = stdexec::__t<__receiver<__id<_Receiver>, _Fun>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <class _Self, class _Env>
        using __completion_signatures = //
          __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __with_error_invoke_t<
              set_stopped_t,
              _Fun,
              __copy_cvref_t<_Self, _Sender>,
              _Env,
              __on_not_callable>,
            __q<__compl_sigs::__default_set_value>,
            __q<__compl_sigs::__default_set_error>,
            __set_value_invoke_t<_Fun>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires __receiver_of_invoke_result<_Receiver, _Fun>
                && sender_to<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_connectable<_Sender, __receiver<_Receiver>>)
            -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>> {
          return stdexec::connect(
            ((_Self&&) __self).__sndr_,
            __receiver<_Receiver>{(_Receiver&&) __rcvr, ((_Self&&) __self).__fun_});
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;
        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completion_signatures<_Self, _Env>
          requires true;

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }
      };
    };

    struct upon_stopped_t {
      template <class _Sender, class _Fun>
      using __sender = __t<__sender<__id<__decay_t<_Sender>>, _Fun>>;

      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<
                   upon_stopped_t,
                   set_stopped_t,
                   _Sender,
                   _Fun>
              && __callable<_Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<
                 upon_stopped_t,
                 __completion_scheduler_for<_Sender, set_stopped_t>,
                 _Sender,
                 _Fun>) {
        auto __sched = get_completion_scheduler<set_stopped_t>(get_env(__sndr)); // TODO ADD TEST!
        return tag_invoke(upon_stopped_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<
                  upon_stopped_t,
                  set_stopped_t,
                  _Sender,
                  _Fun>)
             && tag_invocable<upon_stopped_t, _Sender, _Fun> && __callable<_Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<upon_stopped_t, _Sender, _Fun>) {
        return tag_invoke(upon_stopped_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<
                  upon_stopped_t,
                  set_stopped_t,
                  _Sender,
                  _Fun>)
             && (!tag_invocable<upon_stopped_t, _Sender, _Fun>) && __callable<_Fun>
             && sender<__sender<_Sender, _Fun>>
      __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
        return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
      }

      template <__callable _Fun>
      __binder_back<upon_stopped_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };
  }

  using __upon_stopped::upon_stopped_t;
  inline constexpr upon_stopped_t upon_stopped{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.bulk]
  namespace __bulk {
    inline constexpr __mstring __bulk_context = "In stdexec::bulk(Sender, Shape, Function)..."__csz;
    using __on_not_callable = __callable_error<__bulk_context>;

    template <class _ReceiverId, integral _Shape, class _Fun>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t : receiver_adaptor<__t, _Receiver> {
        friend receiver_adaptor<__t, _Receiver>;

        STDEXEC_NO_UNIQUE_ADDRESS _Shape __shape_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __f_;

        template <class... _As>
        void set_value(_As&&... __as) && noexcept
          requires __nothrow_callable<_Fun, _Shape, _As&...>
        {
          for (_Shape __i{}; __i != __shape_; ++__i) {
            __f_(__i, __as...);
          }
          stdexec::set_value(std::move(this->base()), (_As&&) __as...);
        }

        template <class... _As>
        void set_value(_As&&... __as) && noexcept
          requires __callable<_Fun, _Shape, _As&...>
        {
          try {
            for (_Shape __i{}; __i != __shape_; ++__i) {
              __f_(__i, __as...);
            }
            stdexec::set_value(std::move(this->base()), (_As&&) __as...);
          } catch (...) {
            stdexec::set_error(std::move(this->base()), std::current_exception());
          }
        }

       public:
        using __id = __receiver;

        explicit __t(_Receiver __rcvr, _Shape __shape, _Fun __fun)
          : receiver_adaptor<__t, _Receiver>((_Receiver&&) __rcvr)
          , __shape_(__shape)
          , __f_((_Fun&&) __fun) {
        }
      };
    };

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _SenderId, integral _Shape, class _Fun>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <receiver _Receiver>
      using __receiver = stdexec::__t<__receiver<stdexec::__id<_Receiver>, _Shape, _Fun>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Shape __shape_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fun __fun_;

        template <class _Sender, class _Env, class _Catch>
        using __with_error_invoke_t = //
          __if<
            __try_value_types_of_t<
              _Sender,
              _Env,
              __transform<
                __q<__decay_ref>,
                __mbind_front<__mtry_catch_q<__non_throwing_, _Catch>, _Fun, _Shape>>,
              __q<__mand>>,
            completion_signatures<>,
            __with_exception_ptr>;

        template <class _Self, class _Env>
        using __completion_signatures = //
          __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __with_error_invoke_t<__copy_cvref_t<_Self, _Sender>, _Env, __on_not_callable>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_connectable<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>>)
            -> connect_result_t<__copy_cvref_t<_Self, _Sender>, __receiver<_Receiver>> {
          return stdexec::connect(
            ((_Self&&) __self).__sndr_,
            __receiver<_Receiver>{
              (_Receiver&&) __rcvr, __self.__shape_, ((_Self&&) __self).__fun_});
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completion_signatures<_Self, _Env>
          requires true;

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }
      };
    };

    struct bulk_t {
      template <sender _Sender, integral _Shape, class _Fun>
      using __sender = __t<__sender<stdexec::__id<__decay_t<_Sender>>, _Shape, _Fun>>;

      template <sender _Sender, integral _Shape, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<bulk_t, set_value_t, _Sender, _Shape, _Fun>
      sender auto operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const noexcept(
        nothrow_tag_invocable<
          bulk_t,
          __completion_scheduler_for<_Sender, set_value_t>,
          _Sender,
          _Shape,
          _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(get_env(__sndr));
        return tag_invoke(
          bulk_t{}, std::move(__sched), (_Sender&&) __sndr, (_Shape&&) __shape, (_Fun&&) __fun);
      }

      template <sender _Sender, integral _Shape, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<
                  bulk_t,
                  set_value_t,
                  _Sender,
                  _Shape,
                  _Fun>)
             && tag_invocable<bulk_t, _Sender, _Shape, _Fun>
      sender auto operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const
        noexcept(nothrow_tag_invocable<bulk_t, _Sender, _Shape, _Fun>) {
        return tag_invoke(bulk_t{}, (_Sender&&) __sndr, (_Shape&&) __shape, (_Fun&&) __fun);
      }

      template <sender _Sender, integral _Shape, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<
                  bulk_t,
                  set_value_t,
                  _Sender,
                  _Shape,
                  _Fun>)
             && (!tag_invocable<bulk_t, _Sender, _Shape, _Fun>)
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        __sender<_Sender, _Shape, _Fun>
        operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const {
        return __sender<_Sender, _Shape, _Fun>{(_Sender&&) __sndr, __shape, (_Fun&&) __fun};
      }

      template <integral _Shape, class _Fun>
      __binder_back<bulk_t, _Shape, _Fun> operator()(_Shape __shape, _Fun __fun) const {
        return {
          {},
          {},
          {(_Shape&&) __shape, (_Fun&&) __fun}
        };
      }
    };
  }

  using __bulk::bulk_t;
  inline constexpr bulk_t bulk{};

  ////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.split]
  namespace __split {
    template <class _BaseEnv>
    using __env_t = //
      __make_env_t<
        _BaseEnv, // BUGBUG NOT TO SPEC
        __with<get_stop_token_t, in_place_stop_token>>;

    template <class _CvrefSenderId, class _EnvId>
    struct __sh_state;

    template <class _CvrefSenderId, class _EnvId>
    struct __receiver {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      class __t {
        stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __sh_state_;

       public:
        using is_receiver = void;
        using __id = __receiver;

        template <__completion_tag _Tag, class... _As>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __state = __self.__sh_state_;

          try {
            using __tuple_t = __decayed_tuple<_Tag, _As...>;
            __state.__data_.template emplace<__tuple_t>(__tag, (_As&&) __as...);
          } catch (...) {
            using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
            __state.__data_.template emplace<__tuple_t>(set_error, std::current_exception());
          }
          __state.__notify();
        }

        friend const __env_t<_Env>& tag_invoke(get_env_t, const __t& __self) noexcept {
          return __self.__sh_state_.__env_;
        }

        explicit __t(stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __sh_state) noexcept
          : __sh_state_(__sh_state) {
        }
      };
    };

    struct __operation_base {
      using __notify_fn = void(__operation_base*) noexcept;

      __operation_base* __next_{};
      __notify_fn* __notify_{};
    };

    template <class _CvrefSenderId, class _EnvId>
    struct __sh_state {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        using __id = __sh_state;

        template <class... _Ts>
        using __bind_tuples = //
          __mbind_front_q<
            __variant,
            std::tuple<set_stopped_t>, // Initial state of the variant is set_stopped
            std::tuple<set_error_t, std::exception_ptr>,
            _Ts...>;

        using __bound_values_t = //
          __value_types_of_t<
            _CvrefSender,
            __env_t<_Env>,
            __mbind_front_q<__decayed_tuple, set_value_t>,
            __q<__bind_tuples>>;

        using __variant_t = //
          __error_types_of_t<
            _CvrefSender,
            __env_t<_Env>,
            __transform< __mbind_front_q<__decayed_tuple, set_error_t>, __bound_values_t>>;

        using __receiver_ = stdexec::__t<__receiver<_CvrefSenderId, _EnvId>>;

        void* const __token_{(void*) 0xDEADBEEF};
        in_place_stop_source __stop_source_{};
        __variant_t __data_;
        std::atomic<void*> __head_{nullptr};
        __env_t<_Env> __env_;
        connect_result_t<_CvrefSender, __receiver_> __op_state2_;

        explicit __t(_CvrefSender&& __sndr, _Env __env)
          : __env_(__make_env((_Env&&) __env, __with_(get_stop_token, __stop_source_.get_token())))
          , __op_state2_(connect((_CvrefSender&&) __sndr, __receiver_{*this})) {
        }

        void __notify() noexcept {
          void* const __completion_state = static_cast<void*>(this);
          void* __old = __head_.exchange(__completion_state, std::memory_order_acq_rel);
          __operation_base* __op_state = static_cast<__operation_base*>(__old);

          while (__op_state != nullptr) {
            __operation_base* __next = __op_state->__next_;
            __op_state->__notify_(__op_state);
            __op_state = __next;
          }
        }
      };
    };

    template <class _CvrefSenderId, class _EnvId, class _ReceiverId>
    struct __operation {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t : public __operation_base {
        struct __on_stop_requested {
          in_place_stop_source& __stop_source_;

          void operator()() noexcept {
            __stop_source_.request_stop();
          }
        };

        using __on_stop = //
          std::optional<typename stop_token_of_t< env_of_t<_Receiver>&>::template callback_type<
            __on_stop_requested>>;

        _Receiver __rcvr_;
        __on_stop __on_stop_{};
        std::shared_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state_;

       public:
        using __id = __operation;

        __t( //
          _Receiver&& __rcvr,
          std::shared_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state) //
          noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          : __operation_base{nullptr, __notify}
          , __rcvr_((_Receiver&&) __rcvr)
          , __shared_state_(std::move(__shared_state)) {
        }

        STDEXEC_IMMOVABLE(__t);

        static void __notify(__operation_base* __self) noexcept {
          __t* __op = static_cast<__t*>(__self);
          __op->__on_stop_.reset();

          std::visit(
            [&](const auto& __tupl) noexcept -> void {
              std::apply(
                [&](auto __tag, const auto&... __args) noexcept -> void {
                  __tag((_Receiver&&) __op->__rcvr_, __args...);
                },
                __tupl);
            },
            __op->__shared_state_->__data_);
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>* __shared_state =
            __self.__shared_state_.get();
          STDEXEC_ASSERT(__shared_state->__token_ == (void*) 0xDEADBEEF);
          std::atomic<void*>& __head = __shared_state->__head_;
          void* const __completion_state = static_cast<void*>(__shared_state);
          void* __old = __head.load(std::memory_order_acquire);

          if (__old != __completion_state) {
            __self.__on_stop_.emplace(
              get_stop_token(get_env(__self.__rcvr_)),
              __on_stop_requested{__shared_state->__stop_source_});
          }

          do {
            if (__old == __completion_state) {
              __self.__notify(&__self);
              return;
            }
            __self.__next_ = static_cast<__operation_base*>(__old);
          } while (!__head.compare_exchange_weak(
            __old,
            static_cast<void*>(&__self),
            std::memory_order_release,
            std::memory_order_acquire));

          if (__old == nullptr) {
            // the inner sender isn't running
            if (__shared_state->__stop_source_.stop_requested()) {
              // 1. resets __head to completion state
              // 2. notifies waiting threads
              // 3. propagates "stopped" signal to `out_r'`
              __shared_state->__notify();
            } else {
              start(__shared_state->__op_state2_);
            }
          }
        }
      };
    };

    template <class _CvrefSenderId, class _EnvId>
    struct __sender {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      template <class _Receiver>
      using __operation =
        stdexec::__t<__operation<_CvrefSenderId, _EnvId, stdexec::__id<_Receiver>>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        explicit __t(_CvrefSender&& __sndr, _Env __env)
          : __shared_state_{
            std::make_shared<__sh_state_>(static_cast<_CvrefSender&&>(__sndr), (_Env&&) __env)} {
        }

       private:
        using __sh_state_ = stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>;

        template <class... _Tys>
        using __set_value_t = completion_signatures<set_value_t(const __decay_t<_Tys>&...)>;

        template <class _Ty>
        using __set_error_t = completion_signatures<set_error_t(const __decay_t<_Ty>&)>;

        template <class _Self>
        using __completions_t = //
          __try_make_completion_signatures<
            // NOT TO SPEC:
            // See https://github.com/brycelelbach/wg21_p2300_execution/issues/26
            _CvrefSender,
            __env_t<__mfront<_Env, _Self>>,
            completion_signatures<
              set_error_t(const std::exception_ptr&),
              set_stopped_t()>, // NOT TO SPEC
            __q<__set_value_t>,
            __q<__set_error_t>>;

        std::shared_ptr<__sh_state_> __shared_state_;

        template <__decays_to<__t> _Self, receiver_of<__completions_t<_Self>> _Receiver>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __recvr) //
          noexcept(std::is_nothrow_move_constructible_v<_Receiver>) -> __operation<_Receiver> {
          return __operation<_Receiver>{(_Receiver&&) __recvr, __self.__shared_state_};
        }

        template <__decays_to<__t> _Self, class _OtherEnv>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _OtherEnv&&)
          -> __completions_t<_Self>;
      };
    };

    struct split_t;

    // When looking for user-defined customizations of split, these
    // are the signatures to test against, in order:
    using _Sender = __0;
    using _Env = __1;
    using __cust_sigs = //
      __types<
        tag_invoke_t(
          split_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(
          split_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender,
          _Env),
        tag_invoke_t(split_t, get_scheduler_t(_Env&), _Sender),
        tag_invoke_t(split_t, get_scheduler_t(_Env&), _Sender, _Env),
        tag_invoke_t(split_t, _Sender),
        tag_invoke_t(split_t, _Sender, _Env)>;

    template <class _Sender, class _Env>
    inline constexpr bool __is_split_customized = __minvocable<__which<__cust_sigs>, _Sender, _Env>;

    template <class _Sender, class _Env>
    using __sender_t = __t<__sender<stdexec::__cvref_id<_Sender>, stdexec::__id<__decay_t<_Env>>>>;

    template <class _Sender, class _Env>
    using __dispatcher_for =
      __make_dispatcher<__cust_sigs, __mconstructor_for<__sender_t>, _Sender, _Env>;

    struct split_t {
      template <sender _Sender, class _Env = empty_env>
        requires(sender_in<_Sender, _Env> && __decay_copyable<env_of_t<_Sender>>)
             || __is_split_customized<_Sender, _Env>
      auto operator()(_Sender&& __sndr, _Env&& __env = _Env{}) const
        noexcept(__nothrow_callable<__dispatcher_for<_Sender, _Env>, _Sender, _Env>)
          -> __call_result_t<__dispatcher_for<_Sender, _Env>, _Sender, _Env> {
        return __dispatcher_for<_Sender, _Env>{}((_Sender&&) __sndr, (_Env&&) __env);
      }

      __binder_back<split_t> operator()() const {
        return {{}, {}, {}};
      }
    };
  } // namespace __split

  using __split::split_t;
  inline constexpr split_t split{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.ensure_started]
  namespace __ensure_started {
    template <class _BaseEnv>
    using __env_t = //
      __make_env_t<
        _BaseEnv, // NOT TO SPEC
        __with<get_stop_token_t, in_place_stop_token>>;

    template <class _CvrefSenderId, class _EnvId>
    struct __sh_state;

    template <class _CvrefSenderId, class _EnvId>
    struct __receiver {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      class __t {
        __intrusive_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state_;

       public:
        using is_receiver = void;
        using __id = __receiver;

        explicit __t(stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __shared_state) noexcept
          : __shared_state_(__shared_state.__intrusive_from_this()) {
        }

        template <__completion_tag _Tag, class... _As>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __state = *__self.__shared_state_;

          try {
            using __tuple_t = __decayed_tuple<_Tag, _As...>;
            __state.__data_.template emplace<__tuple_t>(__tag, (_As&&) __as...);
          } catch (...) {
            using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
            __state.__data_.template emplace<__tuple_t>(set_error, std::current_exception());
          }

          __state.__notify();
          __self.__shared_state_.reset();
        }

        friend const __env_t<_Env>& tag_invoke(get_env_t, const __t& __self) noexcept {
          return __self.__shared_state_->__env_;
        }
      };
    };

    struct __operation_base {
      using __notify_fn = void(__operation_base*) noexcept;
      __notify_fn* __notify_{};
    };

    template <class _CvrefSenderId, class _EnvId>
    struct __sh_state {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      struct __t : __enable_intrusive_from_this<__t> {
        using __id = __sh_state;

        template <class... _Ts>
        using __bind_tuples = //
          __mbind_front_q<
            __variant,
            std::tuple<set_stopped_t>, // Initial state of the variant is set_stopped
            std::tuple<set_error_t, std::exception_ptr>,
            _Ts...>;

        using __bound_values_t = //
          __value_types_of_t<
            _CvrefSender,
            __env_t<_Env>,
            __mbind_front_q<__decayed_tuple, set_value_t>,
            __q<__bind_tuples>>;

        using __variant_t = //
          __error_types_of_t<
            _CvrefSender,
            __env_t<_Env>,
            __transform< __mbind_front_q<__decayed_tuple, set_error_t>, __bound_values_t>>;

        using __receiver_t = stdexec::__t<__receiver<_CvrefSenderId, _EnvId>>;

        __variant_t __data_;
        in_place_stop_source __stop_source_{};

        std::atomic<void*> __op_state1_{nullptr};
        __env_t<_Env> __env_;
        connect_result_t<_CvrefSender, __receiver_t> __op_state2_;

        explicit __t(_CvrefSender&& __sndr, _Env __env)
          : __env_(__make_env((_Env&&) __env, __with_(get_stop_token, __stop_source_.get_token())))
          , __op_state2_(connect((_CvrefSender&&) __sndr, __receiver_t{*this})) {
          start(__op_state2_);
        }

        void __notify() noexcept {
          void* const __completion_state = static_cast<void*>(this);
          void* const __old = __op_state1_.exchange(__completion_state, std::memory_order_acq_rel);
          if (__old != nullptr) {
            auto* __op = static_cast<__operation_base*>(__old);
            __op->__notify_(__op);
          }
        }

        void __detach() noexcept {
          __stop_source_.request_stop();
        }
      };
    };

    template <class _CvrefSenderId, class _EnvId, class _ReceiverId>
    struct __operation {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t : public __operation_base {
        struct __on_stop_requested {
          in_place_stop_source& __stop_source_;

          void operator()() noexcept {
            __stop_source_.request_stop();
          }
        };

        using __on_stop = //
          std::optional< typename stop_token_of_t< env_of_t<_Receiver>&>::template callback_type<
            __on_stop_requested>>;

        _Receiver __rcvr_;
        __on_stop __on_stop_{};
        __intrusive_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state_;

       public:
        using __id = __operation;

        __t(                                                                                //
          _Receiver __rcvr,                                                                 //
          __intrusive_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state) //
          noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          : __operation_base{__notify}
          , __rcvr_((_Receiver&&) __rcvr)
          , __shared_state_(std::move(__shared_state)) {
        }

        ~__t() {
          // Check to see if this operation was ever started. If not,
          // detach the (potentially still running) operation:
          if (nullptr == __shared_state_->__op_state1_.load(std::memory_order_acquire)) {
            __shared_state_->__detach();
          }
        }

        STDEXEC_IMMOVABLE(__t);

        static void __notify(__operation_base* __self) noexcept {
          __t* __op = static_cast<__t*>(__self);
          __op->__on_stop_.reset();

          std::visit(
            [&](auto& __tupl) noexcept -> void {
              std::apply(
                [&](auto __tag, auto&... __args) noexcept -> void {
                  __tag((_Receiver&&) __op->__rcvr_, std::move(__args)...);
                },
                __tupl);
            },
            __op->__shared_state_->__data_);
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>* __shared_state =
            __self.__shared_state_.get();
          std::atomic<void*>& __op_state1 = __shared_state->__op_state1_;
          void* const __completion_state = static_cast<void*>(__shared_state);
          void* const __old = __op_state1.load(std::memory_order_acquire);
          if (__old == __completion_state) {
            __self.__notify(&__self);
          } else {
            // register stop callback:
            __self.__on_stop_.emplace(
              get_stop_token(get_env(__self.__rcvr_)),
              __on_stop_requested{__shared_state->__stop_source_});
            // Check if the stop_source has requested cancellation
            if (__shared_state->__stop_source_.stop_requested()) {
              // Stop has already been requested. Don't bother starting
              // the child operations.
              stdexec::set_stopped((_Receiver&&) __self.__rcvr_);
            } else {
              // Otherwise, the inner source hasn't notified completion.
              // Set this operation as the __op_state1 so it's notified.
              void* __old = nullptr;
              if (!__op_state1.compare_exchange_weak(
                    __old, &__self, std::memory_order_release, std::memory_order_acquire)) {
                // We get here when the task completed during the execution
                // of this function. Complete the operation synchronously.
                STDEXEC_ASSERT(__old == __completion_state);
                __self.__notify(&__self);
              }
            }
          }
        }
      };
    };

    template <class _CvrefSenderId, class _EnvId>
    struct __sender {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        explicit __t(_CvrefSender __sndr, _Env __env)
          : __shared_state_{
            __make_intrusive<__sh_state_>((_CvrefSender&&) __sndr, (_Env&&) __env)} {
        }

        ~__t() {
          if (nullptr != __shared_state_) {
            // We're detaching a potentially running operation. Request cancellation.
            __shared_state_->__detach(); // BUGBUG NOT TO SPEC
          }
        }

        // Move-only:
        __t(__t&&) = default;

       private:
        using __sh_state_ = stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>;
        template <class _Receiver>
        using __operation =
          stdexec::__t<__operation<_CvrefSenderId, _EnvId, stdexec::__id<_Receiver>>>;

        template <class... _Tys>
        using __set_value_t = completion_signatures<set_value_t(__decay_t<_Tys>&&...)>;

        template <class _Ty>
        using __set_error_t = completion_signatures<set_error_t(__decay_t<_Ty>&&)>;

        template <class _Self>
        using __completions_t = //
          __try_make_completion_signatures<
            _CvrefSender,
            __env_t<__mfront<_Env, _Self>>,
            completion_signatures<
              set_error_t(std::exception_ptr&&),
              set_stopped_t()>, // BUGBUG NOT TO SPEC
            __q<__set_value_t>,
            __q<__set_error_t>>;

        __intrusive_ptr<__sh_state_> __shared_state_;

        template <same_as<__t> _Self, receiver_of<__completions_t<_Self>> _Receiver>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          noexcept(std::is_nothrow_move_constructible_v<_Receiver>) -> __operation<_Receiver> {
          return __operation<_Receiver>{(_Receiver&&) __rcvr, std::move(__self).__shared_state_};
        }

        template <same_as<__t> _Self, class _OtherEnv>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _OtherEnv&&)
          -> __completions_t<_Self>;
      };
    };

    struct ensure_started_t;

    // When looking for user-defined customizations of ensure_started, these
    // are the signatures to test against, in order:
    using _CvrefSender = __0;
    using _Env = __1;
    using __cust_sigs = //
      __types<
        tag_invoke_t(
          ensure_started_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _CvrefSender&)),
          _CvrefSender),
        tag_invoke_t(
          ensure_started_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _CvrefSender&)),
          _CvrefSender,
          _Env),
        tag_invoke_t(ensure_started_t, get_scheduler_t(_Env&), _CvrefSender),
        tag_invoke_t(ensure_started_t, get_scheduler_t(_Env&), _CvrefSender, _Env),
        tag_invoke_t(ensure_started_t, _CvrefSender),
        tag_invoke_t(ensure_started_t, _CvrefSender, _Env)>;

    template <class _CvrefSender, class _Env>
    inline constexpr bool __is_ensure_started_customized =
      __minvocable<__which<__cust_sigs>, _CvrefSender, _Env>;

    template <class _Sender, class _Env>
    using __sender_t =
      __t<__sender<stdexec::__cvref_id<_Sender, _Sender>, stdexec::__id<__decay_t<_Env>>>>;

    template <class _Sender>
    concept __ensure_started_sender = //
      __is_instance_of<__id<__decay_t<_Sender>>, __sender>;

    template <class _Sender>
    using __fallback =
      __if_c<__ensure_started_sender<_Sender>, __mconst<__first>, __mconstructor_for<__sender_t>>;

    template <class _Sender, class _Env>
    using __dispatcher_for = __make_dispatcher<__cust_sigs, __fallback<_Sender>, _Sender, _Env>;

    struct ensure_started_t {
      template <sender _Sender, class _Env = empty_env>
        requires(sender_in<_Sender, _Env> && __decay_copyable<env_of_t<_Sender>>)
             || __is_ensure_started_customized<_Sender, _Env>
      auto operator()(_Sender&& __sndr, _Env&& __env = _Env{}) const
        noexcept(__nothrow_callable<__dispatcher_for<_Sender, _Env>, _Sender, _Env>)
          -> __call_result_t<__dispatcher_for<_Sender, _Env>, _Sender, _Env> {
        return __dispatcher_for<_Sender, _Env>{}((_Sender&&) __sndr, (_Env&&) __env);
      }

      __binder_back<ensure_started_t> operator()() const {
        return {{}, {}, {}};
      }
    };
  }

  using __ensure_started::ensure_started_t;
  inline constexpr ensure_started_t ensure_started{};

  //////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.let_value]
  // [execution.senders.adaptors.let_error]
  // [execution.senders.adaptors.let_stopped]
  namespace __let {
    template <class _Set>
    struct __on_not_callable_ {
      using __t = __callable_error<"In stdexec::let_value(Sender, Function)..."__csz>;
    };

    template <>
    struct __on_not_callable_<set_error_t> {
      using __t = __callable_error<"In stdexec::let_error(Sender, Function)..."__csz>;
    };

    template <>
    struct __on_not_callable_<set_stopped_t> {
      using __t = __callable_error<"In stdexec::let_stopped(Sender, Function)..."__csz>;
    };

    template <class _Set>
    using __on_not_callable = __t<__on_not_callable_<_Set>>;

    template <class _Tp>
    using __decay_ref = __decay_t<_Tp>&;

    template <class _Fun, class _Set>
    using __result_sender = //
      __transform<
        __q<__decay_ref>,
        __mbind_front<__mtry_catch_q<__call_result_t, __on_not_callable<_Set>>, _Fun>>;

    template <class _Receiver, class _Fun, class _Set>
    using __op_state_for =
      __mcompose< __mbind_back_q<connect_result_t, _Receiver>, __result_sender<_Fun, _Set>>;

    template <class _Set, class _Sig>
    struct __tfx_signal_ {
      template <class, class>
      using __f = completion_signatures<_Sig>;
    };

    template <class _Set, class... _Args>
    struct __tfx_signal_<_Set, _Set(_Args...)> {
      template <class _Env, class _Fun>
      using __f = //
        __try_make_completion_signatures<
          __minvoke<__result_sender<_Fun, _Set>, _Args...>,
          _Env,
          // because we don't know if connect-ing the result sender will throw:
          completion_signatures<set_error_t(std::exception_ptr)>>;
    };

    template <class _Env, class _Fun, class _Set, class _Sig>
    using __tfx_signal_t = __minvoke<__tfx_signal_<_Set, _Sig>, _Env, _Fun>;

    template <class _ReceiverId, class _Fun, class _Set, class... _Tuples>
    struct __operation_base_ {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __immovable {
        using __id = __operation_base_;
        using __results_variant_t = std::variant<std::monostate, _Tuples...>;
        using __op_state_variant_t = //
          __minvoke<
            __transform< __uncurry<__op_state_for<_Receiver, _Fun, _Set>>, __nullable_variant_t>,
            _Tuples...>;

        _Receiver __rcvr_;
        _Fun __fun_;
        __results_variant_t __args_;
        __op_state_variant_t __op_state3_;
      };
    };

    template <class _ReceiverId, class _Fun, class _Set, class... _Tuples>
    struct __receiver_ {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Env = env_of_t<_Receiver>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver_;

        template <__one_of<_Set> _Tag, class... _As>
          requires(1 == __v<__minvoke<__mcount<__decayed_tuple<_As...>>, _Tuples...>>)
               && __minvocable<__result_sender<_Fun, _Set>, _As...>
               && sender_to<__minvoke<__result_sender<_Fun, _Set>, _As...>, _Receiver>
        friend void tag_invoke(_Tag, __t&& __self, _As&&... __as) noexcept {
          try {
            using __tuple_t = __decayed_tuple<_As...>;
            using __op_state_t = __minvoke<__op_state_for<_Receiver, _Fun, _Set>, _As...>;
            auto& __args = __self.__op_state_->__args_.template emplace<__tuple_t>((_As&&) __as...);
            auto& __op = __self.__op_state_->__op_state3_.template emplace<__op_state_t>(
              __conv{[&] {
                return connect(
                  std::apply(std::move(__self.__op_state_->__fun_), __args),
                  std::move(__self.__op_state_->__rcvr_));
              }});
            start(__op);
          } catch (...) {
            set_error(std::move(__self.__op_state_->__rcvr_), std::current_exception());
          }
        }

        template <__completion_tag _Tag, class... _As>
          requires __none_of<_Tag, _Set> && __callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          __tag(std::move(__self.__op_state_->__rcvr_), (_As&&) __as...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }

        using __operation_base_t =
          stdexec::__t<__operation_base_<_ReceiverId, _Fun, _Set, _Tuples...>>;
        __operation_base_t* __op_state_;
      };
    };

    template <class _CvrefSenderId, class _ReceiverId, class _Fun, class _Set>
    using __receiver = //
      stdexec::__t< __gather_completions_for<
        _Set,
        __cvref_t<_CvrefSenderId>,
        env_of_t<__t<_ReceiverId>>,
        __q<__decayed_tuple>,
        __munique<__mbind_front_q<__receiver_, _ReceiverId, _Fun, _Set>>>>;

    template <class _CvrefSenderId, class _ReceiverId, class _Fun, class _Set>
    using __operation_base =
      typename __receiver<_CvrefSenderId, _ReceiverId, _Fun, _Set>::__operation_base_t;

    template <class _CvrefSenderId, class _ReceiverId, class _Fun, class _Set>
    struct __operation {
      using _Sender = stdexec::__cvref_t<_CvrefSenderId>;

      struct __t : __operation_base<_CvrefSenderId, _ReceiverId, _Fun, _Set> {
        using __id = __operation;
        using __op_base_t = __operation_base<_CvrefSenderId, _ReceiverId, _Fun, _Set>;
        using __receiver_t = __receiver<_CvrefSenderId, _ReceiverId, _Fun, _Set>;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_state2_);
        }

        template <class _Receiver2>
        __t(_Sender&& __sndr, _Receiver2&& __rcvr, _Fun __fun)
          : __op_base_t{{}, (_Receiver2&&) __rcvr, (_Fun&&) __fun}
          , __op_state2_(connect((_Sender&&) __sndr, __receiver_t{this})) {
        }

        connect_result_t<_Sender, __receiver_t> __op_state2_;
      };
    };

    template <class _SenderId, class _Fun, class _SetId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;
      using _Set = stdexec::__t<_SetId>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        template <class _Self, class _Receiver>
        using __operation_t = //
          stdexec::__t<
            __operation< stdexec::__cvref_id<_Self, _Sender>, stdexec::__id<_Receiver>, _Fun, _Set>>;
        template <class _Self, class _Receiver>
        using __receiver_t =
          __receiver< stdexec::__cvref_id<_Self, _Sender>, stdexec::__id<_Receiver>, _Fun, _Set>;

        template <class _Sender, class _Env>
        using __completions = //
          __mapply<
            __transform<
              __mbind_front_q<__tfx_signal_t, _Env, _Fun, _Set>,
              __q<__concat_completion_signatures_t>>,
            __completion_signatures_of_t<_Sender, _Env>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return __operation_t<_Self, _Receiver>{
            ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr, ((_Self&&) __self).__fun_};
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completions<__copy_cvref_t<_Self, _Sender>, _Env>
          requires true;

        _Sender __sndr_;
        _Fun __fun_;
      };
    };

    template <class _LetTag, class _SetTag>
    struct __let_xxx_t {
      using __t = _SetTag;
      template <class _Sender, class _Fun>
      using __sender =
        stdexec::__t<__let::__sender<stdexec::__id<__decay_t<_Sender>>, _Fun, _LetTag>>;

      template <sender _Sender, __movable_value _Fun>
        requires __tag_invocable_with_completion_scheduler<_LetTag, set_value_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<
                 _LetTag,
                 __completion_scheduler_for<_Sender, set_value_t>,
                 _Sender,
                 _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(get_env(__sndr));
        return tag_invoke(_LetTag{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<_LetTag, set_value_t, _Sender, _Fun>)
             && tag_invocable<_LetTag, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<_LetTag, _Sender, _Fun>) {
        return tag_invoke(_LetTag{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }

      template <sender _Sender, __movable_value _Fun>
        requires(!__tag_invocable_with_completion_scheduler<_LetTag, set_value_t, _Sender, _Fun>)
             && (!tag_invocable<_LetTag, _Sender, _Fun>) && sender<__sender<_Sender, _Fun>>
      __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
        return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
      }

      template <class _Fun>
      __binder_back<_LetTag, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    };

    struct let_value_t : __let::__let_xxx_t<let_value_t, set_value_t> { };

    struct let_error_t : __let::__let_xxx_t<let_error_t, set_error_t> { };

    struct let_stopped_t : __let::__let_xxx_t<let_stopped_t, set_stopped_t> { };
  } // namespace __let

  using __let::let_value_t;
  inline constexpr let_value_t let_value{};
  using __let::let_error_t;
  inline constexpr let_error_t let_error{};
  using __let::let_stopped_t;
  inline constexpr let_stopped_t let_stopped{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_optional]
  // [execution.senders.adaptors.stopped_as_error]
  namespace __stopped_as_xxx {
    template <class _CvrefSenderId, class _ReceiverId>
    struct __operation;

    template <class _CvrefSenderId, class _ReceiverId>
    struct __receiver {
      using _Sender = stdexec::__t<_CvrefSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver;

        template <same_as<set_value_t> _Tag, class _Ty>
        friend void tag_invoke(_Tag, __t&& __self, _Ty&& __a) noexcept {
          try {
            using _Value = __decay_t<__single_sender_value_t<_Sender, env_of_t<_Receiver>>>;
            static_assert(constructible_from<_Value, _Ty>);
            stdexec::set_value(
              (_Receiver&&) __self.__op_->__rcvr_, std::optional<_Value>{(_Ty&&) __a});
          } catch (...) {
            stdexec::set_error((_Receiver&&) __self.__op_->__rcvr_, std::current_exception());
          }
        }

        template <same_as<set_error_t> _Tag, class _Error>
        friend void tag_invoke(_Tag, __t&& __self, _Error&& __error) noexcept {
          stdexec::set_error((_Receiver&&) __self.__op_->__rcvr_, (_Error&&) __error);
        }

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          using _Value = __decay_t<__single_sender_value_t<_Sender, env_of_t<_Receiver>>>;
          stdexec::set_value(
            (_Receiver&&) __self.__op_->__rcvr_, std::optional<_Value>{std::nullopt});
        }

        template <same_as<get_env_t> _Tag>
        friend env_of_t<_Receiver> tag_invoke(_Tag, const __t& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }

        stdexec::__t<__operation<_CvrefSenderId, _ReceiverId>>* __op_;
      };
    };

    template <class _CvrefSenderId, class _ReceiverId>
    struct __operation {
      using _Sender = stdexec::__t<_CvrefSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = stdexec::__t<__receiver<_CvrefSenderId, _ReceiverId>>;

      struct __t {
        using __id = __operation;

        __t(_Sender&& __sndr, _Receiver&& __rcvr)
          : __rcvr_((_Receiver&&) __rcvr)
          , __op_state_(connect((_Sender&&) __sndr, __receiver_t{this})) {
        }

        STDEXEC_IMMOVABLE(__t);

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_state_);
        }

        _Receiver __rcvr_;
        connect_result_t<_Sender, __receiver_t> __op_state_;
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        template <class _Self, class _Receiver>
        using __operation_t =
          stdexec::__t<__operation<stdexec::__cvref_id<_Self, _Sender>, stdexec::__id<_Receiver>>>;
        template <class _Self, class _Receiver>
        using __receiver_t =
          stdexec::__t<__receiver<stdexec::__cvref_id<_Self, _Sender>, stdexec::__id<_Receiver>>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires __single_typed_sender<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>
                && sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return {((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }

        template <class... _Tys>
          requires(sizeof...(_Tys) == 1)
        using __set_value_t = completion_signatures<set_value_t(std::optional<__decay_t<_Tys>>...)>;

        template <class _Ty>
        using __set_error_t = completion_signatures<set_error_t(_Ty)>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            completion_signatures<set_error_t(std::exception_ptr)>,
            __set_value_t,
            __set_error_t,
            completion_signatures<>>;

        _Sender __sndr_;
      };
    };

    struct stopped_as_optional_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const -> __t<__sender<stdexec::__id<__decay_t<_Sender>>>> {
        return {(_Sender&&) __sndr};
      }

      __binder_back<stopped_as_optional_t> operator()() const noexcept {
        return {};
      }
    };

    struct stopped_as_error_t {
      template <sender _Sender, __movable_value _Error>
      auto operator()(_Sender&& __sndr, _Error __err) const {
        return (_Sender&&) __sndr
             | let_stopped([__err2 = (_Error&&) __err]() mutable //
                           noexcept(std::is_nothrow_move_constructible_v<_Error>) {
                             return just_error((_Error&&) __err2);
                           });
      }

      template <__movable_value _Error>
      auto operator()(_Error __err) const -> __binder_back<stopped_as_error_t, _Error> {
        return {{}, {}, {(_Error&&) __err}};
      }
    };
  } // namespace __stopped_as_xxx

  using __stopped_as_xxx::stopped_as_optional_t;
  inline constexpr stopped_as_optional_t stopped_as_optional{};
  using __stopped_as_xxx::stopped_as_error_t;
  inline constexpr stopped_as_error_t stopped_as_error{};

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
        STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;

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
          using is_sender = void;
          using completion_signatures = //
            __completion_signatures_<
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

          struct __env {
            run_loop* __loop_;

            template <class _CPO>
            friend __scheduler
              tag_invoke(get_completion_scheduler_t<_CPO>, const __env& __self) noexcept {
              return __self.__loop_->get_scheduler();
            }
          };

          friend __env tag_invoke(get_env_t, const __schedule_task& __self) noexcept {
            return __env{__self.__loop_};
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
        friend bool tag_invoke(execute_may_block_caller_t, const __scheduler&) noexcept {
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
    inline void __operation<_ReceiverId>::__t::__start_() noexcept {
      try {
        __loop_->__push_back_(this);
      } catch (...) {
        set_error((_Receiver&&) __rcvr_, std::current_exception());
      }
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
  // [execution.senders.adaptors.schedule_from]
  namespace __schedule_from {
    // Compute a variant type that is capable of storing the results of the
    // input sender when it completes. The variant has type:
    //   variant<
    //     monostate,
    //     tuple<set_stopped_t>,
    //     tuple<set_value_t, __decay_t<_Values1>...>,
    //     tuple<set_value_t, __decay_t<_Values2>...>,
    //        ...
    //     tuple<set_error_t, __decay_t<_Error1>>,
    //     tuple<set_error_t, __decay_t<_Error2>>,
    //        ...
    //   >
    template <class _State, class... _Tuples>
    using __make_bind_ = __mbind_back<_State, _Tuples...>;

    template <class _State>
    using __make_bind = __mbind_front_q<__make_bind_, _State>;

    template <class _Tag>
    using __tuple_t = __mbind_front_q<__decayed_tuple, _Tag>;

    template <class _Sender, class _Env, class _State, class _Tag>
    using __bind_completions_t =
      __gather_completions_for<_Tag, _Sender, _Env, __tuple_t<_Tag>, __make_bind<_State>>;

    template <class _Sender, class _Env>
    using __variant_for_t = //
      __minvoke< __minvoke<
        __mfold_right< __nullable_variant_t, __mbind_front_q<__bind_completions_t, _Sender, _Env>>,
        set_value_t,
        set_error_t,
        set_stopped_t>>;

    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
    struct __operation1;

    // This receiver is to be completed on the execution context
    // associated with the scheduler. When the source sender
    // completes, the completion information is saved off in the
    // operation state so that when this receiver completes, it can
    // read the completion out of the operation state and forward it
    // to the output receiver after transitioning to the scheduler's
    // context.
    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
    struct __receiver2 {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver2;
        stdexec::__t<__operation1<_SchedulerId, _CvrefSenderId, _ReceiverId>>* __op_state_;

        // If the work is successfully scheduled on the new execution
        // context and is ready to run, forward the completion signal in
        // the operation state
        template <same_as<set_value_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          __self.__op_state_->__complete();
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, class... _As>
          requires __callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag, __t&& __self, _As&&... __as) noexcept {
          _Tag{}((_Receiver&&) __self.__op_state_->__rcvr_, (_As&&) __as...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };
    };

    // This receiver is connected to the input sender. When that
    // sender completes (on whatever context it completes on), save
    // the completion information into the operation state. Then,
    // schedule a second operation to __complete on the execution
    // context of the scheduler. That second receiver will read the
    // completion information out of the operation state and propagate
    // it to the output receiver from within the desired context.
    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
    struct __receiver1 {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver2_t = stdexec::__t<__receiver2<_SchedulerId, _CvrefSenderId, _ReceiverId>>;

      struct __t {
        using is_receiver = void;
        stdexec::__t<__operation1<_SchedulerId, _CvrefSenderId, _ReceiverId>>* __op_state_;

        template <class... _Args>
        static constexpr bool __nothrow_complete_ = (__nothrow_decay_copyable<_Args> && ...);

        template <class _Tag, class... _Args>
        static void __complete_(_Tag, __t&& __self, _Args&&... __args) //
          noexcept(__nothrow_complete_<_Args...>) {
          // Write the tag and the args into the operation state so that
          // we can forward the completion from within the scheduler's
          // execution context.
          __self.__op_state_->__data_.template emplace<__decayed_tuple<_Tag, _Args...>>(
            _Tag{}, (_Args&&) __args...);
          // Enqueue the schedule operation so the completion happens
          // on the scheduler's execution context.
          start(__self.__op_state_->__state2_);
        }

        template <__completion_tag _Tag, class... _Args>
          requires __callable<_Tag, _Receiver, __decay_t<_Args>...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _Args&&... __args) noexcept {
          __try_call(
            (_Receiver&&) __self.__op_state_->__rcvr_,
            __fun_c<__complete_<_Tag, _Args...>>,
            (_Tag&&) __tag,
            (__t&&) __self,
            (_Args&&) __args...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };
    };

    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
    struct __operation1 {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver1_t = stdexec::__t<__receiver1<_SchedulerId, _CvrefSenderId, _ReceiverId>>;
      using __receiver2_t = stdexec::__t<__receiver2<_SchedulerId, _CvrefSenderId, _ReceiverId>>;
      using __variant_t = __variant_for_t<_CvrefSender, env_of_t<_Receiver>>;

      struct __t {
        using __id = __operation1;
        _Scheduler __sched_;
        _Receiver __rcvr_;
        __variant_t __data_;
        connect_result_t<_CvrefSender, __receiver1_t> __state1_;
        connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t> __state2_;

        __t(_Scheduler __sched, _CvrefSender&& __sndr, _Receiver&& __rcvr)
          : __sched_((_Scheduler&&) __sched)
          , __rcvr_((_Receiver&&) __rcvr)
          , __state1_(connect((_CvrefSender&&) __sndr, __receiver1_t{this}))
          , __state2_(connect(schedule(__sched_), __receiver2_t{this})) {
        }

        STDEXEC_IMMOVABLE(__t);

        friend void tag_invoke(start_t, __t& __op_state) noexcept {
          start(__op_state.__state1_);
        }

        void __complete() noexcept {
          STDEXEC_ASSERT(!__data_.valueless_by_exception());
          std::visit(
            [&]<class _Tup>(_Tup& __tupl) -> void {
              if constexpr (same_as<_Tup, std::monostate>) {
                std::terminate(); // reaching this indicates a bug in schedule_from
              } else {
                std::apply(
                  [&]<class... _Args>(auto __tag, _Args&... __args) -> void {
                    __tag((_Receiver&&) __rcvr_, (_Args&&) __args...);
                  },
                  __tupl);
              }
            },
            __data_);
        }
      };
    };

    template <class _Tp>
    using __decay_rvalue_ref = __decay_t<_Tp>&&;

    template <class _Tag>
    using __decay_signature =
      __transform<__q<__decay_rvalue_ref>, __mcompose<__q<completion_signatures>, __qf<_Tag>>>;

    template <class _SchedulerId>
    struct __env {
      using _Scheduler = stdexec::__t<_SchedulerId>;

      struct __t {
        using __id = __env;

        _Scheduler __sched_;

        template <__one_of<set_value_t, set_stopped_t> _Tag>
        friend _Scheduler tag_invoke(get_completion_scheduler_t<_Tag>, const __t& __self) noexcept {
          return __self.__sched_;
        }
      };
    };

    template <class _SchedulerId, class _SenderId>
    struct __sender {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Env = stdexec::__t<__env<_SchedulerId>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;
        _Env __env_;
        _Sender __sndr_;

        template <class _Self, class _Receiver>
        using __receiver_t = //
          stdexec::__t<
            __receiver1<_SchedulerId, stdexec::__cvref_id<_Self, _Sender>, stdexec::__id<_Receiver>>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr) //
          -> stdexec::__t< __operation1<
            _SchedulerId,
            stdexec::__cvref_id<_Self, _Sender>,
            stdexec::__id<_Receiver>>> {
          return {__self.__env_.__sched_, ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
        }

        template <class... _Ts>
        using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Ts> && ...)>;

        template <class _Self, class _Env>
        using __all_values_and_errors_nothrow_decay_copyable = __mand<
          error_types_of_t<__copy_cvref_t<_Self, _Sender>, _Env, __all_nothrow_decay_copyable>,
          value_types_of_t<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __all_nothrow_decay_copyable,
            __mand>>;

        template <class _Self, class _Env>
        using __with_error_t = __if<
          __all_values_and_errors_nothrow_decay_copyable<_Self, _Env>,
          completion_signatures<>,
          __with_exception_ptr>;

        template <class _Self, class _Env>
        using __completions_t = //
          __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            __try_make_completion_signatures<
              schedule_result_t<_Scheduler>,
              _Env,
              __with_error_t<_Self, _Env>,
              __mconst<completion_signatures<>>>,
            __decay_signature<set_value_t>,
            __decay_signature<set_error_t>>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completions_t<_Self, _Env>
          requires true;

        friend const _Env& tag_invoke(get_env_t, const __t& __self) noexcept {
          return __self.__env_;
        }
      };
    };

    struct schedule_from_t {
      template <scheduler _Scheduler, sender _Sender>
        requires tag_invocable<schedule_from_t, _Scheduler, _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<schedule_from_t, _Scheduler, _Sender>)
          -> tag_invoke_result_t<schedule_from_t, _Scheduler, _Sender> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const -> stdexec::__t<
        __sender<stdexec::__id<__decay_t<_Scheduler>>, stdexec::__id<__decay_t<_Sender>>>> {
        return {{(_Scheduler&&) __sched}, (_Sender&&) __sndr};
      }
    };
  } // namespace __schedule_from

  using __schedule_from::schedule_from_t;
  inline constexpr schedule_from_t schedule_from{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.transfer]
  namespace __transfer {
    struct transfer_t {
      template <sender _Sender, scheduler _Scheduler>
        requires __tag_invocable_with_completion_scheduler<
          transfer_t,
          set_value_t,
          _Sender,
          _Scheduler>
      tag_invoke_result_t<
        transfer_t,
        __completion_scheduler_for<_Sender, set_value_t>,
        _Sender,
        _Scheduler>
        operator()(_Sender&& __sndr, _Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<
                 transfer_t,
                 __completion_scheduler_for<_Sender, set_value_t>,
                 _Sender,
                 _Scheduler>) {
        auto csch = get_completion_scheduler<set_value_t>(get_env(__sndr));
        return tag_invoke(
          transfer_t{}, std::move(csch), (_Sender&&) __sndr, (_Scheduler&&) __sched);
      }

      template <sender _Sender, scheduler _Scheduler>
        requires(!__tag_invocable_with_completion_scheduler<
                  transfer_t,
                  set_value_t,
                  _Sender,
                  _Scheduler>)
             && tag_invocable<transfer_t, _Sender, _Scheduler>
      tag_invoke_result_t<transfer_t, _Sender, _Scheduler>
        operator()(_Sender&& __sndr, _Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<transfer_t, _Sender, _Scheduler>) {
        return tag_invoke(transfer_t{}, (_Sender&&) __sndr, (_Scheduler&&) __sched);
      }

      // NOT TO SPEC: permit non-typed senders:
      template <sender _Sender, scheduler _Scheduler>
        requires(!__tag_invocable_with_completion_scheduler<
                  transfer_t,
                  set_value_t,
                  _Sender,
                  _Scheduler>)
             && (!tag_invocable<transfer_t, _Sender, _Scheduler>)
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const {
        return schedule_from((_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      template <scheduler _Scheduler>
      __binder_back<transfer_t, __decay_t<_Scheduler>> operator()(_Scheduler&& __sched) const {
        return {{}, {}, {(_Scheduler&&) __sched}};
      }
    };
  } // namespace __transfer

  using __transfer::transfer_t;
  inline constexpr transfer_t transfer{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on {
    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __operation;

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __receiver_ref {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : receiver_adaptor<__t> {
        using __id = __receiver_ref;
        stdexec::__t<__operation<_SchedulerId, _SenderId, _ReceiverId>>* __op_state_;

        _Receiver&& base() && noexcept {
          return (_Receiver&&) __op_state_->__rcvr_;
        }

        const _Receiver& base() const & noexcept {
          return __op_state_->__rcvr_;
        }

        auto get_env() const noexcept
          -> __make_env_t<env_of_t<_Receiver>, __with<get_scheduler_t, _Scheduler>> {
          return __make_env(
            stdexec::get_env(this->base()), __with_(get_scheduler, __op_state_->__scheduler_));
        }
      };
    };

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __receiver {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : receiver_adaptor<__t> {
        using __id = __receiver;
        using __receiver_ref_t = stdexec::__t<__receiver_ref<_SchedulerId, _SenderId, _ReceiverId>>;
        stdexec::__t<__operation<_SchedulerId, _SenderId, _ReceiverId>>* __op_state_;

        _Receiver&& base() && noexcept {
          return (_Receiver&&) __op_state_->__rcvr_;
        }

        const _Receiver& base() const & noexcept {
          return __op_state_->__rcvr_;
        }

        void set_value() && noexcept {
          // cache this locally since *this is going bye-bye.
          auto* __op_state = __op_state_;
          try {
            // This line will invalidate *this:
            start(__op_state->__data_.template emplace<1>(__conv{[__op_state] {
              return connect((_Sender&&) __op_state->__sndr_, __receiver_ref_t{{}, __op_state});
            }}));
          } catch (...) {
            set_error((_Receiver&&) __op_state->__rcvr_, std::current_exception());
          }
        }
      };
    };

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __operation {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __operation;
        using __receiver_t = stdexec::__t<__receiver<_SchedulerId, _SenderId, _ReceiverId>>;
        using __receiver_ref_t = stdexec::__t<__receiver_ref<_SchedulerId, _SenderId, _ReceiverId>>;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(std::get<0>(__self.__data_));
        }

        template <class _Sender2, class _Receiver2>
        __t(_Scheduler __sched, _Sender2&& __sndr, _Receiver2&& __rcvr)
          : __scheduler_((_Scheduler&&) __sched)
          , __sndr_((_Sender2&&) __sndr)
          , __rcvr_((_Receiver2&&) __rcvr)
          , __data_{std::in_place_index<0>, __conv{[this] {
                      return connect(schedule(__scheduler_), __receiver_t{{}, this});
                    }}} {
        }

        STDEXEC_IMMOVABLE(__t);

        _Scheduler __scheduler_;
        _Sender __sndr_;
        _Receiver __rcvr_;
        std::variant<
          connect_result_t<schedule_result_t<_Scheduler>, __receiver_t>,
          connect_result_t<_Sender, __receiver_ref_t>>
          __data_;
      };
    };

    template <class _SchedulerId, class _SenderId>
    struct __sender {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        template <class _ReceiverId>
        using __receiver_ref_t = stdexec::__t<__receiver_ref<_SchedulerId, _SenderId, _ReceiverId>>;
        template <class _ReceiverId>
        using __receiver_t = stdexec::__t<__receiver<_SchedulerId, _SenderId, _ReceiverId>>;
        template <class _ReceiverId>
        using __operation_t = stdexec::__t<__operation<_SchedulerId, _SenderId, _ReceiverId>>;

        _Scheduler __scheduler_;
        _Sender __sndr_;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires constructible_from<_Sender, __copy_cvref_t<_Self, _Sender>>
                && sender_to<schedule_result_t<_Scheduler>, __receiver_t<stdexec::__id<_Receiver>>>
                && sender_to<_Sender, __receiver_ref_t<stdexec::__id<_Receiver>>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<stdexec::__id<_Receiver>> {
          return {
            ((_Self&&) __self).__scheduler_, ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }

        template <class...>
        using __value_t = completion_signatures<>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __try_make_completion_signatures<
            schedule_result_t<_Scheduler>,
            _Env,
            __try_make_completion_signatures<
              __copy_cvref_t<_Self, _Sender>,
              __make_env_t<_Env, __with<get_scheduler_t, _Scheduler>>,
              completion_signatures<set_error_t(std::exception_ptr)>>,
            __q<__value_t>>;
      };
    };

    struct on_t {
      template <scheduler _Scheduler, sender _Sender>
        requires tag_invocable<on_t, _Scheduler, _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<on_t, _Scheduler, _Sender>)
          -> tag_invoke_result_t<on_t, _Scheduler, _Sender> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        -> __t<__sender<stdexec::__id<__decay_t<_Scheduler>>, stdexec::__id<__decay_t<_Sender>>>> {
        // connect-based customization will remove the need for this check
        using __has_customizations = __call_result_t<__has_algorithm_customizations_t, _Scheduler>;
        static_assert(
          !__has_customizations{},
          "For now the default stdexec::on implementation doesn't support scheduling "
          "onto schedulers that customize algorithms.");
        return {(_Scheduler&&) __sched, (_Sender&&) __sndr};
      }
    };
  } // namespace __on

  using __on::on_t;
  inline constexpr on_t on{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  namespace __transfer_just {
    struct transfer_just_t {
      template <scheduler _Scheduler, __movable_value... _Values>
        requires tag_invocable<transfer_just_t, _Scheduler, _Values...>
              && sender<tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...>>
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const
        noexcept(nothrow_tag_invocable<transfer_just_t, _Scheduler, _Values...>)
          -> tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Values&&) __vals...);
      }

      template <scheduler _Scheduler, __movable_value... _Values>
        requires(
          !tag_invocable<transfer_just_t, _Scheduler, _Values...>
          || !sender<tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...>>)
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const
        -> decltype(transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched)) {
        return transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched);
      }
    };
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  inline constexpr transfer_just_t transfer_just{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.into_variant]
  namespace __into_variant {
    template <class _Sender, class _Env>
      requires sender_in<_Sender, _Env>
    using __into_variant_result_t = value_types_of_t<_Sender, _Env>;

    template <class _ReceiverId, class _Variant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver;
        using _Receiver = stdexec::__t<_ReceiverId>;
        _Receiver __rcvr_;

        // Customize set_value by building a variant and passing the result
        // to the base class
        template <same_as<set_value_t> _Tag, class... _As>
          requires constructible_from<_Variant, std::tuple<_As&&...>>
        friend void tag_invoke(_Tag, __t&& __self, _As&&... __as) noexcept {
          try {
            set_value(
              (_Receiver&&) __self.__rcvr_, _Variant{std::tuple<_As&&...>{(_As&&) __as...}});
          } catch (...) {
            set_error((_Receiver&&) __self.__rcvr_, std::current_exception());
          }
        }

        template <same_as<set_error_t> _Tag, class _Error>
        friend void tag_invoke(_Tag, __t&& __self, _Error&& __err) noexcept {
          set_error((_Receiver&&) __self.__rcvr_, (_Error&&) __err);
        }

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          set_stopped((_Receiver&&) __self.__rcvr_);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return get_env(__self.__rcvr_);
        }
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <class _Env>
      using __variant_t = __into_variant_result_t<_Sender, _Env>;

      template <class _Receiver>
      using __receiver_t = //
        stdexec::__t< __receiver<__id<_Receiver>, __variant_t<env_of_t<_Receiver>>>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        template <__decays_to<_Sender> _CvrefSender>
        explicit __t(_CvrefSender&& __sndr)
          : __sndr_((_CvrefSender&&) __sndr) {
        }

       private:
        template <class...>
        using __value_t = completion_signatures<>;

        template <class _Env>
        using __compl_sigs = //
          make_completion_signatures<
            _Sender,
            _Env,
            completion_signatures< set_value_t(__variant_t<_Env>), set_error_t(std::exception_ptr)>,
            __value_t>;

        _Sender __sndr_;

        template <receiver _Receiver>
          requires sender_to<_Sender, __receiver_t<_Receiver>>
        friend auto tag_invoke(connect_t, __t&& __self, _Receiver __rcvr) //
          noexcept(__nothrow_connectable<_Sender, __receiver_t<_Receiver>>)
            -> connect_result_t<_Sender, __receiver_t<_Receiver>> {
          return stdexec::connect(
            (_Sender&&) __self.__sndr_, __receiver_t<_Receiver>{(_Receiver&&) __rcvr});
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }

        template <class _Env>
        friend auto tag_invoke(get_completion_signatures_t, __t&&, _Env&&) //
          -> __compl_sigs<_Env>;
      };
    };

    struct into_variant_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const -> __t<__sender<stdexec::__id<__decay_t<_Sender>>>> {
        return __t<__sender<stdexec::__id<__decay_t<_Sender>>>>{(_Sender&&) __sndr};
      }

      auto operator()() const noexcept {
        return __binder_back<into_variant_t>{};
      }
    };
  } // namespace __into_variant

  using __into_variant::into_variant_t;
  inline constexpr into_variant_t into_variant{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  namespace __when_all {
    enum __state_t {
      __started,
      __error,
      __stopped
    };

    struct __on_stop_requested {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _Env>
    auto __make_env(_Env&& __env, in_place_stop_source& __stop_source) noexcept {
      return __env::__join_env(
        __env::__env_fn{[&](get_stop_token_t) noexcept {
          return __stop_source.get_token();
        }},
        (_Env&&) __env);
    }

    template <class _Env>
    using __env_t = //
      decltype(__when_all::__make_env(__declval<_Env>(), __declval<in_place_stop_source&>()));

    template <class _Tp>
    using __decay_rvalue_ref = __decay_t<_Tp>&&;

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __valid<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    template <class _Env, class _Sender>
    using __single_values_of_t = //
      __try_value_types_of_t<
        _Sender,
        _Env,
        __transform<__q<__decay_rvalue_ref>, __q<__types>>,
        __q<__msingle>>;

    template <class _Env, class... _Senders>
    using __set_values_sig_t = //
      __meval<
        completion_signatures,
        __minvoke< __mconcat<__qf<set_value_t>>, __single_values_of_t<_Env, _Senders>...>>;

    template <class... _Args>
    using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class _Env, class... _SenderIds>
    using __all_value_and_error_args_nothrow_decay_copyable = //
      __mand<                                                 //
        __mand<__try_value_types_of_t<
          __t<_SenderIds>,
          _Env,
          __q<__all_nothrow_decay_copyable>,
          __q<__mand>>...>,
        __mand<__try_error_types_of_t<__t<_SenderIds>, _Env, __q<__all_nothrow_decay_copyable>>...>>;

    template <class _Env, class... _Senders>
    using __completions_t = //
      __concat_completion_signatures_t<
        __if<
          __all_value_and_error_args_nothrow_decay_copyable<_Env, __id<_Senders>...>,
          completion_signatures<set_stopped_t()>,
          completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr&&)>>,
        __minvoke<
          __with_default< __mbind_front_q<__set_values_sig_t, _Env>, completion_signatures<>>,
          _Senders...>,
        __try_make_completion_signatures<
          _Senders,
          _Env,
          completion_signatures<>,
          __mconst<completion_signatures<>>,
          __mcompose<__q<completion_signatures>, __qf<set_error_t>, __q<__decay_rvalue_ref>>>...>;

    struct __not_an_error { };

    struct __tie_fn {
      template <class... _Ty>
      std::tuple<_Ty&...> operator()(_Ty&... __vals) noexcept {
        return std::tuple<_Ty&...>{__vals...};
      }
    };

    template <class _Tag, class _Receiver>
    struct __complete_fn {
      _Receiver& __rcvr_;

      __complete_fn(_Tag, _Receiver& __rcvr) noexcept
        : __rcvr_(__rcvr) {
      }

      template <class _Ty, class... _Ts>
      void operator()(_Ty& __t, _Ts&... __ts) const noexcept {
        if constexpr (!same_as<_Ty, __not_an_error>) {
          _Tag{}((_Receiver&&) __rcvr_, (_Ty&&) __t, (_Ts&&) __ts...);
        }
      }

      void operator()() const noexcept {
        _Tag{}((_Receiver&&) __rcvr_);
      }
    };

    template <class _Receiver, class _ValuesTuple>
    void __set_values(_Receiver& __rcvr, _ValuesTuple& __values) noexcept {
      std::apply(
        [&](auto&... __opt_vals) noexcept -> void {
          std::apply(
            __complete_fn{set_value, __rcvr},
            std::tuple_cat(std::apply(__tie_fn{}, *__opt_vals)...));
        },
        __values);
    }

    template <class _ReceiverId, class _ValuesTuple, class _ErrorsVariant>
    struct __operation_base : __immovable {
      using _Receiver = stdexec::__t<_ReceiverId>;

      void __arrive() noexcept {
        if (0 == --__count_) {
          __complete();
        }
      }

      void __complete() noexcept {
        // Stop callback is no longer needed. Destroy it.
        __on_stop_.reset();
        // All child operations have completed and arrived at the barrier.
        switch (__state_.load(std::memory_order_relaxed)) {
        case __started:
          if constexpr (!same_as<_ValuesTuple, __ignore>) {
            // All child operations completed successfully:
            __when_all::__set_values(__rcvr_, __values_);
          }
          break;
        case __error:
          if constexpr (!same_as<_ErrorsVariant, std::variant<std::monostate>>) {
            // One or more child operations completed with an error:
            std::visit(__complete_fn{set_error, __rcvr_}, __errors_);
          }
          break;
        case __stopped:
          stdexec::set_stopped((_Receiver&&) __rcvr_);
          break;
        default:;
        }
      }

      _Receiver __rcvr_;
      std::atomic<std::size_t> __count_;
      in_place_stop_source __stop_source_{};
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      std::atomic<__state_t> __state_{__started};
      _ErrorsVariant __errors_{};
      STDEXEC_NO_UNIQUE_ADDRESS _ValuesTuple __values_{};
      std::optional<
        typename stop_token_of_t<env_of_t<_Receiver>&>::template callback_type<__on_stop_requested>>
        __on_stop_{};
    };

    template <std::size_t _Index, class _ReceiverId, class _ValuesTuple, class _ErrorsVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;
      template <class _Tuple>
      using __tuple_type = typename std::tuple_element_t<_Index, _Tuple>::value_type;
      using _TupleType = __minvoke< __with_default<__q<__tuple_type>, __ignore>, _ValuesTuple>;

      struct __t {
        using is_receiver = void;
        using __id = __receiver;

        template <class _Error>
        void __set_error(_Error&& __err) noexcept {
          // TODO: What memory orderings are actually needed here?
          if (__error != __op_state_->__state_.exchange(__error)) {
            __op_state_->__stop_source_.request_stop();
            // We won the race, free to write the error into the operation
            // state without worry.
            if constexpr (__nothrow_decay_copyable<_Error>) {
              __op_state_->__errors_.template emplace<__decay_t<_Error>>((_Error&&) __err);
            } else {
              try {
                __op_state_->__errors_.template emplace<__decay_t<_Error>>((_Error&&) __err);
              } catch (...) {
                __op_state_->__errors_.template emplace<std::exception_ptr>(
                  std::current_exception());
              }
            }
          }
        }

        template <same_as<set_value_t> _Tag, class... _Values>
          requires same_as<_ValuesTuple, __ignore> || constructible_from<_TupleType, _Values...>
        friend void tag_invoke(_Tag, __t&& __self, _Values&&... __vals) noexcept {
          if constexpr (!same_as<_ValuesTuple, __ignore>) {
            static_assert(
              same_as<_TupleType, std::tuple<__decay_t<_Values>...>>,
              "One of the senders in this when_all() is fibbing about what types it sends");
            // We only need to bother recording the completion values
            // if we're not already in the "error" or "stopped" state.
            if (__self.__op_state_->__state_ == __started) {
              if constexpr ((__nothrow_decay_copyable<_Values> && ...)) {
                std::get<_Index>(__self.__op_state_->__values_).emplace((_Values&&) __vals...);
              } else {
                try {
                  std::get<_Index>(__self.__op_state_->__values_).emplace((_Values&&) __vals...);
                } catch (...) {
                  __self.__set_error(std::current_exception());
                }
              }
            }
          }
          __self.__op_state_->__arrive();
        }

        template <same_as<set_error_t> _Tag, class _Error>
          requires requires(_ErrorsVariant& __errors, _Error&& __err) {
            __errors.template emplace<__decay_t<_Error>>((_Error&&) __err);
          }
        friend void tag_invoke(_Tag, __t&& __self, _Error&& __err) noexcept {
          __self.__set_error((_Error&&) __err);
          __self.__op_state_->__arrive();
        }

        template <same_as<set_stopped_t> _Tag>
          requires receiver_of<_Receiver, completion_signatures<_Tag()>>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          __state_t __expected = __started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (__self.__op_state_->__state_.compare_exchange_strong(__expected, __stopped)) {
            __self.__op_state_->__stop_source_.request_stop();
          }
          __self.__op_state_->__arrive();
        }

        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const __t& __self) noexcept {
          return __when_all::__make_env(
            get_env(__self.__op_state_->__rcvr_), __self.__op_state_->__stop_source_);
        }

        __operation_base<_ReceiverId, _ValuesTuple, _ErrorsVariant>* __op_state_;
      };
    };

    template <class _Env, class _Sender>
    using __values_opt_tuple_t = //
      __value_types_of_t<
        _Sender,
        __env_t<_Env>,
        __mcompose<__q<std::optional>, __q<__decayed_tuple>>,
        __q<__msingle>>;

    template <class _Env, __max1_sender<__env_t<_Env>>... _Senders>
    struct __traits_ {
      using __completions = __completions_t<__env_t<_Env>, _Senders...>;

      // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
      using __values_tuple = //
        __minvoke<
          __with_default<
            __transform< __mbind_front_q<__values_opt_tuple_t, _Env>, __q<std::tuple>>,
            __ignore>,
          _Senders...>;

      using __nullable_variant_t_ = __munique<__mbind_front_q<std::variant, __not_an_error>>;

      using __error_types = //
        __minvoke<
          __mconcat<__transform<__q<__decay_t>, __nullable_variant_t_>>,
          error_types_of_t<_Senders, __env_t<_Env>, __types>... >;

      using __errors_variant = //
        __if<
          __all_value_and_error_args_nothrow_decay_copyable<_Env, __id<_Senders>...>,
          __error_types,
          __minvoke<__push_back_unique<__q<std::variant>>, __error_types, std::exception_ptr>>;
    };

    template <receiver _Receiver, __max1_sender<__env_t<env_of_t<_Receiver>>>... _Senders>
    struct __traits : __traits_<env_of_t<_Receiver>, _Senders...> {
      using _Traits = __traits_<env_of_t<_Receiver>, _Senders...>;
      using typename _Traits::__completions;
      using typename _Traits::__values_tuple;
      using typename _Traits::__errors_variant;

      template <std::size_t _Index>
      using __receiver =
        __t< __when_all::__receiver< _Index, __id<_Receiver>, __values_tuple, __errors_variant>>;

      using __operation_base =
        __when_all::__operation_base<__id<_Receiver>, __values_tuple, __errors_variant>;

      template <class _Sender, class _Index>
      using __op_state = connect_result_t<_Sender, __receiver<__v<_Index>>>;

      template <class _Tuple = __q<std::tuple>>
      using __op_states_tuple = //
        __minvoke<
          __mzip_with2<__q<__op_state>, _Tuple>,
          __types<_Senders...>,
          __mindex_sequence_for<_Senders...>>;
    };

    template <class _Cvref, class _ReceiverId, class... _SenderIds>
    using __traits_ex = __traits<__t<_ReceiverId>, __minvoke<_Cvref, __t<_SenderIds>>...>;

    template <class _Cvref, class _ReceiverId, class... _SenderIds>
    using __op_states_tuple_ex =
      typename __traits_ex<_Cvref, _ReceiverId, _SenderIds...>::template __op_states_tuple<>;

    template <class _Cvref, class _ReceiverId, class... _SenderIds>
      requires __valid<__op_states_tuple_ex, _Cvref, _ReceiverId, _SenderIds...>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Traits = __traits_ex<_Cvref, _ReceiverId, _SenderIds...>;
      using _Indices = std::index_sequence_for<_SenderIds...>;

      using __operation_base_t = typename _Traits::__operation_base;
      using __op_states_tuple_t = __op_states_tuple_ex<_Cvref, _ReceiverId, _SenderIds...>;

      template <std::size_t _Index>
      using __receiver_t = typename _Traits::template __receiver<_Index>;

      struct __t : __operation_base_t {
        using __id = __operation;

        template <class _SendersTuple, std::size_t... _Is>
        __t(_SendersTuple&& __sndrs, _Receiver __rcvr, std::index_sequence<_Is...>)
          : __operation_base_t{{}, (_Receiver&&) __rcvr, {sizeof...(_Is)}}
          , __op_states_{__conv{[&__sndrs, this]() {
            return stdexec::connect(
              std::get<_Is>((_SendersTuple&&) __sndrs), __receiver_t<_Is>{this});
          }}...} {
        }

        template <class _SendersTuple>
        __t(_SendersTuple&& __sndrs, _Receiver __rcvr)
          : __t((_SendersTuple&&) __sndrs, (_Receiver&&) __rcvr, _Indices{}) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          // register stop callback:
          __self.__on_stop_.emplace(
            get_stop_token(get_env(__self.__rcvr_)), __on_stop_requested{__self.__stop_source_});
          if (__self.__stop_source_.stop_requested()) {
            // Stop has already been requested. Don't bother starting
            // the child operations.
            stdexec::set_stopped((_Receiver&&) __self.__rcvr_);
          } else {
            std::apply(
              [](auto&... __child_ops) noexcept -> void { (stdexec::start(__child_ops), ...); },
              __self.__op_states_);
            if constexpr (sizeof...(_SenderIds) == 0) {
              __self.__complete();
            }
          }
        }

        __op_states_tuple_t __op_states_;
      };
    };

    template <class _From, class _ToId>
    using __cvref_id = __copy_cvref_t<_From, __t<_ToId>>;

    template <class _Indices, class... _SenderIds>
    struct __sender;

    template <std::size_t... _Indices, class... _SenderIds>
    struct __sender<std::index_sequence<_Indices...>, _SenderIds...> {
      template <class _Self, class _Env>
      using __completions_t =
        typename __traits_<_Env, __cvref_id<_Self, _SenderIds>...>::__completions;

      template <class _Self, class _Receiver, std::size_t _Index>
      using __receiver_t =
        typename __traits<_Receiver, __cvref_id<_Self, _SenderIds>...>::template __receiver<_Index>;

      template <class _Self, class _Receiver>
      using __operation_t =
        stdexec::__t<__operation< __copy_cvref_fn<_Self>, stdexec::__id<_Receiver>, _SenderIds...>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;

        template <class... _Sndrs>
        explicit(sizeof...(_Sndrs) == 1) __t(_Sndrs&&... __sndrs)
          : __sndrs_((_Sndrs&&) __sndrs...) {
        }

       private:
        template <__decays_to<__t> _Self, receiver _Receiver>
          requires(
            sender_to< __cvref_id<_Self, _SenderIds>, __receiver_t<_Self, _Receiver, _Indices>>
            && ...)
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return {((_Self&&) __self).__sndrs_, (_Receiver&&) __rcvr};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> dependent_completion_signatures<_Env>;
        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __completions_t<_Self, _Env>
          requires true;

        friend empty_env tag_invoke(get_env_t, const __t&) noexcept {
          return {};
        }

        std::tuple<stdexec::__t<_SenderIds>...> __sndrs_;
      };
    };

    template <class _Sender>
    using __into_variant_result_t = decltype(into_variant(__declval<_Sender>()));

    struct when_all_t {
      template <class... _Senders>
      using __sender_t =
        __t<__sender< std::index_sequence_for<_Senders...>, __id<__decay_t<_Senders>>...>>;

      template <sender... _Senders>
        requires tag_invocable<when_all_t, _Senders...>
              && sender<tag_invoke_result_t<when_all_t, _Senders...>>
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<when_all_t, _Senders...>)
          -> tag_invoke_result_t<when_all_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <sender... _Senders>
        requires(!tag_invocable<when_all_t, _Senders...>) && sender<__sender_t<_Senders...>>
      __sender_t<_Senders...> operator()(_Senders&&... __sndrs) const {
        return __sender_t<_Senders...>{(_Senders&&) __sndrs...};
      }
    };

    struct when_all_with_variant_t {
      template <sender... _Senders>
        requires tag_invocable<when_all_with_variant_t, _Senders...>
              && sender<tag_invoke_result_t<when_all_with_variant_t, _Senders...>>
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<when_all_with_variant_t, _Senders...>)
          -> tag_invoke_result_t<when_all_with_variant_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <sender... _Senders>
        requires(!tag_invocable<when_all_with_variant_t, _Senders...>)
             && (__callable<into_variant_t, _Senders> && ...)
      auto operator()(_Senders&&... __sndrs) const {
        return when_all_t{}(into_variant((_Senders&&) __sndrs)...);
      }
    };

    struct transfer_when_all_t {
      template <scheduler _Sched, sender... _Senders>
        requires tag_invocable<transfer_when_all_t, _Sched, _Senders...>
              && sender<tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...>>
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<transfer_when_all_t, _Sched, _Senders...>)
          -> tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...> {
        return tag_invoke(*this, (_Sched&&) __sched, (_Senders&&) __sndrs...);
      }

      template <scheduler _Sched, sender... _Senders>
        requires(
          (!tag_invocable<transfer_when_all_t, _Sched, _Senders...>)
          || (!sender<tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...>>) )
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const {
        return transfer(when_all_t{}((_Senders&&) __sndrs...), (_Sched&&) __sched);
      }
    };

    struct transfer_when_all_with_variant_t {
      template <scheduler _Sched, sender... _Senders>
        requires tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>
              && sender<tag_invoke_result_t<transfer_when_all_with_variant_t, _Sched, _Senders...>>
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>)
          -> tag_invoke_result_t<transfer_when_all_with_variant_t, _Sched, _Senders...> {
        return tag_invoke(*this, (_Sched&&) __sched, (_Senders&&) __sndrs...);
      }

      template <scheduler _Sched, sender... _Senders>
        requires(!tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>)
             && (__callable<into_variant_t, _Senders> && ...)
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const {
        return transfer_when_all_t{}((_Sched&&) __sched, into_variant((_Senders&&) __sndrs)...);
      }
    };
  } // namespace __when_all

  using __when_all::when_all_t;
  inline constexpr when_all_t when_all{};
  using __when_all::when_all_with_variant_t;
  inline constexpr when_all_with_variant_t when_all_with_variant{};
  using __when_all::transfer_when_all_t;
  inline constexpr transfer_when_all_t transfer_when_all{};
  using __when_all::transfer_when_all_with_variant_t;
  inline constexpr transfer_when_all_with_variant_t transfer_when_all_with_variant{};

  namespace __read {
    template <class _Tag, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __immovable {
        using __id = __operation;
        _Receiver __rcvr_;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          try {
            auto __env = get_env(__self.__rcvr_);
            set_value(std::move(__self.__rcvr_), _Tag{}(__env));
          } catch (...) {
            set_error(std::move(__self.__rcvr_), std::current_exception());
          }
        }
      };
    };

    template <class _Tag>
    struct __sender {
      using __t = __sender;
      using __id = __sender;
      using is_sender = void;

      template <class _Env>
        requires __callable<_Tag, _Env>
      using __completions_t = //
        completion_signatures<
          set_value_t(__call_result_t<_Tag, _Env>),
          set_error_t(std::exception_ptr)>;

      template <class _Receiver>
        requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
      friend auto tag_invoke(connect_t, __sender, _Receiver __rcvr) //
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> stdexec::__t<__operation<_Tag, stdexec::__id<_Receiver>>> {
        return {{}, (_Receiver&&) __rcvr};
      }

      template <class _Env>
      friend auto tag_invoke(get_completion_signatures_t, __sender, _Env&&)
        -> dependent_completion_signatures<_Env>;
      template <__none_of<no_env> _Env>
      friend auto tag_invoke(get_completion_signatures_t, __sender, _Env&&)
        -> __completions_t<_Env>;

      friend empty_env tag_invoke(get_env_t, const __t&) noexcept {
        return {};
      }
    };

    struct __read_t {
      template <class _Tag>
      constexpr __sender<_Tag> operator()(_Tag) const noexcept {
        return {};
      }
    };
  } // namespace __read

  inline constexpr __read::__read_t read{};

  namespace __queries {
    inline auto get_scheduler_t::operator()() const noexcept {
      return read(get_scheduler);
    }

    template <class _Env>
      requires tag_invocable<get_scheduler_t, const _Env&>
    inline auto get_scheduler_t::operator()(const _Env& __env) const noexcept
      -> tag_invoke_result_t<get_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_scheduler_t, const _Env&>>);
      return tag_invoke(get_scheduler_t{}, __env);
    }

    inline auto get_delegatee_scheduler_t::operator()() const noexcept {
      return read(get_delegatee_scheduler);
    }

    template <class _Env>
      requires tag_invocable<get_delegatee_scheduler_t, const _Env&>
    inline auto get_delegatee_scheduler_t::operator()(const _Env& __t) const noexcept
      -> tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_delegatee_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&>>);
      return tag_invoke(get_delegatee_scheduler_t{}, std::as_const(__t));
    }

    inline auto get_allocator_t::operator()() const noexcept {
      return read(get_allocator);
    }

    inline auto get_stop_token_t::operator()() const noexcept {
      return read(get_stop_token);
    }

    template <__completion_tag _CPO>
    template <__has_completion_scheduler_for<_CPO> _Queryable>
    auto get_completion_scheduler_t<_CPO>::operator()(const _Queryable& __queryable) const noexcept
      -> tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&> {
      static_assert(
        nothrow_tag_invocable<get_completion_scheduler_t<_CPO>, const _Queryable&>,
        "get_completion_scheduler<_CPO> should be noexcept");
      static_assert(
        scheduler<tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&>>);
      return tag_invoke(*this, __queryable);
    }
  } // namespace __queries

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    template <class _Sender>
    using __into_variant_result_t = decltype(into_variant(__declval<_Sender>()));

    inline auto __make_env(run_loop& __loop) noexcept {
      return __env::__env_fn{
        [&](__one_of<get_scheduler_t, get_delegatee_scheduler_t> auto) noexcept {
          return __loop.get_scheduler();
        }};
    }

    using __env = decltype(__sync_wait::__make_env(__declval<run_loop&>()));

    // What should sync_wait(just_stopped()) return?
    template <sender_in<__env> _Sender, class _Continuation>
    using __sync_wait_result_impl =
      __value_types_of_t< _Sender, __env, __transform<__q<__decay_t>, _Continuation>, __q<__msingle>>;

    template <class _Sender>
    using __sync_wait_result_t = __mtry_eval<__sync_wait_result_impl, _Sender, __q<std::tuple>>;

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
        using is_receiver = void;
        using __id = __receiver;
        __state<_Values...>* __state_;
        run_loop* __loop_;

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

        template <same_as<set_value_t> _Tag, class... _As>
          requires constructible_from<std::tuple<_Values...>, _As...>
        friend void tag_invoke(_Tag, __t&& __rcvr, _As&&... __as) noexcept {
          try {
            __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
            __rcvr.__loop_->finish();
          } catch (...) {
            __rcvr.__set_error(std::current_exception());
          }
        }

        template <same_as<set_error_t> _Tag, class _Error>
        friend void tag_invoke(_Tag, __t&& __rcvr, _Error __err) noexcept {
          __rcvr.__set_error((_Error&&) __err);
        }

        friend void tag_invoke(set_stopped_t __d, __t&& __rcvr) noexcept {
          __rcvr.__state_->__data_.template emplace<3>(__d);
          __rcvr.__loop_->finish();
        }

        friend __env tag_invoke(get_env_t, const __t& __rcvr) noexcept {
          return __sync_wait::__make_env(*__rcvr.__loop_);
        }
      };
    };

    template <class _Sender>
    using __into_variant_result_t = decltype(into_variant(__declval<_Sender>()));

    struct sync_wait_t;

    using _Sender = __0;
    using __cust_sigs = __types<
      tag_invoke_t(
        sync_wait_t,
        get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
        _Sender),
      tag_invoke_t(sync_wait_t, _Sender)>;

    template <class _Sender>
    inline constexpr bool __is_sync_wait_customized = __minvocable<__which<__cust_sigs>, _Sender>;

    template <class _Sender>
    using __receiver_t = __t<__sync_wait_result_impl<_Sender, __q<__receiver>>>;

    struct __default_impl {
      template <class _Sender>
      auto operator()(_Sender&& __sndr) const -> std::optional<__sync_wait_result_t<_Sender>> {
        using state_t = __sync_wait_result_impl<_Sender, __q<__state>>;
        state_t __state{};
        run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state = connect((_Sender&&) __sndr, __receiver_t<_Sender>{&__state, &__loop});
        start(__op_state);

        // Wait for the variant to be filled in.
        __loop.run();

        if (__state.__data_.index() == 2)
          std::rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return std::nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    };

    template <class _Sender>
    using __dispatcher_for = __make_dispatcher<__cust_sigs, __mconst<__default_impl>, _Sender>;

    // These are for hiding the metaprogramming in diagnostics
    template <class _Sender>
    struct __sync_receiver_for {
      using __t = __receiver_t<_Sender>;
    };
    template <class _Sender>
    using __sync_receiver_for_t = __t<__sync_receiver_for<_Sender>>;

    template <class _Sender>
    struct __value_tuple_for {
      using __t = __sync_wait_result_t<_Sender>;
    };
    template <class _Sender>
    using __value_tuple_for_t = __t<__value_tuple_for<_Sender>>;

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      template <sender_in<__env> _Sender>
        requires __satisfies<__single_value_variant_sender<_Sender, __env>>
              && (sender_to<_Sender, __sync_receiver_for_t<_Sender>>
                  || __is_sync_wait_customized<_Sender>)
      auto operator()(_Sender&& __sndr) const -> std::optional<__value_tuple_for_t<_Sender>> {
        // The selected implementation should return void
        return __dispatcher_for<_Sender>{}((_Sender&&) __sndr);
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait_with_variant]
    struct sync_wait_with_variant_t {
      template <sender_in<__env> _Sender>
        requires __tag_invocable_with_completion_scheduler<
          sync_wait_with_variant_t,
          set_value_t,
          _Sender>
      tag_invoke_result_t<
        sync_wait_with_variant_t,
        __completion_scheduler_for<_Sender, set_value_t>,
        _Sender>
        operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<
                 sync_wait_with_variant_t,
                 __completion_scheduler_for<_Sender, set_value_t>,
                 _Sender>) {

        static_assert(
          std::is_same_v<
            tag_invoke_result_t<
              sync_wait_with_variant_t,
              __completion_scheduler_for<_Sender, set_value_t>,
              _Sender>,
            std::optional<__sync_wait_with_variant_result_t<_Sender>>>,
          "The type of tag_invoke(sync_wait_with_variant, get_completion_scheduler, S) "
          "must be sync-wait-with-variant-type<S, sync-wait-env>");

        auto __sched = get_completion_scheduler<set_value_t>(get_env(__sndr));
        return tag_invoke(sync_wait_with_variant_t{}, std::move(__sched), (_Sender&&) __sndr);
      }

      template <sender_in<__env> _Sender>
        requires(!__tag_invocable_with_completion_scheduler<
                  sync_wait_with_variant_t,
                  set_value_t,
                  _Sender>)
             && tag_invocable<sync_wait_with_variant_t, _Sender>
      tag_invoke_result_t<sync_wait_with_variant_t, _Sender> operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<sync_wait_with_variant_t, _Sender>) {

        static_assert(
          std::is_same_v<
            tag_invoke_result_t<sync_wait_with_variant_t, _Sender>,
            std::optional<__sync_wait_with_variant_result_t<_Sender>>>,
          "The type of tag_invoke(sync_wait_with_variant, S) "
          "must be sync-wait-with-variant-type<S, sync-wait-env>");

        return tag_invoke(sync_wait_with_variant_t{}, (_Sender&&) __sndr);
      }

      template <sender_in<__env> _Sender>
        requires(!__tag_invocable_with_completion_scheduler<
                  sync_wait_with_variant_t,
                  set_value_t,
                  _Sender>)
             && (!tag_invocable<sync_wait_with_variant_t, _Sender>)
             && invocable<sync_wait_t, __into_variant_result_t<_Sender>>
      std::optional<__sync_wait_with_variant_result_t<_Sender>> operator()(_Sender&& __sndr) const {
        return sync_wait_t{}(into_variant((_Sender&&) __sndr));
      }
    };
  } // namespace __sync_wait

  using __sync_wait::sync_wait_t;
  inline constexpr sync_wait_t sync_wait{};
  using __sync_wait::sync_wait_with_variant_t;
  inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};

  struct __ignore_sender {
    using is_sender = void;

    template <sender _Sender>
    constexpr __ignore_sender(_Sender&&) noexcept {
    }
  };

  template <auto _Reason = "You cannot pipe one sender into another."__csz>
  struct _CANNOT_PIPE_INTO_A_SENDER_ { };

  template <class _Sender>
  struct _WITH_SENDER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_CANNOT_PIPE_INTO_A_SENDER_<>, _WITH_SENDER_<_Sender>>;
} // namespace stdexec

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <stdexec::sender _Sender>
  requires stdexec::__ok<stdexec::__bad_pipe_sink_t<_Sender>>
auto operator|(stdexec::__ignore_sender, _Sender&&) noexcept -> stdexec::__ignore_sender;

#include "__detail/__p2300.hpp"

#ifdef __EDG__
#pragma diagnostic pop
#endif

STDEXEC_PRAGMA_POP()
