/*
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

#include "__execution_fwd.hpp"

#include "__concepts.hpp"
#include "__cpo.hpp"
#include "__meta.hpp"
#include "__stop_token.hpp"
#include "__tag_invoke.hpp"

#include <type_traits>
#include <exception>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(probable_guiding_friend)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

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
      constexpr auto operator()(_Query __query) const noexcept -> bool {
        if constexpr (tag_invocable<forwarding_query_t, _Query>) {
          return tag_invoke(*this, static_cast<_Query&&>(__query));
        } else if constexpr (std::derived_from<_Query, forwarding_query_t>) {
          return true;
        } else {
          return false;
        }
      }
    };

    struct query_or_t {
      template <class _Query, class _Queryable, class _Default>
      constexpr auto operator()(_Query, _Queryable&&, _Default&& __default) const
        noexcept(__nothrow_constructible_from<_Default, _Default&&>) -> _Default {
        return static_cast<_Default&&>(__default);
      }

      template <class _Query, class _Queryable, class _Default>
        requires __callable<_Query, _Queryable>
      constexpr auto operator()(_Query __query, _Queryable&& __queryable, _Default&&) const
        noexcept(__nothrow_callable<_Query, _Queryable>) -> __call_result_t<_Query, _Queryable> {
        return static_cast<_Query&&>(__query)(static_cast<_Queryable&&>(__queryable));
      }
    };

    struct execute_may_block_caller_t : __query<execute_may_block_caller_t> {
      template <class _Tp>
        requires tag_invocable<execute_may_block_caller_t, __cref_t<_Tp>>
      constexpr auto operator()(_Tp&& __t) const noexcept -> bool {
        static_assert(same_as<bool, tag_invoke_result_t<execute_may_block_caller_t, __cref_t<_Tp>>>);
        static_assert(nothrow_tag_invocable<execute_may_block_caller_t, __cref_t<_Tp>>);
        return tag_invoke(execute_may_block_caller_t{}, std::as_const(__t));
      }

      constexpr auto operator()(auto&&) const noexcept -> bool {
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

      constexpr auto operator()(auto&&) const noexcept -> stdexec::forward_progress_guarantee {
        return stdexec::forward_progress_guarantee::weakly_parallel;
      }
    };

    struct __has_algorithm_customizations_t : __query<__has_algorithm_customizations_t> {
      template <class _Tp>
      using __result_t = tag_invoke_result_t<__has_algorithm_customizations_t, __cref_t<_Tp>>;

      template <class _Tp>
        requires tag_invocable<__has_algorithm_customizations_t, __cref_t<_Tp>>
      constexpr auto
        operator()(_Tp&&) const noexcept(noexcept(__result_t<_Tp>{})) -> __result_t<_Tp> {
        using _Boolean = tag_invoke_result_t<__has_algorithm_customizations_t, __cref_t<_Tp>>;
        static_assert(_Boolean{} ? true : false); // must be contextually convertible to bool
        return _Boolean{};
      }

      constexpr auto operator()(auto&&) const noexcept -> std::false_type {
        return {};
      }
    };

    // TODO: implement allocator concept
    template <class _T0>
    concept __allocator_c = true;

    struct get_scheduler_t : __query<get_scheduler_t> {
      constexpr STDEXEC_MEMFN_DECL(
        auto forwarding_query)(this const get_scheduler_t&) noexcept -> bool {
        return true;
      }

      template <class _Env>
        requires tag_invocable<get_scheduler_t, const _Env&>
      auto operator()(const _Env& __env) const noexcept
        -> tag_invoke_result_t<get_scheduler_t, const _Env&>;

      template <class _Tag = get_scheduler_t>
      auto operator()() const noexcept;
    };

    struct get_delegatee_scheduler_t : __query<get_delegatee_scheduler_t> {
      constexpr STDEXEC_MEMFN_DECL(
        auto query)(this const get_delegatee_scheduler_t&, forwarding_query_t) noexcept -> bool {
        return true;
      }

      template <class _Env>
        requires tag_invocable<get_delegatee_scheduler_t, const _Env&>
      auto operator()(const _Env& __t) const noexcept
        -> tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&>;

      template <class _Tag = get_delegatee_scheduler_t>
      auto operator()() const noexcept;
    };

    struct get_allocator_t : __query<get_allocator_t> {
      constexpr STDEXEC_MEMFN_DECL(
        auto forwarding_query)(this const get_allocator_t&) noexcept -> bool {
        return true;
      }

      template <class _Env>
        requires tag_invocable<get_allocator_t, const _Env&>
      auto operator()(const _Env& __env) const noexcept
        -> tag_invoke_result_t<get_allocator_t, const _Env&> {
        static_assert(nothrow_tag_invocable<get_allocator_t, const _Env&>);
        static_assert(__allocator_c<tag_invoke_result_t<get_allocator_t, const _Env&>>);
        return tag_invoke(get_allocator_t{}, __env);
      }

      template <class _Tag = get_allocator_t>
      auto operator()() const noexcept;
    };

    struct get_stop_token_t : __query<get_stop_token_t> {
      constexpr STDEXEC_MEMFN_DECL(
        auto forwarding_query)(this const get_stop_token_t&) noexcept -> bool {
        return true;
      }

      template <class _Env, class _Token = never_stop_token>
      auto operator()(const _Env&) const noexcept -> _Token {
        return {};
      }

      template <class _Env, class = void>
        requires tag_invocable<get_stop_token_t, const _Env&>
      auto operator()(const _Env& __env) const noexcept
        -> tag_invoke_result_t<get_stop_token_t, const _Env&> {
        static_assert(nothrow_tag_invocable<get_stop_token_t, const _Env&>);
        static_assert(stoppable_token<tag_invoke_result_t<get_stop_token_t, const _Env&>>);
        return tag_invoke(get_stop_token_t{}, __env);
      }

      template <class _Tag = get_stop_token_t>
      auto operator()() const noexcept;
    };

    template <class _Queryable, class _CPO>
    concept __has_completion_scheduler_for =
      queryable<_Queryable> && //
      tag_invocable<get_completion_scheduler_t<_CPO>, const _Queryable&>;

    template <__completion_tag _CPO>
    struct get_completion_scheduler_t : __query<get_completion_scheduler_t<_CPO>> {
      constexpr STDEXEC_MEMFN_DECL(
        auto
        query)(this const get_completion_scheduler_t<_CPO>&, forwarding_query_t) noexcept -> bool {
        return true;
      }

      template <__has_completion_scheduler_for<_CPO> _Queryable>
      auto operator()(const _Queryable& __queryable) const noexcept
        -> tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&>;
    };

    struct get_domain_t {
      template <class _Ty>
        requires tag_invocable<get_domain_t, const _Ty&>
      constexpr auto operator()(const _Ty& __ty) const noexcept
        -> tag_invoke_result_t<get_domain_t, const _Ty&> {
        static_assert(
          nothrow_tag_invocable<get_domain_t, const _Ty&>,
          "Customizations of get_domain must be noexcept.");
        static_assert(
          __class<tag_invoke_result_t<get_domain_t, const _Ty&>>,
          "Customizations of get_domain must return a class type.");
        return tag_invoke(get_domain_t{}, __ty);
      }

      constexpr STDEXEC_MEMFN_DECL(auto forwarding_query)(this get_domain_t) noexcept -> bool {
        return true;
      }
    };

    struct __is_scheduler_affine_t {
      template <class _Env>
      constexpr auto operator()(const _Env& __env) const noexcept {
        if constexpr (tag_invocable<__is_scheduler_affine_t, const _Env&>) {
          using _Result = tag_invoke_result_t<__is_scheduler_affine_t, const _Env&>;
          static_assert(__same_as<decltype(__v<_Result>), const bool>);
          return _Result();
        } else {
          return std::false_type();
        }
      }

      constexpr STDEXEC_MEMFN_DECL(
        auto forwarding_query)(this __is_scheduler_affine_t) noexcept -> bool {
        return false;
      }
    };

    struct __root_t {
      template <class _Env>
        requires tag_invocable<__root_t, const _Env&>
      constexpr auto operator()(const _Env& __env) const noexcept -> bool {
        STDEXEC_ASSERT(tag_invoke(__root_t{}, __env) == true);
        return true;
      }

      constexpr STDEXEC_MEMFN_DECL(auto forwarding_query)(this __root_t) noexcept -> bool {
        return false;
      }
    };

    struct __root_env_t {
      constexpr STDEXEC_MEMFN_DECL(auto __root)(this const __root_env_t&) noexcept -> bool {
        return true;
      }
    };
  } // namespace __queries

  using __queries::forwarding_query_t;
  using __queries::query_or_t;
  using __queries::execute_may_block_caller_t;
  using __queries::__has_algorithm_customizations_t;
  using __queries::get_forward_progress_guarantee_t;
  using __queries::get_allocator_t;
  using __queries::get_scheduler_t;
  using __queries::get_delegatee_scheduler_t;
  using __queries::get_stop_token_t;
  using __queries::get_completion_scheduler_t;
  using __queries::get_domain_t;
  using __queries::__is_scheduler_affine_t;
  using __queries::__root_t;
  using __queries::__root_env_t;

  inline constexpr forwarding_query_t forwarding_query{};
  inline constexpr query_or_t query_or{}; // NOT TO SPEC
  inline constexpr execute_may_block_caller_t execute_may_block_caller{};
  inline constexpr __has_algorithm_customizations_t __has_algorithm_customizations{};
  inline constexpr get_forward_progress_guarantee_t get_forward_progress_guarantee{};
  inline constexpr get_scheduler_t get_scheduler{};
  inline constexpr get_delegatee_scheduler_t get_delegatee_scheduler{};
  inline constexpr get_allocator_t get_allocator{};
  inline constexpr get_stop_token_t get_stop_token{};
#if !STDEXEC_GCC() || defined(__OPTIMIZE_SIZE__)
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

  inline constexpr get_domain_t get_domain{};

  template <class _Tag, class _Queryable, class _Default>
  using __query_result_or_t = __call_result_t<query_or_t, _Tag, _Queryable, _Default>;

  /////////////////////////////////////////////////////////////////////////////
  namespace __env {
    // For getting an execution environment from a receiver,
    // or the attributes from a sender.
    struct get_env_t {
      template <class _EnvProvider>
        requires tag_invocable<get_env_t, const _EnvProvider&>
      STDEXEC_ATTRIBUTE((always_inline))
      constexpr auto
        operator()(const _EnvProvider& __env_provider) const noexcept
        -> tag_invoke_result_t<get_env_t, const _EnvProvider&> {
        static_assert(queryable<tag_invoke_result_t<get_env_t, const _EnvProvider&>>);
        static_assert(nothrow_tag_invocable<get_env_t, const _EnvProvider&>);
        return tag_invoke(*this, __env_provider);
      }

      template <class _EnvProvider>
      constexpr auto operator()(const _EnvProvider&) const noexcept -> empty_env {
        return {};
      }
    };

    // To be kept in sync with the promise type used in __connect_awaitable
    template <class _Env>
    struct __promise {
      template <class _Ty>
      auto await_transform(_Ty&& __value) noexcept -> _Ty&& {
        return static_cast<_Ty&&>(__value);
      }

      template <class _Ty>
        requires tag_invocable<as_awaitable_t, _Ty, __promise&>
      auto await_transform(_Ty&& __value) //
        noexcept(nothrow_tag_invocable<as_awaitable_t, _Ty, __promise&>)
          -> tag_invoke_result_t<as_awaitable_t, _Ty, __promise&> {
        return tag_invoke(as_awaitable, static_cast<_Ty&&>(__value), *this);
      }

      template <same_as<get_env_t> _Tag>
      friend auto tag_invoke(_Tag, const __promise&) noexcept -> const _Env& {
        std::terminate();
      }
    };

    template <class _Value, class _Tag, class... _Tags>
    struct __with {
      using __t = __with;
      using __id = __with;
      STDEXEC_ATTRIBUTE((no_unique_address))
      _Value __value_;

      __with() = default;

      constexpr explicit __with(_Value __value) noexcept(__nothrow_decay_copyable<_Value>)
        : __value_(static_cast<_Value&&>(__value)) {
      }

      constexpr explicit __with(_Value __value, _Tag, _Tags...) //
        noexcept(__nothrow_decay_copyable<_Value>)
        : __value_(static_cast<_Value&&>(__value)) {
      }

      template <__one_of<_Tag, _Tags...> _Key>
      STDEXEC_MEMFN_DECL(auto query)(this const __with& __self, _Key) //
        noexcept(__nothrow_decay_copyable<_Value>) -> _Value {
        return __self.__value_;
      }
    };

    template <class _Value, class _Tag, class... _Tags>
    __with(_Value, _Tag, _Tags...) -> __with<_Value, _Tag, _Tags...>;

    template <class _Env>
    struct __fwd {
      static_assert(__nothrow_move_constructible<_Env>);
      using __t = __fwd;
      using __id = __fwd;
      STDEXEC_ATTRIBUTE((no_unique_address))
      _Env __env_;

      template <__forwarding_query _Tag>
        requires tag_invocable<_Tag, const _Env&>
      STDEXEC_MEMFN_DECL(
        auto query)(this const __fwd& __self, _Tag) //
        noexcept(nothrow_tag_invocable<_Tag, const _Env&>)
          -> tag_invoke_result_t<_Tag, const _Env&> {
        return _Tag()(__self.__env_);
      }
    };

    template <class _Env>
    __fwd(_Env&&) -> __fwd<_Env>;

    template <class _Env>
    struct __ref {
      using __t = __ref;
      using __id = __ref;
      const _Env& __env_;

      template <class _Tag>
        requires tag_invocable<_Tag, const _Env&>
      STDEXEC_MEMFN_DECL(
        auto query)(this const __ref& __self, _Tag) //
        noexcept(nothrow_tag_invocable<_Tag, const _Env&>)
          -> tag_invoke_result_t<_Tag, const _Env&> {
        return _Tag()(__self.__env_);
      }
    };

    template <class _Env>
    __ref(_Env&) -> __ref<_Env>;

    struct __ref_fn {
      template <class _Env>
      constexpr auto operator()(_Env&& __env) const {
        if constexpr (same_as<_Env, _Env&>) {
          return __ref{static_cast<_Env&&>(__env)};
        } else {
          return static_cast<_Env&&>(__env);
        }
      }
    };

    template <class _Env, class _Tag, class... _Tags>
    struct __without_ : _Env {
      static_assert(__nothrow_move_constructible<_Env>);
      using __t = __without_;
      using __id = __without_;

      constexpr explicit __without_(_Env&& __env, _Tag, _Tags...) noexcept
        : _Env(static_cast<_Env&&>(__env)) {
      }

      template <__one_of<_Tag, _Tags...> _Key, class _Self>
        requires(std::is_base_of_v<__without_, __decay_t<_Self>>)
      STDEXEC_MEMFN_DECL(
        auto query)(this _Self&&, _Key) noexcept = delete;
    };

    struct __without_fn {
      template <class _Env, class _Tag, class... _Tags>
      constexpr auto operator()(_Env&& __env, _Tag, _Tags...) const noexcept -> decltype(auto) {
        if constexpr (tag_invocable<_Tag, _Env> || (tag_invocable<_Tags, _Env> || ...)) {
          return __without_{__ref_fn()(static_cast<_Env&&>(__env)), _Tag(), _Tags()...};
        } else {
          return static_cast<_Env>(static_cast<_Env&&>(__env));
        }
      }
    };

    inline constexpr __without_fn __without{};

    template <class _Tag, auto _Value>
    struct __static_env {
      using value_type = decltype(_Value);

      static constexpr value_type query(_Tag) noexcept {
        return _Value;
      }
    };

    template <class _Env, class _Tag, class... _Tags>
    using __without_t = __result_of<__without, _Env, _Tag, _Tags...>;

    template <class _Second, class _First>
    struct __joined : _Second {
      static_assert(__nothrow_move_constructible<_First>);
      static_assert(__nothrow_move_constructible<_Second>);
      using __t = __joined;
      using __id = __joined;

      STDEXEC_ATTRIBUTE((no_unique_address))
      _First __env_;

      template <class _Tag>
        requires tag_invocable<_Tag, const _First&>
      STDEXEC_MEMFN_DECL(
        auto query)(this const __joined& __self, _Tag) //
        noexcept(nothrow_tag_invocable<_Tag, const _First&>)
          -> tag_invoke_result_t<_Tag, const _First&> {
        return _Tag()(__self.__env_);
      }
    };

    template <class _Second, class _First>
    __joined(_Second&&, _First&&) -> __joined<_Second, _First>;

    template <__nothrow_move_constructible _Fun>
    struct __from {
      using __t = __from;
      using __id = __from;
      STDEXEC_ATTRIBUTE((no_unique_address))
      _Fun __fun_;

      template <class _Tag>
        requires __callable<const _Fun&, _Tag>
      STDEXEC_MEMFN_DECL(auto query)(this const __from& __self, _Tag) //
        noexcept(__nothrow_callable<const _Fun&, _Tag>) -> __call_result_t<const _Fun&, _Tag> {
        return __self.__fun_(_Tag());
      }
    };

    template <class _Fun>
    __from(_Fun) -> __from<_Fun>;

    struct __fwd_fn {
      template <class Env>
      auto operator()(Env&& env) const {
        return __fwd{static_cast<Env&&>(env)};
      }

      auto operator()(empty_env) const -> empty_env {
        return {};
      }
    };

    struct __join_fn {
      auto operator()() const -> empty_env {
        return {};
      }

      template <class _Env>
      auto operator()(_Env&& __env) const -> _Env {
        return static_cast<_Env&&>(__env);
      }

      auto operator()(empty_env) const -> empty_env {
        return {};
      }

      template <class _Env>
      auto operator()(_Env&& __env, empty_env) const -> _Env {
        return static_cast<_Env&&>(__env);
      }

      auto operator()(empty_env, empty_env) const -> empty_env {
        return {};
      }

      template <class... Rest>
      auto operator()(empty_env, Rest&&... rest) const -> decltype(auto) {
        return __fwd_fn()(__join_fn()(static_cast<Rest&&>(rest)...));
      }

      template <class First, class... Rest>
      auto operator()(First&& first, Rest&&... rest) const -> decltype(auto) {
        return __joined{
          __fwd_fn()(__join_fn()(static_cast<Rest&&>(rest)...)), static_cast<First&&>(first)};
      }
    };

    inline constexpr __join_fn __join{};

    template <class... _Envs>
    using __join_t = __result_of<__join, _Envs...>;
  } // namespace __env

  using __env::get_env_t;
  using __env::empty_env;
  inline constexpr get_env_t get_env{};

  template <class _EnvProvider>
  concept environment_provider = //
    requires(_EnvProvider& __ep) {
      { get_env(std::as_const(__ep)) } -> queryable;
    };

  inline constexpr auto __as_root_env = []<class _Env>(_Env __env) noexcept {
    return __env::__join(__root_env_t{}, static_cast<_Env&&>(__env));
  };

  template <class _Env>
  using __as_root_env_t = __result_of<__as_root_env, _Env>;

  template <class _Env>
  concept __is_root_env = requires(_Env&& __env) {
    { __root_t{}(__env) } -> same_as<bool>;
  };

  template <class _Sender>
  concept __is_scheduler_affine = //
    requires { requires __v<__call_result_t<__is_scheduler_affine_t, env_of_t<_Sender>>>; };
} // namespace stdexec

STDEXEC_PRAGMA_POP()
