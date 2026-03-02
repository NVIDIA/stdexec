/*
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

#include "__concepts.hpp"
#include "__config.hpp"  // IWYU pragma: export
#include "__meta.hpp"
#include "__type_traits.hpp"
#include "__utility.hpp"

// IWYU pragma: always_keep

STDEXEC_NAMESPACE_STD_BEGIN
  struct monostate;

  template <class...>
  class variant;

  template <class...>
  class tuple;
STDEXEC_NAMESPACE_STD_END

namespace STDEXEC
{
  struct __none_such;

  namespace __detail
  {
    struct __not_a_variant
    {
      constexpr __not_a_variant() = delete;
    };
  }  // namespace __detail

  template <class... _Ts>
  using __std_variant = __minvoke_if_c<sizeof...(_Ts) == 0,
                                       __mconst<__detail::__not_a_variant>,
                                       __mtransform<__q1<__decay_t>, __munique<__qq<std::variant>>>,
                                       _Ts...>;

  template <class... _Ts>
  using __nullable_std_variant =
    __mcall<__munique<__mbind_front<__qq<std::variant>, std::monostate>>, __decay_t<_Ts>...>;

  template <class... _Ts>
  using __decayed_std_tuple = std::tuple<__decay_t<_Ts>...>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct default_domain;

  template <class...>
  struct indeterminate_domain;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct set_value_t;
  struct set_error_t;
  struct set_stopped_t;

  extern set_value_t const   set_value;
  extern set_error_t const   set_error;
  extern set_stopped_t const set_stopped;

  template <class _Tag>
  concept __completion_tag = __one_of<_Tag, set_value_t, set_error_t, set_stopped_t>;

  template <class _Sender>
  extern bool const enable_receiver;

  namespace __env
  {
    template <class _Query, auto _Value>
    struct cprop;
  }  // namespace __env

  template <class _Query, class _Value>
  struct prop;

  template <class... _Envs>
  struct env;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct get_env_t;
  extern get_env_t const get_env;

  template <class _EnvProvider>
  using env_of_t = __call_result_t<get_env_t, _EnvProvider>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  enum class forward_progress_guarantee
  {
    concurrent,
    parallel,
    weakly_parallel
  };

  struct __execute_may_block_caller_t;
  struct get_forward_progress_guarantee_t;
  struct get_scheduler_t;
  struct get_delegation_scheduler_t;
  template <__completion_tag _CPO>
  struct get_completion_scheduler_t;
  template <class _CPO = void>
  struct get_completion_domain_t;
  template <__completion_tag _CPO>
  struct __get_completion_behavior_t;
  struct get_domain_t;
  struct get_await_completion_adaptor_t;

  struct __debug_env_t;

  extern __execute_may_block_caller_t const     __execute_may_block_caller;
  extern get_forward_progress_guarantee_t const get_forward_progress_guarantee;
  extern get_scheduler_t const                  get_scheduler;
  extern get_delegation_scheduler_t const       get_delegation_scheduler;
  template <__completion_tag _CPO>
  extern get_completion_scheduler_t<_CPO> const get_completion_scheduler;
  template <class _CPO = void>
  extern get_completion_domain_t<_CPO> const  get_completion_domain;
  extern get_domain_t const                   get_domain;
  extern get_await_completion_adaptor_t const get_await_completion_adaptor;

  template <class _Env>
  concept __is_debug_env = __callable<__debug_env_t, _Env>;

  namespace __debug
  {
    struct __completion_signatures
    {};
  }  // namespace __debug

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // concept tag types:
  struct sender_t;
  struct operation_state_t;
  struct scheduler_t;
  struct receiver_t;

  template <class _Tag, class _Sndr, class... _Env>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto __get_completion_behavior() noexcept;

  template <class _Env>
  using __domain_of_t = __decay_t<__call_result_t<get_domain_t, _Env>>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <class... _Sigs>
  struct completion_signatures;

  template <class _Completions>
  concept __valid_completion_signatures = __ok<_Completions>
                                       && __is_instance_of<_Completions, completion_signatures>;

  struct dependent_sender_error;

  namespace __cmplsigs
  {
    struct get_completion_signatures_t;
  }  // namespace __cmplsigs

  using __cmplsigs::get_completion_signatures_t;

#if STDEXEC_NO_STDCPP_CONSTEXPR_EXCEPTIONS()

  template <class... _What, class... _Values>
  consteval auto __throw_compile_time_error(_Values...) -> __mexception<_What...>;

#else  // ^^^ no constexpr exceptions ^^^ / vvv constexpr exceptions vvv

  // C++26, https://wg21.link/p3068
  template <class _What, class... _More, class... _Values>
  consteval auto __throw_compile_time_error(_Values...) -> completion_signatures<>;

#endif  // ^^^ constexpr exceptions ^^^

  template <class... _What>
  consteval auto __throw_compile_time_error(__mexception<_What...>);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct connect_t;
  extern connect_t const connect;

  template <class _Sender, class _Receiver>
  using connect_result_t = __call_result_t<connect_t, _Sender, _Receiver>;

  template <class _Sender>
  extern bool const enable_sender;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct start_t;
  extern start_t const start;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct schedule_t;
  extern schedule_t const schedule;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct as_awaitable_t;
  extern as_awaitable_t const as_awaitable;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct transform_sender_t;
  extern transform_sender_t const transform_sender;

  template <class _Sender, class... _Env>
  using transform_sender_result_t = __call_result_t<transform_sender_t, _Sender, _Env...>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct starts_on_t;
  extern starts_on_t const starts_on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct schedule_from_t;
  extern schedule_from_t const schedule_from;

  struct continues_on_t;
  extern continues_on_t const continues_on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct bulk_t;
  struct bulk_chunked_t;
  struct bulk_unchunked_t;

  extern bulk_t const           bulk;
  extern bulk_chunked_t const   bulk_chunked;
  extern bulk_unchunked_t const bulk_unchunked;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct just_t;
  extern just_t const just;

  struct just_error_t;
  extern just_error_t const just_error;

  struct just_stopped_t;
  extern just_stopped_t const just_stopped;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct then_t;
  extern then_t const then;

  struct upon_error_t;
  extern upon_error_t const upon_error;

  struct upon_stopped_t;
  extern upon_stopped_t const upon_stopped;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct let_value_t;
  extern let_value_t const let_value;

  struct let_error_t;
  extern let_error_t const let_error;

  struct let_stopped_t;
  extern let_stopped_t const let_stopped;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct when_all_t;
  extern when_all_t const when_all;

  struct when_all_with_variant_t;
  extern when_all_with_variant_t const when_all_with_variant;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct __read_env_t;
  extern __read_env_t const read_env;

  struct __write_env_t;
  extern __write_env_t const write_env;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct into_variant_t;
  extern into_variant_t const into_variant;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct on_t;
  extern on_t const on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct affine_on_t;
  extern affine_on_t const affine_on;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct stopped_as_error_t;
  extern stopped_as_error_t const stopped_as_error;

  struct stopped_as_optional_t;
  extern stopped_as_optional_t const stopped_as_optional;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <__class _Derived>
  struct sender_adaptor_closure;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // deprecated interfaces (see __deprecations.hpp):
  struct __transfer_just_t;
  extern __transfer_just_t const __transfer_just;
}  // namespace STDEXEC

// Moved to namespace experimental::execution from namespace STDEXEC because they are no longer getting standardized:
namespace experimental::execution
{
  struct split_t;
  struct ensure_started_t;
  struct start_detached_t;
  struct __execute_t;

  extern split_t const          split;
  extern ensure_started_t const ensure_started;
  extern start_detached_t const start_detached;
  extern __execute_t const      __execute;
}  // namespace experimental::execution

namespace exec = experimental::execution;

STDEXEC_P2300_NAMESPACE_BEGIN()
  struct forwarding_query_t;
  struct get_allocator_t;
  struct get_stop_token_t;

  extern forwarding_query_t const forwarding_query;
  extern get_allocator_t const    get_allocator;
  extern get_stop_token_t const   get_stop_token;

  template <class _Env>
  using stop_token_of_t = STDEXEC::__decay_t<STDEXEC::__call_result_t<get_stop_token_t, _Env>>;

  struct never_stop_token;
  class inplace_stop_source;
  class inplace_stop_token;
  template <class _Fn>
  class inplace_stop_callback;
STDEXEC_P2300_NAMESPACE_END()

////////////////////////////////////////////////////////////////////////////////////////////////////
STDEXEC_P2300_NAMESPACE_BEGIN(this_thread)
  struct sync_wait_t;
  struct sync_wait_with_variant_t;
  extern sync_wait_t const              sync_wait;
  extern sync_wait_with_variant_t const sync_wait_with_variant;
STDEXEC_P2300_NAMESPACE_END(this_thread)

// NOT TO SPEC: make sync_wait et. al. available in namespace STDEXEC (possibly
// std::execution) as well:
namespace STDEXEC
{
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::forwarding_query_t)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::get_allocator_t)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::get_stop_token_t)

  STDEXEC_P2300_DEPRECATED_SYMBOL(std::forwarding_query)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::get_stop_token)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::get_allocator)

  STDEXEC_P2300_DEPRECATED_SYMBOL(std::stop_token_of_t)

  STDEXEC_P2300_DEPRECATED_SYMBOL(std::never_stop_token)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::inplace_stop_source)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::inplace_stop_token)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::inplace_stop_callback)

  STDEXEC_P2300_DEPRECATED_SYMBOL(std::this_thread::sync_wait_t)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::this_thread::sync_wait)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::this_thread::sync_wait_with_variant_t)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::this_thread::sync_wait_with_variant)
}  // namespace STDEXEC
