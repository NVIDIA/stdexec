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

// include these after __execution_fwd.hpp
#include "__basic_sender.hpp"
#include "__env.hpp"
#include "__queries.hpp"
#include "__sender_adaptor_closure.hpp"

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write
  {
    // write_env can change the scheduler in the receiver's environment, which
    // invalidates the child sender's completion behavior claims (e.g., a task
    // that reports "asynchronous_affine" is affine to *its* scheduler, not to
    // the scheduler that write_env substitutes). So we must not forward the
    // child's completion behavior; we report __unknown instead.
    template <class _Sender>
    struct __write_env_attrs
    {
      template <__forwarding_query _Query, class... _Args>
        requires(!__completion_query<_Query>)
             && __queryable_with<env_of_t<_Sender>, _Query, _Args...>
      [[nodiscard]]
      constexpr auto query(_Query, _Args &&...__args) const
        noexcept(__nothrow_queryable_with<env_of_t<_Sender>, _Query, _Args...>)
          -> __query_result_t<env_of_t<_Sender>, _Query, _Args...>
      {
        return __query<_Query>()(get_env(__sndr_), static_cast<_Args &&>(__args)...);
      }

      _Sender const &__sndr_;
    };

    template <class _Sender>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __write_env_attrs(_Sender const &) -> __write_env_attrs<_Sender>;

    struct __write_env_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Child>(__ignore, __ignore, _Child const &__child) noexcept
      {
        return __write_env_attrs{__child};
      };

      static constexpr auto __get_env = []<class _State>(__ignore, _State const &__state) noexcept
        -> decltype(__env::__join(__state.__data_, STDEXEC::get_env(__state.__rcvr_)))
      {
        return __env::__join(__state.__data_, STDEXEC::get_env(__state.__rcvr_));
      };

      template <class _Self, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Self, __write_env_t>);
        return STDEXEC::get_completion_signatures<
          __child_of<_Self>,
          __minvoke_q<__join_env_t, __decay_t<__data_of<_Self>> const &, _Env>...>();
      }
    };
  }  // namespace __write

  struct __write_env_t
  {
    template <sender _Sender, class _Env>
    constexpr auto operator()(_Sender &&__sndr, _Env __env) const
    {
      return __make_sexpr<__write_env_t>(static_cast<_Env &&>(__env),
                                         static_cast<_Sender &&>(__sndr));
    }

    template <class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Env __env) const
    {
      return __closure(*this, static_cast<_Env &&>(__env));
    }
  };

  inline constexpr __write_env_t write_env{};

  template <>
  struct __sexpr_impl<__write_env_t> : __write::__write_env_impl
  {};
}  // namespace STDEXEC
