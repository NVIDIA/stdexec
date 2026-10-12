/*
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

#include "../stdexec/execution.hpp"
#include "completion_signatures.hpp"

namespace experimental::execution
{
  namespace _into_tuple
  {
    using namespace STDEXEC;

    struct into_tuple_t;
    struct THE_INPUT_SENDER_MUST_HAVE_AT_EXACTLY_ONE_POSSIBLE_VALUE_COMPLETION;

    struct _impl : __sexpr_defaults
    {
      template <class Receiver>
      struct __state
      {
        Receiver __rcvr_;
      };

      static constexpr auto _value_cmplsigs = []<class... Args>() noexcept
      {
        constexpr bool __is_nothrow = __nothrow_decay_copyable<Args...>;
        return exec::concat_completion_signatures(
          STDEXEC::completion_signatures<set_value_t(__decayed_std_tuple<Args...>)>(),
          __eptr_completion_unless_t<__mbool<__is_nothrow>>());
      };

      template <class Sender, class... Env>
      static consteval auto __get_completion_signatures()
      {
        auto completions = STDEXEC::get_completion_signatures<__child_of<Sender>, Env...>();
        STDEXEC_IF_OK(completions)
        {
          if constexpr (completions.__count(set_value) != 1)
          {
            return exec::throw_compile_time_error<
              _WHAT_(_INVALID_ARGUMENT_),
              _WHY_(THE_INPUT_SENDER_MUST_HAVE_AT_EXACTLY_ONE_POSSIBLE_VALUE_COMPLETION),
              _WHERE_(_IN_ALGORITHM_, into_tuple_t),
              _WITH_SENDER_<Sender>,
              _WITH_ENVIRONMENT_(Env)...>();
          }
          else
          {
            return exec::transform_completion_signatures(completions, _value_cmplsigs);
          }
        }
      }

      static constexpr auto __get_state = []<class Receiver>(__ignore, Receiver &&__rcvr) noexcept
      {
        return __state<Receiver>{static_cast<Receiver &&>(__rcvr)};
      };

      static constexpr auto __complete =
        []<class State, class Tag, class... Args>(__ignore,
                                                  State &__state,
                                                  Tag,
                                                  Args &&...__args) noexcept -> void
      {
        if constexpr (Tag() == set_value)
        {
          STDEXEC_TRY
          {
            set_value(static_cast<State &&>(__state).__rcvr_,
                      __decayed_std_tuple<Args...>{static_cast<Args &&>(__args)...});
          }
          STDEXEC_CATCH_ALL
          {
            if constexpr (__nothrow_decay_copyable<Args...>)
              __std::unreachable();
            else
              set_error(static_cast<State &&>(__state).__rcvr_, std::current_exception());
          }
        }
        else
        {
          Tag()(static_cast<State &&>(__state).__rcvr_, static_cast<Args &&>(__args)...);
        }
      };
    };
  }  // namespace _into_tuple

  struct into_tuple_t
  {
    template <STDEXEC::sender Sender>
    constexpr auto operator()(Sender &&__sndr) const  //
      noexcept(STDEXEC::__nothrow_decay_copyable<Sender>)
    {
      using namespace STDEXEC;
      return __sexpr{into_tuple_t(), __(), static_cast<Sender &&>(__sndr)};
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()() const noexcept
    {
      return STDEXEC::__closure(*this);
    }
  };

  inline constexpr into_tuple_t into_tuple{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace STDEXEC
{
  template <>
  struct __sexpr_impl<exec::into_tuple_t> : exec::_into_tuple::_impl
  {};
}  // namespace STDEXEC
