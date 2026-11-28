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

    template <class Receiver>
    struct _state
    {
      Receiver __rcvr_;
    };

    // into_tuple(sndr): the child's value completion, which must be its only one, becomes a
    // value completion with a tuple of the decayed values. If decay-copying them can throw,
    // the exception is sent as an error.
    struct _algo : __facade::__defaults
    {
      template <class Sender, class Receiver>
      using _completions_t =
        __facade::__completions_of_t<_algo, __facade::__child<0>, Sender, Receiver>;

      template <class Sender, class Receiver>
        requires(_completions_t<Sender, Receiver>::__count(set_value) == 1)
      static constexpr auto __get_state(Sender &&, Receiver __rcvr) noexcept -> _state<Receiver>
      {
        return _state<Receiver>{static_cast<Receiver &&>(__rcvr)};
      }

      template <class Sender, class Receiver>
      static constexpr auto __get_state(Sender &&, Receiver) noexcept
      {
        using namespace STDEXEC;
        return __mexception<_WHAT_(_INVALID_ARGUMENT_),
                            _WHY_(
                              THE_INPUT_SENDER_MUST_HAVE_AT_EXACTLY_ONE_POSSIBLE_VALUE_COMPLETION),
                            _WHERE_(_IN_ALGORITHM_, into_tuple_t),
                            _WITH_SENDER_<__decay_t<Sender>>,
                            _WITH_ENVIRONMENT_(env_of_t<Receiver>)>();
      }

      template <class State, class Tag, class... Args>
      static constexpr auto
      __complete(__facade::__child<0>, State &__state, Tag __tag, Args &&...__args) noexcept
      {
        if constexpr (__same_as<Tag, set_value_t>)
        {
          return __facade::__set_value_from<into_tuple_t>(
            __state.__rcvr_,
            [&]() noexcept(__nothrow_decay_copyable<Args...>)
            { return __decayed_std_tuple<Args...>{static_cast<Args &&>(__args)...}; });
        }
        else
        {
          return __facade::__send(__state.__rcvr_, __tag, static_cast<Args &&>(__args)...);
        }
      }
    };
  }  // namespace _into_tuple

  struct into_tuple_t
  {
    template <STDEXEC::sender Sender>
    constexpr auto operator()(Sender &&__sndr) const  //
      noexcept(STDEXEC::__nothrow_decay_copyable<Sender>)
    {
      using namespace STDEXEC;
      return __facade::__sender{into_tuple_t(), __(), static_cast<Sender &&>(__sndr)};
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

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<exec::into_tuple_t> : exec::_into_tuple::_algo
  {};
}  // namespace STDEXEC
