/*
 * Copyright (c) 2023 Runner-2019
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "../stdexec/__detail/__sender_facade.hpp"
#include "../stdexec/execution.hpp"

#include "detail/repeat.hpp"

#include <cstddef>

namespace experimental::execution
{
  struct repeat_n_t;
  struct _THE_INPUT_SENDER_MUST_HAVE_VOID_VALUE_COMPLETION_;

  namespace __repeat_n
  {
    using namespace STDEXEC;

    // repeat_n(sndr, count): the body completes with no values count times, unless it
    // completes otherwise first.
    struct __algo : __repeat::__algo_base<__algo>
    {
      static constexpr bool __decays_errors = true;

      static constexpr auto __iterates(std::size_t __count) noexcept -> bool
      {
        return __count != 0;
      }

      // Zero repetitions complete without starting the child.
      template <class _State, class _Children>
      static constexpr auto __start(_State &__state, _Children &) noexcept
      {
        return __facade::__choose(
          __state.__data_ != 0,
          [&]() noexcept { return __algo::__start_iteration(__state); },
          [&]() noexcept { return __facade::__send(__state.__rcvr_, set_value_t()); });
      }

      using __repeat::__algo_base<__algo>::__complete;

      template <class _State, class... _Args>
      static constexpr auto
      __complete(__repeat::__body, _State &__state, set_value_t, _Args &&...) noexcept
      {
        if constexpr (sizeof...(_Args) != 0)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_INVALID_ARGUMENT_),
                         _WHERE_(_IN_ALGORITHM_, repeat_n_t),
                         _WHY_(_THE_INPUT_SENDER_MUST_HAVE_VOID_VALUE_COMPLETION_),
                         _WITH_SENDER_<typename _State::__child_t &>>());
        }
        else
        {
          return __facade::__choose(
            --__state.__data_ == 0,
            [&]() noexcept { return __algo::__finish(__state); },
            [&]() noexcept { return __algo::__next(__state); });
        }
      }
    };
  }  // namespace __repeat_n

  struct repeat_n_t
  {
    template <STDEXEC::sender _Sender>
    constexpr auto operator()(_Sender &&__sndr, std::size_t __count) const  //
      -> auto
    {
      return STDEXEC::__check_well_formed(
        STDEXEC::__facade::__sender{repeat_n_t(), __count, static_cast<_Sender &&>(__sndr)});
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(std::size_t __count) const noexcept
    {
      return STDEXEC::__closure(*this, __count);
    }
  };

  inline constexpr repeat_n_t repeat_n{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<exec::repeat_n_t> : exec::__repeat_n::__algo
  {};
}  // namespace STDEXEC
