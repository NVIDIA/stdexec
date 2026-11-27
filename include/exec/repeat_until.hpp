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

#include <exception>
#include <type_traits>

namespace experimental::execution
{
  struct repeat_t;
  struct repeat_until_t;

  struct _EXPECTING_A_SENDER_OF_ONE_VALUE_THAT_IS_CONVERTIBLE_TO_BOOL_;
  struct _EXPECTING_A_SENDER_OF_VOID_;

  namespace __repeat
  {
    template <class _Boolean, bool _Expected>
    concept __is_bool_constant = __decay_t<_Boolean>::value == _Expected;

    // repeat_until(sndr): the body completes with a value that says whether the loop is
    // done, unless it completes otherwise.
    struct __until_algo : __algo_base<__until_algo>
    {
      static constexpr bool __decays_errors = false;

      static constexpr auto __iterates(__ignore) noexcept -> bool
      {
        return true;
      }

      template <class _State, class _Children>
      static constexpr auto __start(_State &__state, _Children &) noexcept
      {
        return __until_algo::__start_iteration(__state);
      }

      using __algo_base<__until_algo>::__complete;

      template <class _State, class... _Booleans>
      static constexpr auto
      __complete(__body, _State &__state, set_value_t, _Booleans &&...__bools) noexcept
      {
        // (Each branch instantiates only the handlers it can call. Instantiating one it can't
        // call would instantiate completions that the operation does not declare, as when the
        // body always says the loop is done.)
        if constexpr (sizeof...(_Booleans) != 1 || (!__std::convertible_to<_Booleans, bool> || ...))
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_INVALID_ARGUMENT_),
                         _WHERE_(_IN_ALGORITHM_, repeat_until_t),
                         _WHY_(_EXPECTING_A_SENDER_OF_ONE_VALUE_THAT_IS_CONVERTIBLE_TO_BOOL_),
                         _WITH_SENDER_<typename _State::__child_t &>>());
        }
        else if constexpr ((__is_bool_constant<_Booleans, true> && ...))
        {
          return __until_algo::__finish(__state);  // always done
        }
        else if constexpr ((__is_bool_constant<_Booleans, false> && ...))
        {
          return __until_algo::__next(__state);  // never done
        }
        else if constexpr ((std::is_nothrow_convertible_v<_Booleans, bool> && ...))
        {
          return __facade::__choose((static_cast<bool>(static_cast<_Booleans &&>(__bools)) && ...),
                                    [&]() noexcept { return __until_algo::__finish(__state); },
                                    [&]() noexcept { return __until_algo::__next(__state); });
        }
        else
        {
          using __result_t =
            __facade::__union_t<decltype(__until_algo::__finish(__state)),
                                decltype(__until_algo::__next(__state)),
                                __facade::__effects<set_error_t(std::exception_ptr)>>;
          bool __done = false;
          STDEXEC_TRY
          {
            __done = (static_cast<bool>(static_cast<_Booleans &&>(__bools)) && ...);
          }
          STDEXEC_CATCH_ALL
          {
            __state.__cleanup();
            return __result_t(
              __facade::__send(__state.__rcvr_, set_error_t(), std::current_exception()));
          }
          return __done ? __result_t(__until_algo::__finish(__state))
                        : __result_t(__until_algo::__next(__state));
        }
      }
    };

    // repeat(sndr): the body completes with no values, and the loop is never done, unless
    // the body completes otherwise.
    struct __forever_algo : __algo_base<__forever_algo>
    {
      static constexpr bool __decays_errors = false;

      static constexpr auto __iterates(__ignore) noexcept -> bool
      {
        return true;
      }

      template <class _State, class _Children>
      static constexpr auto __start(_State &__state, _Children &) noexcept
      {
        return __forever_algo::__start_iteration(__state);
      }

      using __algo_base<__forever_algo>::__complete;

      template <class _State, class... _Args>
      static constexpr auto __complete(__body, _State &__state, set_value_t, _Args &&...) noexcept
      {
        if constexpr (sizeof...(_Args) != 0)
        {
          return __facade::__type_error(
            __mexception<_WHAT_(_INVALID_ARGUMENT_, _EXPECTING_A_SENDER_OF_VOID_),
                         _WHERE_(_IN_ALGORITHM_, repeat_t),
                         _WITH_SENDER_<typename _State::__child_t &>>());
        }
        else
        {
          return __forever_algo::__next(__state);
        }
      }
    };
  }  // namespace __repeat

  struct repeat_until_t
  {
    template <STDEXEC::sender _Sender>
    constexpr auto operator()(_Sender &&__sndr) const -> auto
    {
      return STDEXEC::__check_well_formed(
        STDEXEC::__facade::__sender{repeat_until_t(),
                                    STDEXEC::__(),
                                    static_cast<_Sender &&>(__sndr)});
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()() const
    {
      return STDEXEC::__closure(*this);
    }
  };

  inline constexpr repeat_until_t repeat_until{};

  struct repeat_t
  {
    template <STDEXEC::sender _Sender>
    constexpr auto operator()(_Sender &&__sndr) const -> auto
    {
      return STDEXEC::__check_well_formed(
        STDEXEC::__facade::__sender{repeat_t(), STDEXEC::__(), static_cast<_Sender &&>(__sndr)});
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()() const
    {
      return STDEXEC::__closure(*this);
    }
  };

  inline constexpr repeat_t repeat{};

  //! deprecated interfaces
  using repeat_effect_t [[deprecated("use exec::repeat_t instead")]]             = repeat_t;
  using repeat_effect_until_t [[deprecated("use exec::repeat_until_t instead")]] = repeat_until_t;
  [[deprecated("use exec::repeat instead")]]
  inline constexpr repeat_t const &repeat_effect = repeat;
  [[deprecated("use exec::repeat_until instead")]]
  inline constexpr repeat_until_t const &repeat_effect_until = repeat_until;
}  // namespace experimental::execution

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<exec::repeat_until_t> : exec::__repeat::__until_algo
  {};

  template <>
  struct __facade::__algorithm<exec::repeat_t> : exec::__repeat::__forever_algo
  {};
}  // namespace STDEXEC
