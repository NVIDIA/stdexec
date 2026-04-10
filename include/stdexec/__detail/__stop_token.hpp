/*
 * Copyright (c) 2021-2022 Facebook, Inc. and its affiliates
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

#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
#  include <stop_token>  // IWYU pragma: export
#endif

// This shouldn't be necessary, but some standard library implementations claim support
// for jthread but don't actually provide std::stop_token.
STDEXEC_NAMESPACE_STD_BEGIN
  class stop_token;
  template <class _Callback>
  class stop_callback;
STDEXEC_NAMESPACE_STD_END

STDEXEC_P2300_NAMESPACE_BEGIN()

  template <template <class> class _Callback>
  struct __check_type_alias_exists;

  template <class _StopToken>
  inline constexpr bool __has_stop_callback_v = requires {
    typename __check_type_alias_exists<_StopToken::template callback_type>;
  };

#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
  template <>
  inline constexpr bool __has_stop_callback_v<std::stop_token> = true;
#endif

  template <class _Token>
  struct __stop_callback_for
  {
    template <class _Callback>
    using __f = _Token::template callback_type<_Callback>;
  };

#if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
  template <>
  struct __stop_callback_for<std::stop_token>
  {
    template <class _Callback>
    using __f = std::stop_callback<_Callback>;
  };
#endif

  template <class _Token, class _Callback>
  using stop_callback_for_t = STDEXEC::__mcall1<__stop_callback_for<_Token>, _Callback>;

  template <class _Token>
  concept stoppable_token =
    requires(_Token const __token) {
      requires __has_stop_callback_v<_Token>;
      { __token.stop_requested() } noexcept -> STDEXEC::__boolean_testable_;
      { __token.stop_possible() } noexcept -> STDEXEC::__boolean_testable_;
      { _Token(__token) } noexcept;
    } && STDEXEC::__std::copyable<_Token>  //
    && STDEXEC::__std::equality_comparable<_Token>;

  template <class _Token>
  concept unstoppable_token =
    stoppable_token<_Token> //
    && requires {
      { _Token::stop_possible() } -> STDEXEC::__boolean_testable_;
    } //
    && (!_Token::stop_possible());

  // [stoptoken.never], class never_stop_token
  struct never_stop_token
  {
   private:
    struct __callback_type
    {
      constexpr explicit __callback_type(never_stop_token, STDEXEC::__ignore) noexcept {}
    };
   public:
    template <class>
    using callback_type = __callback_type;

    static constexpr auto stop_requested() noexcept -> bool
    {
      return false;
    }

    static constexpr auto stop_possible() noexcept -> bool
    {
      return false;
    }

    constexpr auto operator==(never_stop_token const &) const noexcept -> bool = default;
  };
STDEXEC_P2300_NAMESPACE_END()

namespace STDEXEC
{
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::stop_callback_for_t)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::stoppable_token)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::unstoppable_token)
  STDEXEC_P2300_DEPRECATED_SYMBOL(std::never_stop_token)
}  // namespace STDEXEC
