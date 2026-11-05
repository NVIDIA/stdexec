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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "__concepts.hpp"
#  include "__optional.hpp"

#  if !STDEXEC_USE_MODULES()
#    if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
#      include <stop_token>  // IWYU pragma: export
#    endif

// This shouldn't be necessary, but some standard library implementations claim
// support for jthread but don't actually provide std::stop_token.
STDEXEC_NAMESPACE_STD_BEGIN
  class stop_token;
  template <class _Callback>
  class stop_callback;
STDEXEC_NAMESPACE_STD_END
#  endif

#  include "__prologue.hpp"

STDEXEC_P2300_NAMESPACE_BEGIN()

  template <template <class> class _Callback>
  struct __check_type_alias_exists;

  template <class _StopToken>
  inline constexpr bool __has_stop_callback_v = requires {
    typename __check_type_alias_exists<_StopToken::template callback_type>;
  };

#  if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
  template <>
  inline constexpr bool __has_stop_callback_v<std::stop_token> = true;
#  endif

  template <class _Token>
  struct __stop_callback_for
  {
    template <class _Callback>
    using __f = _Token::template callback_type<_Callback>;
  };

#  if defined(__cpp_lib_jthread) && __cpp_lib_jthread >= 201911L
  template <>
  struct __stop_callback_for<std::stop_token>
  {
    template <class _Callback>
    using __f = std::stop_callback<_Callback>;
  };
#  endif

  STDEXEC_MODULE_EXPORT template <class _Token, class _Callback>
  using stop_callback_for_t = STDEXEC::__mcall1<__stop_callback_for<_Token>, _Callback>;

  template <class _Token, class _Stoppable, class _Tag>
  class __stoppable_base_impl
  {
    struct __cb
    {
      void operator()() const noexcept
      {
        __self_->__on_stop_();
      }

      __stoppable_base_impl* __self_;
    };

    using __callback_t = stop_callback_for_t<_Token, __cb>;

   protected:
    __stoppable_base_impl()                        = default;
    __stoppable_base_impl(__stoppable_base_impl&&) = delete;

    void register_stoppable(_Token __token)
      noexcept(STDEXEC::__nothrow_constructible_from<__callback_t, _Token, __cb>)
    {
      STDEXEC_ASSERT(!__cb_.has_value());
      __cb_.emplace(static_cast<_Token&&>(__token), __cb{this});
    }

    void deregister_stoppable() noexcept
    {
      __cb_.reset();
    }

   private:
    void __on_stop_() noexcept
    {
      if constexpr (STDEXEC::__same_as<_Tag, void>)
        static_cast<_Stoppable*>(this)->on_stop_requested();
      else
        static_cast<_Stoppable*>(this)->on_stop_requested(_Tag{});
    }

    STDEXEC::__optional<__callback_t> __cb_;
  };

  template <class _Token, class _Stoppable, class _Tag>
  class __unstoppable_base
  {
   protected:
    __unstoppable_base() = default;

    void register_stoppable(_Token) noexcept {}

    void deregister_stoppable() noexcept {}
  };

  STDEXEC_MODULE_EXPORT template <class _Token>
  struct __stoppable_base_for
  {
    template <class _Stoppable, class _Tag>
    using __f = __stoppable_base_impl<_Token, _Stoppable, _Tag>;
  };

  STDEXEC_MODULE_EXPORT template <class _Token, class _Stoppable, class _Tag = void>
  using __stoppable_base_for_t =
    typename __stoppable_base_for<_Token>::template __f<_Stoppable, _Tag>;

  STDEXEC_MODULE_EXPORT template <class _Token>
  concept stoppable_token =
    requires(_Token const __token) {
      requires __has_stop_callback_v<_Token>;
      { __token.stop_requested() } noexcept -> STDEXEC::__boolean_testable_;
      { __token.stop_possible() } noexcept -> STDEXEC::__boolean_testable_;
      { _Token(__token) } noexcept;
    } && STDEXEC::__std::copyable<_Token>  //
    && STDEXEC::__std::equality_comparable<_Token>;

  STDEXEC_MODULE_EXPORT template <class _Token>
  concept unstoppable_token =
    stoppable_token<_Token> //
    && requires {
      { _Token::stop_possible() } -> STDEXEC::__boolean_testable_;
    } //
    && (!_Token::stop_possible());

  template <unstoppable_token _Token>
  struct __stoppable_base_for<_Token>
  {
    template <class _Stoppable, class _Tag>
    using __f = __unstoppable_base<_Token, _Stoppable, _Tag>;
  };

  // [stoptoken.never], class never_stop_token
  STDEXEC_MODULE_EXPORT struct never_stop_token
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

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
