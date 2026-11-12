/*
 * Copyright (c) 2023 NVIDIA Corporation
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../stdexec/__detail/__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)
import stdexec;
#else
#  include "../sequence_senders.hpp"

#  include "../../stdexec/__detail/__prologue.hpp"

namespace experimental::execution
{
  //////////////////////////////////////////////////////////////////////////////
  // __seqexpr
  template <class _Tag, class _Data, class... _Child>
  struct __seqexpr : STDEXEC::__tuple<_Tag, _Data, _Child...>
  {
    using sender_concept = sequence_sender_tag;
    using __tag_t        = _Tag;
    using __data_t       = _Data;
    using __children_t   = STDEXEC::__tuple<_Child...>;

    STDEXEC_ATTRIBUTE(always_inline)
    static constexpr auto __tag() noexcept -> __tag_t
    {
      return {};
    }

    template <class _Self = __seqexpr>
    auto get_env() const noexcept -> decltype(_Self::__tag().get_env(*this))
    {
      static_assert(noexcept(_Self::__tag().get_env(*this)));
      return _Self::__tag().get_env(*this);
    }

    template <class _Self, class... _Env>
    static consteval auto get_completion_signatures()
    {
      static_assert(STDEXEC::__decays_to_derived_from<_Self, __seqexpr>);
      return __tag_t::template get_completion_signatures<_Self, _Env...>();
    }

    template <class _Self, class... _Env>
    static consteval auto get_item_types()
    {
      static_assert(STDEXEC::__decays_to_derived_from<_Self, __seqexpr>);
      return __tag_t::template get_item_types<_Self, _Env...>();
    }

    // clang-format off
      template <class _Self, STDEXEC::receiver _Receiver>
      static constexpr auto __static_subscribe(_Self&& __self, _Receiver __rcvr) STDEXEC_AUTO_RETURN
      (
        __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))
      )

      template <STDEXEC::receiver _Receiver>
      constexpr auto subscribe(_Receiver __rcvr) && STDEXEC_AUTO_RETURN
      (
        __tag_t::subscribe(static_cast<__seqexpr&&>(*this), static_cast<_Receiver&&>(__rcvr))
      )

      template <STDEXEC::receiver _Receiver>
      constexpr auto subscribe(_Receiver __rcvr) const & STDEXEC_AUTO_RETURN
      (
        __tag_t::subscribe(*this, static_cast<_Receiver&&>(__rcvr))
      )
    // clang-format on
  };

  template <class _Tag, class _Data, class... _Child>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  __seqexpr(_Tag, _Data, _Child...) -> __seqexpr<_Tag, _Data, _Child...>;

  namespace __mkseqexpr
  {
    template <class _Tag, class _Domain = STDEXEC::default_domain>
    struct make_sequence_expr_t
    {
      template <class _Data = STDEXEC::__, class... _Children>
      constexpr auto operator()(_Data __data = {}, _Children... __children) const
      {
        return __seqexpr<_Tag, _Data, _Children...>{_Tag(),
                                                    static_cast<_Data&&>(__data),
                                                    static_cast<_Children&&>(__children)...};
      }
    };
  }  // namespace __mkseqexpr

  template <class _Tag, class _Domain = STDEXEC::default_domain>
  inline constexpr __mkseqexpr::make_sequence_expr_t<_Tag, _Domain> make_sequence_expr{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

#  include "../../stdexec/__detail/__epilogue.hpp"
#endif
