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

#include "../sequence_senders.hpp"

#include "../../stdexec/__detail/__basic_sender.hpp"

namespace exec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __seqexpr
  template <class...>
  struct __seqexpr {
    using __id = __seqexpr;
    using __t = __seqexpr;
  };

  template <class _ImplFn>
  struct __seqexpr<_ImplFn> {
    using sender_concept = sequence_sender_t;
    using __t = __seqexpr;
    using __id = __seqexpr;
    using __tag_t = stdexec::__call_result_t<_ImplFn, stdexec::__cp, stdexec::__detail::__get_tag>;

    static __tag_t __tag() noexcept {
      return {};
    }

    mutable _ImplFn __impl_;

    STDEXEC_ATTRIBUTE((host, device))
    explicit __seqexpr(_ImplFn __impl)
      : __impl_((_ImplFn&&) __impl) {
    }

    template <stdexec::same_as<stdexec::get_env_t> _Tag, stdexec::same_as<__seqexpr> _Self>
    friend auto tag_invoke(_Tag, const _Self& __self) noexcept //
      -> stdexec::__msecond<
        stdexec::__if_c<stdexec::same_as<_Tag, stdexec::get_env_t>>, //
        decltype(__self.__tag().get_env(__self))> {
      static_assert(noexcept(__self.__tag().get_env(__self)));
      return __tag_t::get_env(__self);
    }

    template <
      stdexec::same_as<stdexec::get_completion_signatures_t> _Tag,
      stdexec::__decays_to<__seqexpr> _Self,
      class _Env>
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) //
      -> stdexec::__msecond<
        stdexec::__if_c<stdexec::same_as<_Tag, stdexec::get_completion_signatures_t>>,
        decltype(__self.__tag().get_completion_signatures((_Self&&) __self, (_Env&&) __env))> {
      return {};
    }

    template <
      stdexec::same_as<get_item_types_t> _Tag,
      stdexec::__decays_to<__seqexpr> _Self,
      class _Env>
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) //
      -> stdexec::__msecond<
        stdexec::__if_c<stdexec::same_as<_Tag, get_item_types_t>>,
        decltype(__self.__tag().get_item_types((_Self&&) __self, (_Env&&) __env))> {
      return {};
    }

    template <
      stdexec::same_as<subscribe_t> _Tag,
      stdexec::__decays_to<__seqexpr> _Self,
      /*receiver*/ class _Receiver>
    friend auto tag_invoke(_Tag, _Self&& __self, _Receiver&& __rcvr)                       //
      noexcept(noexcept(__self.__tag().subscribe((_Self&&) __self, (_Receiver&&) __rcvr))) //
      -> stdexec::__msecond<
        stdexec::__if_c<stdexec::same_as<_Tag, subscribe_t>>,
        decltype(__self.__tag().subscribe((_Self&&) __self, (_Receiver&&) __rcvr))> {
      return __tag_t::subscribe((_Self&&) __self, (_Receiver&&) __rcvr);
    }

    template <class _Sender, class _ApplyFn>
    STDEXEC_DEFINE_EXPLICIT_THIS_MEMFN(auto apply)(this _Sender&& __sndr, _ApplyFn&& __fun) //
      noexcept(stdexec::__nothrow_callable<
               stdexec::__detail::__impl_of<_Sender>,
               stdexec::__copy_cvref_fn<_Sender>,
               _ApplyFn>) //
      -> stdexec::__call_result_t<
        stdexec::__detail::__impl_of<_Sender>,
        stdexec::__copy_cvref_fn<_Sender>,
        _ApplyFn> { //
      return ((_Sender&&) __sndr)
        .__impl_(stdexec::__copy_cvref_fn<_Sender>(), (_ApplyFn&&) __fun); //
    }
  };

  template <class _ImplFn>
  STDEXEC_ATTRIBUTE((host, device))
  __seqexpr(_ImplFn) -> __seqexpr<_ImplFn>;

#if STDEXEC_NVHPC() || (STDEXEC_GCC() && __GNUC__ < 13)
  namespace __detail {
    template <class _Tag, class _Domain = stdexec::default_domain>
    struct make_sequence_expr_t {
      template <class _Data = stdexec::__, class... _Children>
      constexpr auto operator()(_Data __data = {}, _Children... __children) const {
        return __seqexpr{stdexec::__detail::__make_tuple(
          _Tag(), stdexec::__detail::__mbc(__data), stdexec::__detail::__mbc(__children)...)};
      }
    };
  }
#else
  namespace __detail {
    template <class _Tag, class _Domain = stdexec::default_domain>
    struct make_sequence_expr_t {
      template <class _Data = stdexec::__, class... _Children>
      constexpr auto operator()(_Data __data = {}, _Children... __children) const {
        return __seqexpr{
          stdexec::__detail::__make_tuple(_Tag(), (_Data&&) __data, (_Children&&) __children...)};
      }
    };
  }
#endif

  template <class _Tag, class _Domain = stdexec::default_domain>
  inline constexpr __detail::make_sequence_expr_t<_Tag, _Domain> make_sequence_expr{};

  template <class _Tag, class _Data, class... _Children>
  using __seqexpr_t = stdexec::__result_of<make_sequence_expr<_Tag>, _Data, _Children...>;
}