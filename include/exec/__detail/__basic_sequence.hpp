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
#include "../../stdexec/__detail/__meta.hpp"
#include "../../stdexec/__detail/__basic_sender.hpp"

#include "../sequence_senders.hpp"

namespace exec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __seqexpr
  template <class...>
  struct __basic_sequence_sender {
    using __id = __basic_sequence_sender;
    using __t = __basic_sequence_sender;
  };

  template <auto _DescriptorFn, class = stdexec::__anon>
  struct __seqexpr {
    using sender_concept = sequence_sender_t;
    using __t = __seqexpr;
    using __id = __seqexpr;
    using __desc_t = decltype(_DescriptorFn());
    using __tag_t = typename __desc_t::__tag;
    using __captures_t =
      stdexec::__minvoke<__desc_t, stdexec::__q<stdexec::__detail::__captures_t>>;

    static auto __tag() noexcept -> __tag_t {
      return {};
    }

    mutable __captures_t __impl_;

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_ATTRIBUTE((host, device))
    explicit __seqexpr(_Tag, _Data&& __data, _Child&&... __child)
      : __impl_(stdexec::__detail::__captures(
        _Tag(),
        static_cast<_Data&&>(__data),
        static_cast<_Child&&>(__child)...)) {
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
        decltype(__self.__tag().get_completion_signatures(
          static_cast<_Self&&>(__self),
          static_cast<_Env&&>(__env)))> {
      return {};
    }

    template <stdexec::same_as<get_item_types_t> _Tag, stdexec::__decays_to<__seqexpr> _Self, class _Env>
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) //
      -> stdexec::__msecond<
        stdexec::__if_c<stdexec::same_as<_Tag, get_item_types_t>>,
        decltype(__self.__tag()
                   .get_item_types(static_cast<_Self&&>(__self), static_cast<_Env&&>(__env)))> {
      return {};
    }

    template <
      stdexec::same_as<subscribe_t> _Tag,
      stdexec::__decays_to<__seqexpr> _Self,
      /*receiver*/ class _Receiver>
    friend auto tag_invoke(_Tag, _Self&& __self, _Receiver&& __rcvr) //
      noexcept(noexcept(__self.__tag().subscribe(
        static_cast<_Self&&>(__self),
        static_cast<_Receiver&&>(__rcvr)))) //
      -> stdexec::__msecond<
        stdexec::__if_c<stdexec::same_as<_Tag, subscribe_t>>,
        decltype(__self.__tag()
                   .subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr)))> {
      return __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
    }

    template <class _Sender, class _ApplyFn>
    static auto apply(_Sender&& __sndr, _ApplyFn&& __fun) //
      noexcept(stdexec::__nothrow_callable<
               stdexec::__detail::__impl_of<_Sender>,
               stdexec::__copy_cvref_fn<_Sender>,
               _ApplyFn>) //
      -> stdexec::__call_result_t<
        stdexec::__detail::__impl_of<_Sender>,
        stdexec::__copy_cvref_fn<_Sender>,
        _ApplyFn> { //
      return static_cast<_Sender&&>(__sndr).__impl_(
        stdexec::__copy_cvref_fn<_Sender>(), static_cast<_ApplyFn&&>(__fun)); //
    }
  };

  template <class _Tag, class _Data, class... _Child>
  STDEXEC_ATTRIBUTE((host, device))
  __seqexpr(_Tag, _Data, _Child...) -> __seqexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  template <class _Tag, class _Data, class... _Child>
  using __seqexpr_t = __seqexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  namespace __mkseqexpr {
    template <class _Tag, class _Domain = stdexec::default_domain>
    struct make_sequence_expr_t {
      template <class _Data = stdexec::__, class... _Children>
      constexpr auto operator()(_Data __data = {}, _Children... __children) const {
        return __seqexpr_t<_Tag, _Data, _Children...>{
          _Tag(), static_cast<_Data&&>(__data), static_cast<_Children&&>(__children)...};
      }
    };
  } // namespace __mkseqexpr

  template <class _Tag, class _Domain = stdexec::default_domain>
  inline constexpr __mkseqexpr::make_sequence_expr_t<_Tag, _Domain> make_sequence_expr{};
} // namespace exec