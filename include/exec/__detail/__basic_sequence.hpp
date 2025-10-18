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
#include "stdexec/__detail/__completion_signatures.hpp"
#include "stdexec/__detail/__concepts.hpp"
#include "stdexec/__detail/__debug.hpp"
#include <type_traits>

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

    static constexpr auto __tag() noexcept -> __tag_t {
      return {};
    }

    mutable __captures_t __impl_;

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_ATTRIBUTE(host, device)
    explicit __seqexpr(_Tag, _Data&& __data, _Child&&... __child)
      : __impl_(
          stdexec::__detail::__captures(
            _Tag(),
            static_cast<_Data&&>(__data),
            static_cast<_Child&&>(__child)...)) {
    }

    template <stdexec::same_as<__seqexpr> _Self = __seqexpr>
    auto get_env() const noexcept -> decltype(_Self::__tag().get_env(*this)) {
      static_assert(noexcept(_Self::__tag().get_env(*this)));
      return _Self::__tag().get_env(*this);
    }

    // make sure that get_completion_signatures does not SFINAE out
    // when the trailing return-type is invalid but keep the
    // trailing return-type when it is valid
    struct get_completion_signatures_sfinae {
      template <stdexec::__decays_to<__seqexpr> _Self, class... _Env>
      auto operator()(_Self&& __self, _Env&&... __env) const
        -> decltype(__self.__tag().get_completion_signatures(
          static_cast<_Self&&>(__self),
          static_cast<_Env&&>(__env)...)) {
        return {};
      }
    };
    template <stdexec::__decays_to<__seqexpr> _Self, class... _Env>
      requires
        (stdexec::__is_debug_env<_Env> || ... || false)
        || (!stdexec::__callable<get_completion_signatures_sfinae, _Self, _Env...>)
    static auto get_completion_signatures(_Self&& __self, _Env&&... __env) {
      return __self.__tag().get_completion_signatures(
        static_cast<_Self&&>(__self),
        static_cast<_Env&&>(__env)...);
    }
    template <stdexec::__decays_to<__seqexpr> _Self, class... _Env>
      requires (!stdexec::__is_debug_env<_Env> && ... && true)
    static auto get_completion_signatures(_Self&& __self, _Env&&... __env)
      -> decltype(__self.__tag().get_completion_signatures(
        static_cast<_Self&&>(__self),
        static_cast<_Env&&>(__env)...)) {
      return {};
    }

    // make sure that get_item_types does not SFINAE out
    // when the trailing return-type is invalid but keep the
    // trailing return-type when it is valid
    struct get_item_types_sfinae {
      template <stdexec::__decays_to<__seqexpr> _Self, class... _Env>
      auto operator()(_Self&& __self, _Env&&... __env) const
      -> decltype(__self.__tag().get_item_types(
        static_cast<_Self&&>(__self),
        static_cast<_Env&&>(__env)...)) {
        return {};
      }
    };
    template <stdexec::__decays_to<__seqexpr> _Self, class... _Env>
      requires
        (stdexec::__is_debug_env<_Env> || ... || false)
        || (!stdexec::__callable<get_item_types_sfinae, _Self, _Env...>)
    static auto get_item_types(_Self&& __self, _Env&&... __env) {
      return __self.__tag().get_item_types(
        static_cast<_Self&&>(__self),
        static_cast<_Env&&>(__env)...);
    }
    template <stdexec::__decays_to<__seqexpr> _Self, class... _Env>
      requires (!stdexec::__is_debug_env<_Env> && ... && true)
    static auto get_item_types(_Self&& __self, _Env&&... __env)
      -> decltype(__self.__tag().get_item_types(
        static_cast<_Self&&>(__self),
        static_cast<_Env&&>(__env)...)) {
      return {};
    }

    // make sure that subscribe does not SFINAE out
    // when the trailing return-type is invalid but keep the
    // trailing return-type when it is valid
    struct subscribe_sfinae {
      template <stdexec::__decays_to<__seqexpr> _Self, stdexec::receiver _Receiver>
      auto operator()(_Self&& __self, _Receiver&& __rcvr) const noexcept(noexcept(
        __self.__tag().subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))))
        -> decltype(__self.__tag()
                      .subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))) {
        return __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
      }
    };
    template <stdexec::__decays_to<__seqexpr> _Self, stdexec::receiver _Receiver>
      requires
        stdexec::__is_debug_env<stdexec::env_of_t<_Receiver>>
        || (!stdexec::__callable<subscribe_sfinae, _Self, _Receiver>)
    static auto subscribe(_Self&& __self, _Receiver&& __rcvr) noexcept(noexcept(
      __self.__tag().subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr)))) {
      return __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
    }
    template <stdexec::__decays_to<__seqexpr> _Self, stdexec::receiver _Receiver>
      requires (!stdexec::__is_debug_env<stdexec::env_of_t<_Receiver>>)
    static auto subscribe(_Self&& __self, _Receiver&& __rcvr) noexcept(noexcept(
      __self.__tag().subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))))
      -> decltype(__self.__tag()
                    .subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))) {
      return __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
    }

    template <class _Sequence, class _ApplyFn>
    static auto
      apply(_Sequence&& __sequence, _ApplyFn&& __fun) noexcept(stdexec::__nothrow_callable<
                                                         stdexec::__detail::__impl_of<_Sequence>,
                                                         stdexec::__copy_cvref_fn<_Sequence>,
                                                         _ApplyFn
      >)
        -> stdexec::__call_result_t<
          stdexec::__detail::__impl_of<_Sequence>,
          stdexec::__copy_cvref_fn<_Sequence>,
          _ApplyFn
        > {
      return static_cast<_Sequence&&>(__sequence)
        .__impl_(stdexec::__copy_cvref_fn<_Sequence>(), static_cast<_ApplyFn&&>(__fun));
    }
  };

  template <class _Tag, class _Data, class... _Child>
  STDEXEC_ATTRIBUTE(host, device)
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

  struct __basic_sequence_sender_name {
    template <class _Tag, class _Data, class... _Child>
    using __result = __basic_sequence_sender<_Tag, _Data, stdexec::__name_of<_Child>...>;

    template <class _Sender>
    using __f =
      stdexec::__minvoke<typename stdexec::__decay_t<_Sender>::__desc_t, stdexec::__q<__result>>;
  };

  template <class _Tag, class _Domain = stdexec::default_domain>
  inline constexpr __mkseqexpr::make_sequence_expr_t<_Tag, _Domain> make_sequence_expr{};
} // namespace exec

namespace stdexec::__detail {
  template <auto _DescriptorFn>
  extern exec::__basic_sequence_sender_name __name_of_v<exec::__seqexpr<_DescriptorFn>>;
} // namespace stdexec::__detail
