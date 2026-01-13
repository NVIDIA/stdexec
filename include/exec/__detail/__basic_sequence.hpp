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

#include "../../stdexec/__detail/__basic_sender.hpp"
#include "../../stdexec/__detail/__config.hpp"
#include "../../stdexec/__detail/__meta.hpp"

#include "../sequence_senders.hpp"

namespace exec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __seqexpr
  template <class...>
  struct __basic_sequence_sender {
    using __id = __basic_sequence_sender;
    using __t = __basic_sequence_sender;
  };

  template <auto _DescriptorFn, class = STDEXEC::__anon>
  struct __seqexpr {
    using sender_concept = sequence_sender_t;
    using __t = __seqexpr;
    using __id = __seqexpr;
    using __desc_t = decltype(_DescriptorFn());
    using __tag_t = __desc_t::__tag;
    using __captures_t =
      STDEXEC::__minvoke<__desc_t, STDEXEC::__q<STDEXEC::__detail::__captures_t>>;

    static constexpr auto __tag() noexcept -> __tag_t {
      return {};
    }

    mutable __captures_t __impl_;

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_ATTRIBUTE(host, device)
    explicit __seqexpr(_Tag, _Data&& __data, _Child&&... __child)
      : __impl_(
          STDEXEC::__detail::__captures(
            _Tag(),
            static_cast<_Data&&>(__data),
            static_cast<_Child&&>(__child)...)) {
    }

    template <STDEXEC::same_as<__seqexpr> _Self = __seqexpr>
    auto get_env() const noexcept -> decltype(_Self::__tag().get_env(*this)) {
      static_assert(noexcept(_Self::__tag().get_env(*this)));
      return _Self::__tag().get_env(*this);
    }

    template <STDEXEC::__decays_to_derived_from<__seqexpr> _Self, class... _Env>
    static consteval auto get_completion_signatures() {
      return __tag_t::template get_completion_signatures<_Self, _Env...>();
    }

    template <STDEXEC::__decays_to_derived_from<__seqexpr> _Self, class... _Env>
    static consteval auto get_item_types() {
      return __tag_t::template get_item_types<_Self, _Env...>();
    }

    template <STDEXEC::__decays_to_derived_from<__seqexpr> _Self, STDEXEC::receiver _Receiver>
    static auto subscribe(_Self&& __self, _Receiver&& __rcvr) noexcept(noexcept(
      __self.__tag().subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))))
      -> decltype(__self.__tag()
                    .subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))) {
      return __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
    }

    template <class _Sequence, class _ApplyFn>
    static auto apply(_Sequence&& __sequence, _ApplyFn&& __fun)
      noexcept(STDEXEC::__nothrow_callable<
               STDEXEC::__detail::__impl_of<_Sequence>,
               STDEXEC::__copy_cvref_fn<_Sequence>,
               _ApplyFn
      >)
        -> STDEXEC::__call_result_t<
          STDEXEC::__detail::__impl_of<_Sequence>,
          STDEXEC::__copy_cvref_fn<_Sequence>,
          _ApplyFn
        > {
      return static_cast<_Sequence&&>(__sequence)
        .__impl_(STDEXEC::__copy_cvref_fn<_Sequence>(), static_cast<_ApplyFn&&>(__fun));
    }
  };

  template <class _Tag, class _Data, class... _Child>
  STDEXEC_ATTRIBUTE(host, device)
  __seqexpr(_Tag, _Data, _Child...) -> __seqexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  template <class _Tag, class _Data, class... _Child>
  using __seqexpr_t = __seqexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  namespace __mkseqexpr {
    template <class _Tag, class _Domain = STDEXEC::default_domain>
    struct make_sequence_expr_t {
      template <class _Data = STDEXEC::__, class... _Children>
      constexpr auto operator()(_Data __data = {}, _Children... __children) const {
        return __seqexpr_t<_Tag, _Data, _Children...>{
          _Tag(), static_cast<_Data&&>(__data), static_cast<_Children&&>(__children)...};
      }
    };
  } // namespace __mkseqexpr

  struct __basic_sequence_sender_name {
    template <class _Tag, class _Data, class... _Child>
    using __result = __basic_sequence_sender<_Tag, _Data, STDEXEC::__demangle_t<_Child>...>;

    template <class _Sender>
    using __f =
      STDEXEC::__minvoke<typename STDEXEC::__decay_t<_Sender>::__desc_t, STDEXEC::__q<__result>>;
  };

  template <class _Tag, class _Domain = STDEXEC::default_domain>
  inline constexpr __mkseqexpr::make_sequence_expr_t<_Tag, _Domain> make_sequence_expr{};
} // namespace exec

namespace STDEXEC::__detail {
  template <auto _DescriptorFn>
  extern exec::__basic_sequence_sender_name __demangle_v<exec::__seqexpr<_DescriptorFn>>;
} // namespace STDEXEC::__detail
