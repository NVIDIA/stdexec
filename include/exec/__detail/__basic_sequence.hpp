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
  struct __basic_sequence_sender { };

  namespace {
    template <auto _DescriptorFn>
    struct __seqexpr
      : STDEXEC::__minvoke<decltype(_DescriptorFn()), STDEXEC::__qq<STDEXEC::__tuple>> {
      using sender_concept = sequence_sender_t;
      using __desc_t = decltype(_DescriptorFn());
      using __tag_t = __desc_t::__tag;

      static constexpr auto __tag() noexcept -> __tag_t {
        return {};
      }

      template <class _Self = __seqexpr>
      auto get_env() const noexcept -> decltype(_Self::__tag().get_env(*this)) {
        static_assert(noexcept(_Self::__tag().get_env(*this)));
        return _Self::__tag().get_env(*this);
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(STDEXEC::__decays_to_derived_from<_Self, __seqexpr>);
        return __tag_t::template get_completion_signatures<_Self, _Env...>();
      }

      template <class _Self, class... _Env>
      static consteval auto get_item_types() {
        static_assert(STDEXEC::__decays_to_derived_from<_Self, __seqexpr>);
        return __tag_t::template get_item_types<_Self, _Env...>();
      }

      template <class _Self, STDEXEC::receiver _Receiver>
      static auto subscribe(_Self&& __self, _Receiver&& __rcvr) noexcept(noexcept(
        __self.__tag().subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))))
        -> decltype(__self.__tag()
                      .subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr))) {
        static_assert(STDEXEC::__decays_to_derived_from<_Self, __seqexpr>);
        return __tag_t::subscribe(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
      }
    };

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __seqexpr(_Tag, _Data, _Child...)
      -> __seqexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;
  } // namespace

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

  template <class _Tag, class _Data, class... _Child>
  using __basic_sequence_sender_t =
    __basic_sequence_sender<_Tag, _Data, STDEXEC::__demangle_t<_Child>...>;

  template <class _Tag, class _Domain = STDEXEC::default_domain>
  inline constexpr __mkseqexpr::make_sequence_expr_t<_Tag, _Domain> make_sequence_expr{};
} // namespace exec

namespace STDEXEC::__detail {
  template <auto _DescriptorFn>
  extern __declfn_t<__minvoke<__result_of<_DescriptorFn>, __q<exec::__basic_sequence_sender_t>>>
    __demangle_v<exec::__seqexpr<_DescriptorFn>>;
} // namespace STDEXEC::__detail
