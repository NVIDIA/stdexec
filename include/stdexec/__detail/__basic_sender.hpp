/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__meta.hpp"
#include "__type_traits.hpp"

#include "../concepts.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // Generic __sender type

  template <class _Ty>
  auto __unconst_(const _Ty&&) -> _Ty;
  template <class _Ty>
  auto __unconst_(const _Ty&) -> _Ty&;
  template <class _Ty>
  using __unconst_t = decltype(stdexec::__unconst_(__declval<_Ty>()));

  inline constexpr struct __apply_fn {
    template <class _ImplFn, class _ApplyFn>
    auto operator()(_ImplFn&& __impl, _ApplyFn&& __fun) const                                //
      noexcept(__nothrow_callable<__unconst_t<_ImplFn>, __copy_cvref_fn<_ImplFn>, _ApplyFn>) //
      -> __call_result_t<__unconst_t<_ImplFn>, __copy_cvref_fn<_ImplFn>, _ApplyFn> {         //
      return const_cast<__unconst_t<_ImplFn>&&>(__impl)(                                     //
        __copy_cvref_fn<_ImplFn>(),                                                          //
        (_ApplyFn&&) __fun);                                                                 //
    }
  } __apply{};

  struct __get_tag {
    template <class _Tag>
    _Tag operator()(_Tag, __ignore...) const noexcept;
  };

  template <class _ImplFn>
  using __tag_from = __call_result_t<__apply_fn, _ImplFn, __get_tag>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __basic_sender
  template <class _ImplFn>
  struct __basic_sender {
    using is_sender = void;
    using __tag_t = __tag_from<_ImplFn>;

    _ImplFn __impl_;

    STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
    explicit __basic_sender(_ImplFn __impl_)
      : __impl_((_ImplFn&&) __impl_) {
    }

    template <same_as<get_env_t> _Tag, same_as<__basic_sender> _Self>
    friend auto tag_invoke(_Tag, const _Self& __self) noexcept
      -> decltype(__tag_t::get_env(__self)) {
      static_assert(noexcept(__tag_t::get_env(__self)));
      return __tag_t::get_env(__self);
    }

    template <
      same_as<get_completion_signatures_t> _Tag,
      __decays_to<__basic_sender> _Self,
      class _Env>
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) //
      -> __msecond<
        __if_c<same_as<_Tag, get_completion_signatures_t>>,
        decltype(__tag_t::get_completion_signatures((_Self&&) __self, (_Env&&) __env))>;

    // BUGBUG fix receiver constraint here:
    template <
      same_as<connect_t> _Tag,
      __decays_to<__basic_sender> _Self,
      /*receiver*/ class _Receiver>
    friend auto tag_invoke(_Tag, _Self&& __self, _Receiver&& __rcvr)               //
      noexcept(noexcept(__tag_t::connect((_Self&&) __self, (_Receiver&&) __rcvr))) //
      -> __msecond<
        __if_c<same_as<_Tag, connect_t>>,
        decltype(__tag_t::connect((_Self&&) __self, (_Receiver&&) __rcvr))> {
      return __tag_t::connect((_Self&&) __self, (_Receiver&&) __rcvr);
    }
  };

  template <class _ImplFn>
  STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
  __basic_sender(_ImplFn) -> __basic_sender<_ImplFn>;

  template <class _Sender>
  using __impl_of = decltype((__declval<_Sender>().__impl_));

  template <class _Sender>
  using __tag_of = __tag_from<__impl_of<_Sender>>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __make_basic_sender
  template <class _Tag, class _Data, class... _Children>
  STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
  auto __make_basic_sender(_Data __data, _Children... __children) {
    return __basic_sender{
      [__data = (_Data&&) __data, ... __children = (_Children&&) __children] //
      <class _Cvref, class _Fun>(_Cvref, _Fun && __fun) mutable noexcept(
        __nothrow_callable<_Fun, _Tag, __minvoke<_Cvref, _Data>, __minvoke<_Cvref, _Children>...>)
        -> __call_result_t<_Fun, _Tag, __minvoke<_Cvref, _Data>, __minvoke<_Cvref, _Children>...> {
        return static_cast<_Fun&&>(__fun)(
          _Tag(),
          const_cast<__minvoke<_Cvref, decltype(__data)>&&>(__data),
          const_cast<__minvoke<_Cvref, decltype(__children)>&&>(__children)...);
      }};
  }

  namespace __detail {
    template <class _Tag, class _ImplFn>
      requires same_as<_Tag, __tag_from<_ImplFn>>
    void __test_basic_sender_for(const __basic_sender<_ImplFn>&);
  } // namespace __detail

  template <class _Sender, class _Tag>
  concept __basic_sender_for = //
    requires(_Sender& __sndr) {
      __detail::__test_basic_sender_for<_Tag>(__sndr);
    };

} // namespace stdexec
