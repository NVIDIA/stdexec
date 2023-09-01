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
  namespace __detail {
    template <class _Sender>
    using __impl_of = decltype((__declval<_Sender>().__impl_));

    struct __get_tag {
      template <class _Tag, class... _Rest>
      _Tag operator()(_Tag, _Rest&&...) const noexcept {
        return {};
      }
    };

    struct __get_data {
      template <class _Data, class... _Rest>
      _Data&& operator()(__ignore, _Data&& __data, _Rest&&...) const noexcept {
        return (_Data&&) __data;
      }
    };

    template <class _Continuation>
    struct __get_children {
      template <class... _Children>
      auto operator()(__ignore, __ignore, _Children&&...) const noexcept
        -> __mtype<__minvoke<_Continuation, _Children...>> (*)() {
        return nullptr;
      }
    };
  } // namespace __detail

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __basic_sender
  template <class...>
  struct __basic_sender {
    using __id = __basic_sender;
    using __t = __basic_sender;
  };

  template <class _ImplFn>
  struct __basic_sender<_ImplFn> {
    using is_sender = void;
    using __t = __basic_sender;
    using __id = __basic_sender;
    using __tag_t = __call_result_t<_ImplFn, __cp, __detail::__get_tag>;

    static __tag_t __tag() noexcept {
      return {};
    }

    mutable _ImplFn __impl_;

    STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
      explicit __basic_sender(_ImplFn __impl)
      : __impl_((_ImplFn&&) __impl) {
    }

    template <same_as<get_env_t> _Tag, same_as<__basic_sender> _Self>
    friend auto tag_invoke(_Tag, const _Self& __self) noexcept //
      -> __msecond<
        __if_c<same_as<_Tag, get_env_t>>, //
        decltype(__self.__tag().get_env(__self))> {
      static_assert(noexcept(__self.__tag().get_env(__self)));
      return __tag_t::get_env(__self);
    }

    template <
      same_as<get_completion_signatures_t> _Tag,
      __decays_to<__basic_sender> _Self,
      class _Env>
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) //
      -> __msecond<
        __if_c<same_as<_Tag, get_completion_signatures_t>>,
        decltype(__self.__tag().get_completion_signatures((_Self&&) __self, (_Env&&) __env))>;

    // BUGBUG fix receiver constraint here:
    template <
      same_as<connect_t> _Tag,
      __decays_to<__basic_sender> _Self,
      /*receiver*/ class _Receiver>
    friend auto tag_invoke(_Tag, _Self&& __self, _Receiver&& __rcvr)                     //
      noexcept(noexcept(__self.__tag().connect((_Self&&) __self, (_Receiver&&) __rcvr))) //
      -> __msecond<
        __if_c<same_as<_Tag, connect_t>>,
        decltype(__self.__tag().connect((_Self&&) __self, (_Receiver&&) __rcvr))> {
      return __tag_t::connect((_Self&&) __self, (_Receiver&&) __rcvr);
    }

    template <class _Sender, class _ApplyFn>
    STDEXEC_DEFINE_EXPLICIT_THIS_MEMFN(auto apply)(this _Sender&& __sndr, _ApplyFn&& __fun) //
      noexcept(
        __nothrow_callable<__detail::__impl_of<_Sender>, __copy_cvref_fn<_Sender>, _ApplyFn>) //
      -> __call_result_t<__detail::__impl_of<_Sender>, __copy_cvref_fn<_Sender>, _ApplyFn> {  //
      return ((_Sender&&) __sndr).__impl_(__copy_cvref_fn<_Sender>(), (_ApplyFn&&) __fun);    //
    }
  };

  template <class _ImplFn>
  STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
    __basic_sender(_ImplFn) -> __basic_sender<_ImplFn>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __make_basic_sender
  namespace __detail {
    struct __make_basic_sender_ {
      template <class _Tag, class _Data = __, class... _Children>
      constexpr auto operator()(_Tag, _Data __data = {}, _Children... __children) const;
    };

#if STDEXEC_NVHPC()
    // The NVIDIA HPC compiler struggles with capture initializers for a parameter pack.
    // As a workaround, we use a wrapper that performs moves when non-const lvalues are
    // copied. That constructor is only used when capturing the variables, never when
    // the resulting lambda is copied or moved.

    // Move-by-copy
    template <class _Ty>
    struct __mbc {
      template <class _Cvref>
      using __f = __minvoke<_Cvref, _Ty>;

      _Ty __value;

      explicit __mbc(_Ty& __v) noexcept(std::is_nothrow_move_constructible_v<_Ty>)
        : __value((_Ty&&) __v) {
      }

      // This is a template so as to not be considered a copy/move constructor. Therefore,
      // it doesn't suppress the generation of the default copy/move constructors.
      __mbc(same_as<__mbc> auto& __that) noexcept(std::is_nothrow_move_constructible_v<_Ty>)
        : __value(static_cast<_Ty&&>(__that.__value)) {
      }
    };

    // Anonymous namespace here is to avoid symbol name collisions with the
    // lambda functions returned by __make_tuple.
    namespace {
      constexpr auto __make_tuple = //
        []<class _Tag, class... _Captures>(_Tag, _Captures&&... __captures) {
          return [=]<class _Cvref, class _Fun>(_Cvref __cvref, _Fun && __fun) mutable      //
                 noexcept(__nothrow_callable<_Fun, _Tag, __minvoke<_Captures, _Cvref>...>) //
                 -> decltype(auto)                                                         //
                   requires __callable<_Fun, _Tag, __minvoke<_Captures, _Cvref>...>
          {
            return ((_Fun&&) __fun)(
              _Tag(), const_cast<__minvoke<_Captures, _Cvref>&&>(__captures.__value)...);
          };
        };
    } // anonymous namespace

    template <class _Tag, class _Data, class... _Children>
    constexpr auto
      __make_basic_sender_::operator()(_Tag, _Data __data, _Children... __children) const {
      return __basic_sender{
        __detail::__make_tuple(_Tag(), __detail::__mbc(__data), __detail::__mbc(__children)...)};
    }
#else
    // Anonymous namespace here is to avoid symbol name collisions with the
    // lambda functions returned by __make_tuple.
    namespace {
      [[gnu::deprecated]] void printsMe(auto&&...) {
      }

      template <typename... Ts>
      void printAllTypes(Ts&&... ts) {
        (printsMe(ts), ...);
      }
      constexpr auto __make_tuple = //
        []<class _Tag, class... _Captures>(_Tag tag, _Captures&&... __captures) {
          return [... __captures = (_Captures&&) __captures]<class _Cvref, class _Fun>(
                   _Cvref ref, _Fun && __fun) mutable                                          //
                 noexcept(__nothrow_callable<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>) //
                 -> decltype(auto)                                                         //
                   requires __callable<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>
          { 
            printsMe(ref);
            auto val = ((_Fun&&) __fun)(
              _Tag(), const_cast<__minvoke<_Cvref, _Captures>&&>(__captures)...);
              printsMe(val);
            return ((_Fun&&) __fun)(
              _Tag(), const_cast<__minvoke<_Cvref, _Captures>&&>(__captures)...);
          };
        };
    } // anonymous namespace

    template <class _Tag, class _Data, class... _Children>
    constexpr auto
      __make_basic_sender_::operator()(_Tag tag, _Data __data, _Children... __children) const {
        auto sender = __basic_sender{
        __detail::__make_tuple(_Tag(), (_Data&&) __data, (_Children&&) __children...)};
      return sender;
    };
#endif
  } // namespace __detail

  inline constexpr __detail::__make_basic_sender_ __make_basic_sender{};

  namespace __detail {
    struct __sender_apply_fn {
      template <class _Sender, class _ApplyFn>
      auto operator()(_Sender&& __sndr, _ApplyFn&& __fun) const //
        noexcept(noexcept(
          STDEXEC_CALL_EXPLICIT_THIS_MEMFN(((_Sender&&) __sndr), apply)((_ApplyFn&&) __fun))) //
        -> decltype(STDEXEC_CALL_EXPLICIT_THIS_MEMFN(((_Sender&&) __sndr), apply)(
          (_ApplyFn&&) __fun)) {                                                                  //
        return STDEXEC_CALL_EXPLICIT_THIS_MEMFN(((_Sender&&) __sndr), apply)((_ApplyFn&&) __fun); //
      }
    };
  } // namespace __detail

  using __detail::__sender_apply_fn;
  inline constexpr __sender_apply_fn __sender_apply{};

  template <class _Sender, class _ApplyFn>
  using __sender_apply_result_t = __call_result_t<__sender_apply_fn, _Sender, _ApplyFn>;

  template <class _Sender>
  using __tag_of = __call_result_t<__sender_apply_fn, _Sender, __detail::__get_tag>;

  template <class _Sender>
  using __data_of = __call_result_t<__sender_apply_fn, _Sender, __detail::__get_data>;

  template <class _Sender, class _Continuation = __q<__types>>
  using __children_of = __t<__call_result_t<
    __call_result_t<__sender_apply_fn, _Sender, __detail::__get_children<_Continuation>>>>;

  template <class _Ny, class _Sender>
  using __nth_child_of = __children_of<_Sender, __mbind_front_q<__m_at, _Ny>>;

  template <std::size_t _Ny, class _Sender>
  using __nth_child_of_c = __children_of<_Sender, __mbind_front_q<__m_at, __msize_t<_Ny>>>;

  template <class _Sender>
  using __child_of = __children_of<_Sender, __q<__mfront>>;

  template <class _Sender>
  inline constexpr std::size_t __nbr_children_of = __v<__children_of<_Sender, __msize>>;

  template <class _Sender, class _Tag>
  concept __lazy_sender_for = //
    same_as<__tag_of<_Sender>, _Tag>;

  namespace __detail {
    template <class _Sender>
    extern __q<__midentity> __name_of_v;

    template <class _Sender>
    using __name_of_fn = decltype(__name_of_v<_Sender>);

    template <class _Sender>
    using __name_of = __minvoke<__name_of_fn<_Sender>, _Sender>;

    struct __lazy_sender_name {
      template <class _Sender>
      using __f = //
        __call_result_t<__sender_apply_result_t<_Sender, __lazy_sender_name>>;

      template <class _Tag, class _Data, class... _Children>
      auto operator()(_Tag, _Data&&, _Children&&...) const //
        -> __basic_sender<_Tag, _Data, __name_of<_Children>...> (*)();
    };

    struct __id_name {
      template <class _Sender>
      using __f = __name_of<__id<_Sender>>;
    };

    template <class _Sender>
    extern __mcompose<__cplr, __name_of_fn<_Sender>> __name_of_v<_Sender&>;

    template <class _Sender>
    extern __mcompose<__cprr, __name_of_fn<_Sender>> __name_of_v<_Sender&&>;

    template <class _Sender>
    extern __mcompose<__cpclr, __name_of_fn<_Sender>> __name_of_v<const _Sender>;

    template <class _ImplOf>
    extern __lazy_sender_name __name_of_v<__basic_sender<_ImplOf>>;

    template <__has_id _Sender>
      requires(!same_as<__id<_Sender>, _Sender>)
    extern __id_name __name_of_v<_Sender>;
  } // namespace __detail

  template <class _Sender>
  using __name_of = __detail::__name_of<_Sender>;
} // namespace stdexec
