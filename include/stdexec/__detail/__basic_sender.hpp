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

#include <utility> // for tuple_size/tuple_element

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

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-local-typedef")

    struct __get_meta {
      template <class _Tag, class _Data, class... _Children>
      constexpr auto operator()(_Tag, _Data&&, _Children&&...) const noexcept {
        struct __meta {
          using __tag = _Tag;
          using __data = _Data;
          using __children = __types<_Children...>;
        };

        return __meta{};
      }
    };

    STDEXEC_PRAGMA_POP()

    struct __tie {
      template <class _Tag, class _Data, class... _Children>
      constexpr auto operator()(_Tag, _Data&& __data, _Children&&... __children) const noexcept {
        return std::tuple<_Tag, _Data&&, _Children&&...>{
          {}, (_Data&&) __data, (_Children&&) __children...};
      }
    };
  } // namespace __detail

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __sexpr
  template <class...>
  struct __sexpr {
    using __id = __sexpr;
    using __t = __sexpr;
  };

  template <class _ImplFn>
  struct __sexpr<_ImplFn> {
    using is_sender = void;
    using __t = __sexpr;
    using __id = __sexpr;
    using __meta_t = __call_result_t<_ImplFn, __cp, __detail::__get_meta>;
    using __tag_t = typename __meta_t::__tag;
    using __data_t = typename __meta_t::__data;
    using __children_t = typename __meta_t::__children;
    using __arity_t = __mapply<__msize, __children_t>;

    STDEXEC_ATTRIBUTE((always_inline)) //
    static __tag_t __tag() noexcept {
      return {};
    }

    mutable _ImplFn __impl_;

    STDEXEC_ATTRIBUTE((host, device, always_inline))
    explicit __sexpr(_ImplFn __impl)
      : __impl_((_ImplFn&&) __impl) {
    }

    template <same_as<get_env_t> _Tag, same_as<__sexpr> _Self>
    STDEXEC_ATTRIBUTE((always_inline))                         //
    friend auto tag_invoke(_Tag, const _Self& __self) noexcept //
      -> __msecond<
        __if_c<same_as<_Tag, get_env_t>>, //
        decltype(__self.__tag().get_env(__self))> {
      static_assert(noexcept(__self.__tag().get_env(__self)));
      return __tag_t::get_env(__self);
    }

    template < same_as<get_completion_signatures_t> _Tag, __decays_to<__sexpr> _Self, class _Env>
    STDEXEC_ATTRIBUTE((always_inline))                         //
    friend auto tag_invoke(_Tag, _Self&& __self, _Env&& __env) //
      -> __msecond<
        __if_c<same_as<_Tag, get_completion_signatures_t>>,
        decltype(__self.__tag().get_completion_signatures((_Self&&) __self, (_Env&&) __env))> {
      return {};
    }

    // BUGBUG fix receiver constraint here:
    template <
      same_as<connect_t> _Tag,
      __decays_to<__sexpr> _Self,
      /*receiver*/ class _Receiver>
    STDEXEC_ATTRIBUTE((always_inline))                                                   //
    friend auto tag_invoke(_Tag, _Self&& __self, _Receiver&& __rcvr)                     //
      noexcept(noexcept(__self.__tag().connect((_Self&&) __self, (_Receiver&&) __rcvr))) //
      -> __msecond<
        __if_c<same_as<_Tag, connect_t>>,
        decltype(__self.__tag().connect((_Self&&) __self, (_Receiver&&) __rcvr))> {
      return __tag_t::connect((_Self&&) __self, (_Receiver&&) __rcvr);
    }

    template <class _Sender, class _ApplyFn>
    STDEXEC_ATTRIBUTE((always_inline))                                                      //
    STDEXEC_DEFINE_EXPLICIT_THIS_MEMFN(auto apply)(this _Sender&& __sndr, _ApplyFn&& __fun) //
      noexcept(
        __nothrow_callable<__detail::__impl_of<_Sender>, __copy_cvref_fn<_Sender>, _ApplyFn>) //
      -> __call_result_t<__detail::__impl_of<_Sender>, __copy_cvref_fn<_Sender>, _ApplyFn> {  //
      return ((_Sender&&) __sndr).__impl_(__copy_cvref_fn<_Sender>(), (_ApplyFn&&) __fun);    //
    }

    template <std::size_t _Idx, __decays_to_derived_from<__sexpr> _Self>
    STDEXEC_ATTRIBUTE((always_inline))
    friend decltype(auto) get(_Self&& __self) noexcept
      requires(_Idx < (__v<__arity_t> + 2))
    {
      if constexpr (_Idx == 0) {
        return __tag_t();
      } else {
        return __self.__impl_(__copy_cvref_fn<_Self>(), __nth_pack_element<_Idx>);
      }
      STDEXEC_UNREACHABLE();
    }
  };

  template <class _ImplFn>
  STDEXEC_ATTRIBUTE((host, device))
  __sexpr(_ImplFn) -> __sexpr<_ImplFn>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __make_sexpr
  namespace __detail {
    template <class _Tag>
    struct __make_sexpr_t {
      template <class _Data = __, class... _Children>
      constexpr auto operator()(_Data __data = {}, _Children... __children) const;
    };

#if STDEXEC_NVHPC() || (STDEXEC_GCC() && __GNUC__ < 13)
    // The NVIDIA HPC compiler and gcc prior to v13 struggle with capture
    // initializers for a parameter pack. As a workaround, we use a wrapper that
    // performs moves when non-const lvalues are copied. That constructor is
    // only used when capturing the variables, never when the resulting lambda
    // is copied or moved.

    // Move-by-copy
    template <class _Ty>
    struct __mbc {
      template <class _Cvref>
      using __f = __minvoke<_Cvref, _Ty>;

      _Ty __value;

      STDEXEC_ATTRIBUTE((always_inline))
      explicit __mbc(_Ty& __v) noexcept(std::is_nothrow_move_constructible_v<_Ty>)
        : __value((_Ty&&) __v) {
      }

      // This is a template so as to not be considered a copy/move constructor. Therefore,
      // it doesn't suppress the generation of the default copy/move constructors.
      STDEXEC_ATTRIBUTE((always_inline))
      __mbc(same_as<__mbc> auto& __that) noexcept(std::is_nothrow_move_constructible_v<_Ty>)
        : __value(static_cast<_Ty&&>(__that.__value)) {
      }
    };

    // Rather strange definition of the lambda return type below is to reap the
    // benefits of SFINAE without nvc++ encoding the whole return type into the
    // symbol name.
    template <class _Ty>
    extern _Ty (*__f)();

    // Anonymous namespace here is to avoid symbol name collisions with the
    // lambda functions returned by __make_tuple.
    namespace {
      constexpr auto __make_tuple = //
        []<class _Tag, class... _Captures>(_Tag, _Captures&&... __captures) {
          return [=]<class _Cvref, class _Fun>(_Cvref __cvref, _Fun && __fun) mutable      //
                 noexcept(__nothrow_callable<_Fun, _Tag, __minvoke<_Captures, _Cvref>...>) //
                 -> decltype(__f<__call_result_t<_Fun, _Tag, __minvoke<_Captures, _Cvref>...>>())
                   requires __callable<_Fun, _Tag, __minvoke<_Captures, _Cvref>...>
          {
            return ((_Fun&&) __fun)(
              _Tag(), const_cast<__minvoke<_Captures, _Cvref>&&>(__captures.__value)...);
          };
        };
    } // anonymous namespace

    template <class _Tag>
    template <class _Data, class... _Children>
    constexpr auto __make_sexpr_t<_Tag>::operator()(_Data __data, _Children... __children) const {
      return __sexpr{__make_tuple(_Tag(), __detail::__mbc(__data), __detail::__mbc(__children)...)};
    }
#else
    // Anonymous namespace here is to avoid symbol name collisions with the
    // lambda functions returned by __make_tuple.
    namespace {
      constexpr auto __make_tuple = //
        []<class _Tag, class... _Captures>(_Tag, _Captures&&... __captures) {
          return [... __captures = (_Captures&&) __captures]<class _Cvref, class _Fun>(
                   _Cvref, _Fun && __fun) mutable                                          //
                 noexcept(__nothrow_callable<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>) //
                 -> __call_result_t<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>
                   requires __callable<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>
          {
            return ((_Fun&&) __fun)(
              _Tag(), const_cast<__minvoke<_Cvref, _Captures>&&>(__captures)...);
          };
        };
    } // anonymous namespace

    template <class _Tag>
    template <class _Data, class... _Children>
    constexpr auto __make_sexpr_t<_Tag>::operator()(_Data __data, _Children... __children) const {
      return __sexpr{__make_tuple(_Tag(), (_Data&&) __data, (_Children&&) __children...)};
    };
#endif

    template <class _Tag>
    inline constexpr __make_sexpr_t<_Tag> __make_sexpr{};
  } // namespace __detail

  using __detail::__make_sexpr;

  template <class _Tag, class _Data, class... _Children>
  using __sexpr_t = __result_of<__make_sexpr<_Tag>, _Data, _Children...>;

  namespace __detail {
    struct __sexpr_apply_t {
      template <class _Sender, class _ApplyFn>
      STDEXEC_ATTRIBUTE((always_inline))                        //
      auto operator()(_Sender&& __sndr, _ApplyFn&& __fun) const //
        noexcept(noexcept(
          STDEXEC_CALL_EXPLICIT_THIS_MEMFN(((_Sender&&) __sndr), apply)((_ApplyFn&&) __fun))) //
        -> decltype(STDEXEC_CALL_EXPLICIT_THIS_MEMFN(((_Sender&&) __sndr), apply)(
          (_ApplyFn&&) __fun)) {
        return STDEXEC_CALL_EXPLICIT_THIS_MEMFN(((_Sender&&) __sndr), apply)((_ApplyFn&&) __fun); //
      }
    };
  } // namespace __detail

  using __detail::__sexpr_apply_t;
  inline constexpr __sexpr_apply_t __sexpr_apply{};

  template <class _Sender, class _ApplyFn>
  using __sexpr_apply_result_t = __call_result_t<__sexpr_apply_t, _Sender, _ApplyFn>;

  namespace __detail {
    template <class _Sender>
    using __meta_of = __call_result_t<__sexpr_apply_t, _Sender, __detail::__get_meta>;
  }

  template <class _Sender>
  using __tag_of = typename __detail::__meta_of<_Sender>::__tag;

  template <class _Sender>
  using __data_of = typename __detail::__meta_of<_Sender>::__data;

  template <class _Sender, class _Continuation = __q<__types>>
  using __children_of = //
    __mapply< _Continuation, typename __detail::__meta_of<_Sender>::__children>;

  template <class _Ny, class _Sender>
  using __nth_child_of = __children_of<_Sender, __mbind_front_q<__m_at, _Ny>>;

  template <std::size_t _Ny, class _Sender>
  using __nth_child_of_c = __children_of<_Sender, __mbind_front_q<__m_at, __msize_t<_Ny>>>;

  template <class _Sender>
  using __child_of = __children_of<_Sender, __q<__mfront>>;

  template <class _Sender>
  inline constexpr std::size_t __nbr_children_of = __v<__children_of<_Sender, __msize>>;

  template <class _Sender>
  concept sender_expr = //
    __mvalid<__tag_of, _Sender>;

  template <class _Sender, class _Tag>
  concept sender_expr_for = //
    sender_expr<_Sender> && same_as<__tag_of<_Sender>, _Tag>;

  // The __name_of utility defined below is used to pretty-print the type names of
  // senders in compiler diagnostics.
  namespace __detail {
    template <class _Sender>
    extern __q<__midentity> __name_of_v;

    template <class _Sender>
    using __name_of_fn = decltype(__name_of_v<_Sender>);

    template <class _Sender>
    using __name_of = __minvoke<__name_of_fn<_Sender>, _Sender>;

    struct __basic_sender_name {
      template <class _Sender>
      using __f = //
        __call_result_t<__sexpr_apply_result_t<_Sender, __basic_sender_name>>;

      template <class _Tag, class _Data, class... _Children>
      auto operator()(_Tag, _Data&&, _Children&&...) const //
        -> __sexpr<_Tag, _Data, __name_of<_Children>...> (*)();
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
    extern __mcompose<__cpclr, __name_of_fn<_Sender>> __name_of_v<const _Sender&>;

    template <class _Impl>
    extern __basic_sender_name __name_of_v<__sexpr<_Impl>>;

    template <__has_id _Sender>
      requires(!same_as<__id<_Sender>, _Sender>)
    extern __id_name __name_of_v<_Sender>;

    template <class _Ty>
    _Ty __remove_rvalue_reference_fn(_Ty&&);

    template <class _Ty>
    using __remove_rvalue_reference_t =
      decltype(__detail::__remove_rvalue_reference_fn(__declval<_Ty>()));
  } // namespace __detail

  template <class _Sender>
  using __name_of = __detail::__name_of<_Sender>;
} // namespace stdexec

namespace std {
  template <class _Impl>
  struct tuple_size<stdexec::__sexpr<_Impl>>
    : integral_constant< size_t, stdexec::__v<typename stdexec::__sexpr<_Impl>::__arity_t> + 2> { };

  template <size_t _Idx, class _Impl>
  struct tuple_element<_Idx, stdexec::__sexpr<_Impl>> {
    using type = stdexec::__detail::__remove_rvalue_reference_t<
      stdexec::__call_result_t<_Impl, stdexec::__cp, stdexec::__nth_pack_element_t<_Idx>>>;
  };
}
