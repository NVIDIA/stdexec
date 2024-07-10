/*
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

#include "__execution_fwd.hpp"

#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__senders_core.hpp"
#include "__sender_introspection.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

#include <utility> // for tuple_size/tuple_element
#include <cstddef>
#include <type_traits>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // Generic __sender type
  namespace __detail {
    template <class _Sender>
    using __impl_of = decltype((__declval<_Sender>().__impl_));

    struct __get_data {
      template <class _Data>
      STDEXEC_ATTRIBUTE((always_inline))
      _Data&&
        operator()(__ignore, _Data&& __data, auto&&...) const noexcept {
        return static_cast<_Data&&>(__data);
      }
    };
  } // namespace __detail

  namespace {
    template <
      class _Descriptor,
      auto _DescriptorFn =
        [] {
          return _Descriptor();
        }>
    inline constexpr auto __descriptor_fn_v = _DescriptorFn;

    template <class _Tag, class _Data, class... _Child>
    inline constexpr auto __descriptor_fn() {
      return __descriptor_fn_v<__detail::__desc<_Tag, _Data, _Child...>>;
    }
  } // namespace

#if STDEXEC_NVHPC()
#  define STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child)                                            \
    stdexec::__descriptor_fn<_Tag, _Data, _Child>()
#else
#  define STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child)                                            \
    stdexec::__descriptor_fn_v<stdexec::__detail::__desc<_Tag, _Data, _Child>>
#endif

  template <class _Tag>
  struct __sexpr_impl;

  namespace __detail {
    template <class _Sexpr, class _Receiver>
    struct __op_state;

    template <class _Sexpr, class _Receiver>
    struct __connect_fn;

    template <class _Tag, class _Sexpr, class _Receiver>
    using __state_type_t =
      __decay_t<__result_of<__sexpr_impl<_Tag>::get_state, _Sexpr, _Receiver&>>;

    template <class _Self, class _Tag, class _Index, class _Sexpr, class _Receiver>
    using __env_type_t = __result_of<
      __sexpr_impl<__meval<__msecond, _Self, _Tag>>::get_env,
      _Index,
      __state_type_t<__meval<__msecond, _Self, _Tag>, _Sexpr, _Receiver>&,
      _Receiver&>;

    template <class _Sexpr, class _Receiver>
    concept __connectable =
      __callable<__impl_of<_Sexpr>, __copy_cvref_fn<_Sexpr>, __connect_fn<_Sexpr, _Receiver>>
      && __mvalid<__state_type_t, tag_of_t<_Sexpr>, _Sexpr, _Receiver>;

    // // Note: This is UB. UBSAN allows it for now.
    // template <class _Parent, class _Child>
    // _Parent* __parent_from_child(_Child* __child, _Child _Parent::*__mbr_ptr) noexcept {
    //   alignas(_Parent) char __buf[sizeof(_Parent)];
    //   _Parent* __parent = (_Parent*) &__buf;
    //   const std::ptrdiff_t __offset = (char*) &(__parent->*__mbr_ptr) - __buf;
    //   return (_Parent*) (static_cast<char*>(__child) - __offset);
    // }

    inline constexpr auto __get_attrs = //
      [](__ignore, const auto&... __child) noexcept -> decltype(auto) {
      if constexpr (sizeof...(__child) == 1) {
        return stdexec::get_env(__child...); // BUGBUG: should be only the forwarding queries
      } else {
        return empty_env();
      }
    };

    inline constexpr auto __get_env = //
      []<class _Receiver>(__ignore, __ignore, const _Receiver& __rcvr) noexcept
      -> env_of_t<const _Receiver&> {
      return stdexec::get_env(__rcvr);
    };

    inline constexpr auto __get_state = //
      []<class _Sender>(_Sender&& __sndr, __ignore) noexcept -> decltype(auto) {
      return __sndr.apply(static_cast<_Sender&&>(__sndr), __get_data());
    };

    inline constexpr auto __connect =                                        //
      []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver __rcvr) //
      noexcept(__nothrow_constructible_from<__op_state<_Sender, _Receiver>, _Sender, _Receiver>)
      -> __op_state<_Sender, _Receiver>
      requires __connectable<_Sender, _Receiver>
    {
      return __op_state<_Sender, _Receiver>{
        static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)};
    };

    inline constexpr auto __start = //
      []<class _StartTag = start_t, class... _ChildOps>(
        __ignore,
        __ignore,
        _ChildOps&... __ops) noexcept {
      (_StartTag()(__ops), ...);
    };

    inline constexpr auto __complete = //
      []<class _Index, class _Receiver, class _SetTag, class... _Args>(
        _Index,
        __ignore,
        _Receiver& __rcvr,
        _SetTag,
        _Args&&... __args) noexcept {
        static_assert(__v<_Index> == 0, "I don't know how to complete this operation.");
        _SetTag()(std::move(__rcvr), static_cast<_Args&&>(__args)...);
      };

    inline constexpr auto __sigs = //
      []<class _Sender>(_Sender&& __sndr, __ignore = {}) noexcept {
        static_assert(
          __mnever<tag_of_t<_Sender>>,
          "No customization of get_completion_signatures for this sender tag type.");
      };

    template <class _ReceiverId, class _Sexpr, class _Idx>
    struct __receiver {
      struct __t {
        using receiver_concept = receiver_t;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using __sexpr = _Sexpr;
        using __index = _Idx;
        using __id = __receiver;
        using __parent_op_t = __op_state<_Sexpr, _Receiver>;
        using __tag_t = tag_of_t<_Sexpr>;

        // A pointer to the parent operation state, which contains the one created with
        // this receiver.
        __parent_op_t* __op_;

        // template <class _ChildSexpr, class _ChildReceiver>
        // static __t __from_op_state(__op_state<_ChildSexpr, _ChildReceiver>* __child) noexcept {
        //   using __parent_op_t = __op_state<_Sexpr, _Receiver>;
        //   std::ptrdiff_t __offset = __parent_op_t::template __get_child_op_offset<__v<_Idx>>();
        //   __parent_op_t* __parent = (__parent_op_t*) (static_cast<char*>(__child) - __offset);
        //   return __t{__parent};
        // }

        template <class... _Args>
        STDEXEC_ATTRIBUTE((always_inline))
        void
          set_value(_Args&&... __args) noexcept {
          __op_->__complete(_Idx(), stdexec::set_value, static_cast<_Args&&>(__args)...);
        }

        template <class _Error>
        STDEXEC_ATTRIBUTE((always_inline))
        void
          set_error(_Error&& __err) noexcept {
          __op_->__complete(_Idx(), stdexec::set_error, static_cast<_Error&&>(__err));
        }

        STDEXEC_ATTRIBUTE((always_inline))
        void
          set_stopped() noexcept {
          __op_->__complete(_Idx(), stdexec::set_stopped);
        }

        template <__same_as<__t> _Self = __t>
        STDEXEC_ATTRIBUTE((always_inline))
        auto
          get_env() const noexcept -> __env_type_t<_Self, __tag_t, _Idx, _Sexpr, _Receiver> {
          return __op_->__get_env(_Idx());
        }
      };
    };

    // template <class _Receiver>
    // using __sexpr_connected_with = __mapply<
    //   __mbind_front_q<__m_at, typename _Receiver::__index>,
    //   typename __call_result_t<__impl_of<typename _Receiver::__sexpr>, __cp, __get_desc>::__children>;

    template <class _Sexpr, class _Receiver>
    struct __op_base : __immovable {
      using __tag_t = typename __decay_t<_Sexpr>::__tag_t;
      using __state_t = __state_type_t<__tag_t, _Sexpr, _Receiver>;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS __state_t __state_;

      __op_base(_Sexpr&& __sndr, _Receiver&& __rcvr) //
        noexcept(__nothrow_decay_copyable<_Receiver> && __nothrow_move_constructible<__state_t>)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __state_(__sexpr_impl<__tag_t>::get_state(static_cast<_Sexpr&&>(__sndr), __rcvr_)) {
      }

      auto __rcvr() & noexcept -> _Receiver& {
        return __rcvr_;
      }

      auto __rcvr() const & noexcept -> const _Receiver& {
        return __rcvr_;
      }
    };

    // template <class _Sexpr, class _Receiver>
    //   requires __is_instance_of<__id<_Receiver>, __receiver>
    //         && __decays_to<_Sexpr, __sexpr_connected_with<_Receiver>>
    // struct __op_base<_Sexpr, _Receiver> : __immovable {
    //   using __tag_t = typename __decay_t<_Sexpr>::__tag_t;
    //   using __state_t = __state_type_t<__tag_t, _Sexpr, _Receiver>;

    //   STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS __state_t __state_;

    //   __op_base(_Sexpr&& __sndr, _Receiver&& __rcvr)
    //     : __state_(__sexpr_impl<__tag_t>::get_state(static_cast<_Sexpr&&>(__sndr), __rcvr)) {
    //     STDEXEC_ASSERT(this->__rcvr().__op_ == __rcvr.__op_);
    //   }

    //   _Receiver __rcvr() const noexcept {
    //     return _Receiver::__from_op_state(             //
    //       static_cast<__op_state<_Sexpr, _Receiver>*>( //
    //         const_cast<__op_base*>(this)));
    //   }
    // };

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Winvalid-offsetof")
    STDEXEC_PRAGMA_IGNORE_EDG(offset_in_non_POD_nonstandard)

    template <class _Sexpr, class _Receiver>
    struct __enable_receiver_from_this {
      using __op_base_t = __op_base<_Sexpr, _Receiver>;

      auto __receiver() noexcept -> decltype(auto) {
        using __derived_t = decltype(__op_base_t::__state_);
        auto* __derived = static_cast<__derived_t*>(this);
        constexpr std::size_t __offset = offsetof(__op_base_t, __state_);
        auto* __base =
          reinterpret_cast<__op_base_t*>(reinterpret_cast<char*>(__derived) - __offset);
        return __base->__rcvr();
      }
    };

    STDEXEC_PRAGMA_POP()

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

    template <class _Sexpr, class _Receiver>
    struct __connect_fn {
      template <std::size_t _Idx>
      using __receiver_t = __t<__receiver<__id<_Receiver>, _Sexpr, __msize_t<_Idx>>>;

      __op_state<_Sexpr, _Receiver>* __op_;

      struct __impl {
        __op_state<_Sexpr, _Receiver>* __op_;

        template <std::size_t... _Is, class... _Child>
        auto operator()(__indices<_Is...>, _Child&&... __child) const
          noexcept((__nothrow_connectable<_Child, __receiver_t<_Is>> && ...))
            -> __tuple_for<connect_result_t<_Child, __receiver_t<_Is>>...> {
          return __tuple{connect(static_cast<_Child&&>(__child), __receiver_t<_Is>{__op_})...};
        }
      };

      template <class... _Child>
      auto operator()(__ignore, __ignore, _Child&&... __child) const
        noexcept(__nothrow_callable<__impl, __indices_for<_Child...>, _Child...>)
          -> __call_result_t<__impl, __indices_for<_Child...>, _Child...> {
        return __impl{__op_}(__indices_for<_Child...>(), static_cast<_Child&&>(__child)...);
      }

      auto operator()(__ignore, __ignore) const noexcept -> __tuple_for<> {
        return {};
      }
    };
    STDEXEC_PRAGMA_POP()

    template <class _Sexpr, class _Receiver>
    struct __op_state : __op_base<_Sexpr, _Receiver> {
      using __desc_t = typename __decay_t<_Sexpr>::__desc_t;
      using __tag_t = typename __desc_t::__tag;
      using __data_t = typename __desc_t::__data;
      //using __children_t = typename __desc_t::__children;
      using __state_t = typename __op_state::__state_t;
      using __inner_ops_t = __result_of<__sexpr_apply, _Sexpr, __connect_fn<_Sexpr, _Receiver>>;

      __inner_ops_t __inner_ops_;

      // template <std::size_t _Idx>
      // static std::ptrdiff_t __get_child_op_offset() noexcept {
      //   __op_state* __self = (__op_state*) &__self;
      //   return (std::ptrdiff_t)((char*) &__tup::__get<_Idx>(__self->__inner_ops_) - static_cast<char*>(__self));
      // }

      __op_state(_Sexpr&& __sexpr, _Receiver __rcvr) //
        noexcept(
          __nothrow_constructible_from<__op_base<_Sexpr, _Receiver>, _Sexpr&&, _Receiver&&>
          && __nothrow_callable<__sexpr_apply_t, _Sexpr&&, __connect_fn<_Sexpr, _Receiver>>)
        : __op_state::__op_base{static_cast<_Sexpr&&>(__sexpr), static_cast<_Receiver&&>(__rcvr)}
        , __inner_ops_(
            __sexpr_apply(static_cast<_Sexpr&&>(__sexpr), __connect_fn<_Sexpr, _Receiver>{this})) {
      }

      STDEXEC_ATTRIBUTE((always_inline))
      void
        start() & noexcept {
        using __tag_t = typename __op_state::__tag_t;
        auto&& __rcvr = this->__rcvr();
        __inner_ops_.apply(
          [&](auto&... __ops) noexcept {
            __sexpr_impl<__tag_t>::start(this->__state_, __rcvr, __ops...);
          },
          __inner_ops_);
      }

      template <class _Index, class _Tag2, class... _Args>
      STDEXEC_ATTRIBUTE((always_inline))
      void
        __complete(_Index, _Tag2, _Args&&... __args) noexcept {
        using __tag_t = typename __op_state::__tag_t;
        auto&& __rcvr = this->__rcvr();
        using _CompleteFn = __mtypeof<__sexpr_impl<__tag_t>::complete>;
        if constexpr (__callable<_CompleteFn, _Index, __op_state&, _Tag2, _Args...>) {
          __sexpr_impl<__tag_t>::complete(_Index(), *this, _Tag2(), static_cast<_Args&&>(__args)...);
        } else {
          __sexpr_impl<__tag_t>::complete(
            _Index(), this->__state_, __rcvr, _Tag2(), static_cast<_Args&&>(__args)...);
        }
      }

      template <class _Index>
      STDEXEC_ATTRIBUTE((always_inline))
      auto
        __get_env(_Index) const noexcept
        -> __env_type_t<_Index, __tag_t, _Index, _Sexpr, _Receiver> {
        const auto& __rcvr = this->__rcvr();
        return __sexpr_impl<__tag_t>::get_env(_Index(), this->__state_, __rcvr);
      }
    };

    inline constexpr auto __drop_front = //
      []<class _Fn>(_Fn __fn) noexcept {
        return [__fn = std::move(__fn)]<class... _Rest>(auto&&, _Rest&&... __rest) //
               noexcept(__nothrow_callable<const _Fn&, _Rest...>)
                 -> __call_result_t<const _Fn&, _Rest...> {
                 return __fn(static_cast<_Rest&&>(__rest)...);
               };
      };

    template <class _Tag, class... _Captures>
    STDEXEC_ATTRIBUTE((host, device, always_inline))
    constexpr auto
      __captures(_Tag, _Captures&&... __captures2) {
      return [... __captures3 = static_cast<_Captures&&>(__captures2)]<class _Cvref, class _Fun>(
               _Cvref, _Fun&& __fun) mutable                                           //
             noexcept(__nothrow_callable<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>) //
             -> __call_result_t<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>
               requires __callable<_Fun, _Tag, __minvoke<_Cvref, _Captures>...>
      {
        // The use of decltype(__captures3) here instead of _Captures is a workaround for
        // a codegen bug in nvc++.
        return static_cast<_Fun&&>(__fun)(
          _Tag(), const_cast<__minvoke<_Cvref, decltype(__captures3)>&&>(__captures3)...);
      };
    }

    template <class _Tag, class _Data, class... _Child>
    using __captures_t =
      decltype(__detail::__captures(_Tag(), __declval<_Data>(), __declval<_Child>()...));

    template <class, class, class... _Child>
    using __tuple_size_t = char[sizeof...(_Child) + 2];

    template <std::size_t _Idx, class _Descriptor>
    concept __in_range = (_Idx < sizeof(__minvoke<_Descriptor, __q<__tuple_size_t>>));

  } // namespace __detail

  struct __sexpr_defaults {
    static constexpr auto get_attrs = __detail::__get_attrs;
    static constexpr auto get_env = __detail::__get_env;
    static constexpr auto get_state = __detail::__get_state;
    static constexpr auto connect = __detail::__connect;
    static constexpr auto start = __detail::__start;
    static constexpr auto complete = __detail::__complete;
    static constexpr auto get_completion_signatures = __detail::__sigs;
  };

  template <class _Tag>
  struct __sexpr_impl : __sexpr_defaults { };

  using __detail::__enable_receiver_from_this;

  template <class _Tag>
  using __get_attrs_fn =
    __result_of<__detail::__drop_front, __mtypeof<__sexpr_impl<_Tag>::get_attrs>>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __basic_sender
  template <class...>
  struct __basic_sender {
    using __id = __basic_sender;
    using __t = __basic_sender;
  };

  template <auto _DescriptorFn, class = __anon>
  struct __sexpr {
    using sender_concept = sender_t;

    using __id = __sexpr;
    using __t = __sexpr;
    using __desc_t = decltype(_DescriptorFn());
    using __tag_t = typename __desc_t::__tag;
    using __captures_t = __minvoke<__desc_t, __q<__detail::__captures_t>>;

    mutable __captures_t __impl_;

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_ATTRIBUTE((host, device, always_inline))
    explicit __sexpr(_Tag, _Data&& __data, _Child&&... __child)
      : __impl_(__detail::__captures(
        _Tag(),
        static_cast<_Data&&>(__data),
        static_cast<_Child&&>(__child)...)) {
    }

    template <class _Self>
    using __impl = __sexpr_impl<__meval<__msecond, _Self, __tag_t>>;

    template <class _Self = __sexpr>
    STDEXEC_ATTRIBUTE((always_inline))
    auto
      get_env() const noexcept
      -> __result_of<__sexpr_apply, const _Self&, __get_attrs_fn<__tag_t>> {
      return __sexpr_apply(*this, __detail::__drop_front(__impl<_Self>::get_attrs));
    }

    template <__decays_to<__sexpr> _Self, class... _Env>
    STDEXEC_ATTRIBUTE((always_inline))
    static auto
      get_completion_signatures(_Self&&, _Env&&...) noexcept //
      -> __msecond<
        __if_c<__decays_to<_Self, __sexpr>>,
        __result_of<__impl<_Self>::get_completion_signatures, _Self, _Env...>> {
      return {};
    }

    // BUGBUG fix receiver constraint here:
    template <__decays_to<__sexpr> _Self, /*receiver*/ class _Receiver>
    STDEXEC_ATTRIBUTE((always_inline))
    static auto
      connect(_Self&& __self, _Receiver&& __rcvr)                       //
      noexcept(__noexcept_of<__impl<_Self>::connect, _Self, _Receiver>) //
      -> __msecond<
        __if_c<__decays_to<_Self, __sexpr>>,
        __result_of<__impl<_Self>::connect, _Self, _Receiver>> {
      return __impl<_Self>::connect(static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
    }

    template <class _Sender, class _ApplyFn>
    STDEXEC_ATTRIBUTE((always_inline))
    static auto
      apply(_Sender&& __sndr, _ApplyFn&& __fun) //
      noexcept(
        __nothrow_callable<__detail::__impl_of<_Sender>, __copy_cvref_fn<_Sender>, _ApplyFn>) //
      -> __call_result_t<__detail::__impl_of<_Sender>, __copy_cvref_fn<_Sender>, _ApplyFn> {  //
      return static_cast<_Sender&&>(__sndr).__impl_(
        __copy_cvref_fn<_Sender>(), static_cast<_ApplyFn&&>(__fun)); //
    }

    template <std::size_t _Idx, __decays_to_derived_from<__sexpr> _Self>
    STDEXEC_ATTRIBUTE((always_inline))
    friend auto
      get(_Self&& __self) noexcept -> decltype(auto)
      requires __detail::__in_range<_Idx, __desc_t>
    {
      if constexpr (_Idx == 0) {
        return __tag_t();
      } else {
        return __self.__impl_(__copy_cvref_fn<_Self>(), __nth_pack_element<_Idx>);
      }
    }
  };

  template <class _Tag, class _Data, class... _Child>
  STDEXEC_ATTRIBUTE((host, device))
  __sexpr(_Tag, _Data, _Child...) -> __sexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  template <class _Tag, class _Data, class... _Child>
  using __sexpr_t = __sexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __make_sexpr
  namespace __detail {
    template <class _Tag>
    struct __make_sexpr_t {
      template <class _Data = __, class... _Child>
      constexpr auto operator()(_Data __data = {}, _Child... __child) const {
        return __sexpr_t<_Tag, _Data, _Child...>{
          _Tag(), static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)...};
      }
    };
  } // namespace __detail

  template <class _Tag>
  inline constexpr __detail::__make_sexpr_t<_Tag> __make_sexpr{};

  // The __name_of utility defined below is used to pretty-print the type names of
  // senders in compiler diagnostics.
  namespace __detail {
    struct __basic_sender_name {
      template <class _Tag, class _Data, class... _Child>
      using __result = __basic_sender<_Tag, _Data, __name_of<_Child>...>;

      template <class _Sender>
      using __f = __minvoke<typename __decay_t<_Sender>::__desc_t, __q<__result>>;
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

    template <auto _Descriptor>
    extern __basic_sender_name __name_of_v<__sexpr<_Descriptor>>;

    template <__has_id _Sender>
      requires(!same_as<__id<_Sender>, _Sender>)
    extern __id_name __name_of_v<_Sender>;
  } // namespace __detail
} // namespace stdexec

namespace std {
  template <auto _Descriptor>
  struct tuple_size<stdexec::__sexpr<_Descriptor>>
    : integral_constant<
        size_t,
        stdexec::__v<stdexec::__minvoke<stdexec::__result_of<_Descriptor>, stdexec::__msize>>> { };

  template <size_t _Idx, auto _Descriptor>
  struct tuple_element<_Idx, stdexec::__sexpr<_Descriptor>> {
    using type =                            //
      stdexec::__remove_rvalue_reference_t< //
        stdexec::__call_result_t<           //
          stdexec::__detail::__impl_of<stdexec::__sexpr<_Descriptor>>,
          stdexec::__cp,
          stdexec::__nth_pack_element_t<_Idx>>>;
  };
} // namespace std
