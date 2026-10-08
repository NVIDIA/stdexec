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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "__completion_signatures_of.hpp"
#  include "__concepts.hpp"
#  include "__connect.hpp"
#  include "__diagnostics.hpp"
#  include "__env.hpp"
#  include "__memory.hpp"
#  include "__meta.hpp"
#  include "__operation_states.hpp"
#  include "__receivers.hpp"
#  include "__sender_introspection.hpp"
#  include "__tuple.hpp"
#  include "__type_traits.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <cstddef>
#  endif

#  include "__prologue.hpp"

STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // Generic __sender type

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Tag>
  struct __sexpr_impl;

  template <class _Sexpr, class _Receiver>
  struct __opstate;

  template <class _Tag, class _State, std::size_t _Idx>
  struct __rcvr;

  namespace __detail
  {
    // A decay_copyable trait that uses C++17 guaranteed copy elision, so
    // that __decay_copyable_if<immovable_type> is satisfied.
    template <class _Ty, class _Uy = __decay_t<_Ty>>
    concept __decay_copyable_if = requires(__declfn_t<_Ty> __val) { _Uy(__val()); };

    template <class _Ty, class _Uy = __decay_t<_Ty>>
    concept __nothrow_decay_copyable_if = requires(__declfn_t<_Ty> __val) {
      { _Uy(__val()) } noexcept;
    };

    template <__decay_copyable_if _Ty>
    using __decay_if_t = __decay_t<_Ty>;

    template <class _Sexpr, class _Receiver>
    using __state_type_t =
      __decay_if_t<__result_of<__sexpr_impl<tag_of_t<_Sexpr>>::__get_state, _Sexpr, _Receiver>>;

    template <class _Tag, class _Index, class _State>
    using __env_type_t = __result_of<__sexpr_impl<_Tag>::__get_env, _Index, _State const &>;

    template <class _Sexpr>
    using __child_indices_t = __decay_t<_Sexpr>::__indices_t;

    template <class _Receiver, class _Data>
    struct __state
    {
      using __receiver_t = _Receiver;
      using __data_t     = _Data;

      template <class _CvData>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr __state(_Receiver __rcvr, _CvData&& __data)
        noexcept(__nothrow_decay_copyable<_CvData>)
        : __rcvr_(static_cast<_Receiver&&>(__rcvr))
        , __data_(STDEXEC::__allocator_aware_forward(static_cast<_CvData&&>(__data), __rcvr_))
      {}

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Data     __data_;
    };

    template <class _Receiver, class _Data>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __state(_Receiver, _Data) -> __state<_Receiver, _Data>;

    template <class _Tag, class _Indices>
    struct __connect;

    template <class _Sexpr>
    using __connect_t = __connect<tag_of_t<_Sexpr>, __child_indices_t<_Sexpr>>;

    template <class _Tag, std::size_t... _Idx>
    struct __connect<_Tag, __indices<_Idx...>>
    {
      template <class _State, class... _Child>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_State& __state, __ignore, __ignore, _Child&&... __child) const
        noexcept((__nothrow_connectable<_Child, __rcvr<_Tag, _State, _Idx>> && ...))
          -> __tuple<connect_result_t<_Child, __rcvr<_Tag, _State, _Idx>>...>
      {
        return __tuple{
          STDEXEC::connect(static_cast<_Child&&>(__child), __rcvr<_Tag, _State, _Idx>{__state})...};
      }
    };

    template <class _Sexpr, class _Receiver>
    concept __connectable_to =
      __applicable<__connect_t<_Sexpr>, _Sexpr, __state_type_t<_Sexpr, _Receiver>&>;

    STDEXEC_MODULE_EXPORT_AUTHORING
    struct __defaults
    {
      static constexpr auto __get_attrs =  //
        [](__ignore, __ignore, auto const &... __child) noexcept -> decltype(auto)
      {
        if constexpr (sizeof...(__child) == 1)
        {
          return __fwd_env(STDEXEC::get_env(__child...));
        }
        else
        {
          return env<>();
        }
      };

      static constexpr auto __get_state =  //
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(
          __nothrow_decay_copyable<__data_of<_Sender>>)
        -> __state<std::remove_cvref_t<_Receiver>, std::remove_cvref_t<__data_of<_Sender>>>
      {
        return __state{static_cast<_Receiver&&>(__rcvr),
                       STDEXEC::__get<1>(static_cast<_Sender&&>(__sndr))};
      };

      static constexpr auto __get_env =                              //
        []<class _State>(__ignore, _State const & __state) noexcept  //
        -> env_of_t<decltype(_State::__rcvr_)>
      {
        return STDEXEC::get_env(__state.__rcvr_);
      };

      static constexpr auto __connect =  //
        []<class _Receiver, __connectable_to<_Receiver> _Sender>(_Sender&&   __sndr,
                                                                 _Receiver&& __rcvr) noexcept(  //
          __nothrow_constructible_from<__opstate<_Sender, _Receiver>, _Sender, _Receiver>)
        -> __opstate<_Sender, _Receiver>
      {
        return __opstate<_Sender, _Receiver>(static_cast<_Sender&&>(__sndr),
                                             static_cast<_Receiver&&>(__rcvr));
      };

      static constexpr auto __submit = //
        [] {
        };

      static constexpr auto __start =  //
        []<class... _ChildOps>(__ignore, _ChildOps&... __ops) noexcept
      {
        static_assert(sizeof...(_ChildOps) > 0);
        (STDEXEC::start(__ops), ...);
      };

      static constexpr auto __complete =  //
        []<class _Idx, class _State, class _Set, class... _As>(_Idx,
                                                               _State& __state,
                                                               _Set,
                                                               _As&&... __as) noexcept -> void
      {
        static_assert(_Idx::value == 0, "I don't know how to complete this operation.");
        _Set()(static_cast<_State&&>(__state).__rcvr_, static_cast<_As&&>(__as)...);
      };

      template <class _Sender, class _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__mnever<tag_of_t<_Sender>>,
                      "No customization of get_completion_signatures for this sender tag type.");
      }
    };

    template <class _Tag, class _Self, class... _Env>
    inline constexpr bool __has_get_completion_signatures_v = requires {
      __sexpr_impl<_Tag>::template __get_completion_signatures<_Self>();
    };

    template <class _Tag, class _Self, class _Env>
    inline constexpr bool __has_get_completion_signatures_v<_Tag, _Self, _Env> = requires {
      __sexpr_impl<_Tag>::template __get_completion_signatures<_Self, _Env>();
    };
  }  // namespace __detail

  STDEXEC_MODULE_EXPORT_AUTHORING
  using __sexpr_defaults = __detail::__defaults;

  template <class _Tag, class _State, std::size_t _Idx>
  struct __rcvr
  {
    using receiver_concept = receiver_tag;
    using __index_t        = __msize_t<_Idx>;

#  if STDEXEC_APPLE_CLANG()
    // These constructors are a work-around for bad codegen with apple-clang
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr explicit __rcvr(_State& __state) noexcept
      : __state_(__state)
    {}

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr __rcvr(__rcvr const & __other) noexcept
      : __state_(__other.__state_)
    {}
#  endif  // STDEXEC_APPLE_CLANG()

    template <class... _Args>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_value(_Args&&... __args) noexcept
    {
      static_assert(
        __noexcept_of<__sexpr_impl<_Tag>::__complete, __index_t, _State&, set_value_t, _Args...>);
      __sexpr_impl<_Tag>::__complete(__index_t(),
                                     __state_,
                                     STDEXEC::set_value,
                                     static_cast<_Args&&>(__args)...);
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_error(_Error&& __err) noexcept
    {
      static_assert(
        __noexcept_of<__sexpr_impl<_Tag>::__complete, __index_t, _State&, set_error_t, _Error>);
      __sexpr_impl<_Tag>::__complete(__index_t(),
                                     __state_,
                                     STDEXEC::set_error,
                                     static_cast<_Error&&>(__err));
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_stopped() noexcept
    {
      static_assert(
        __noexcept_of<__sexpr_impl<_Tag>::__complete, __index_t, _State&, set_stopped_t>);
      __sexpr_impl<_Tag>::__complete(__index_t(), __state_, STDEXEC::set_stopped);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto get_env() const noexcept -> __detail::__env_type_t<_Tag, __index_t, _State>
    {
      static_assert(__noexcept_of<__sexpr_impl<_Tag>::__get_env, __index_t, _State const &>);
      return __sexpr_impl<_Tag>::__get_env(__index_t(), const_cast<_State const &>(__state_));
    }

    _State& __state_;
  };

  template <class _Sexpr, class _Receiver>
  struct __opstate
  {
    using __tag_t       = __decay_t<_Sexpr>::__tag_t;
    using __state_t     = __detail::__state_type_t<_Sexpr, _Receiver>;
    using __connect_t   = __detail::__connect<__tag_t, __detail::__child_indices_t<_Sexpr>>;
    using __child_ops_t = __apply_result_t<__connect_t, _Sexpr, __state_t&>;

    constexpr explicit __opstate(_Sexpr&& __sndr, _Receiver __rcvr)
      noexcept(noexcept(__state_t(__sexpr_impl<__tag_t>::__get_state(__declval<_Sexpr>(),
                                                                     __declval<_Receiver>())))
               && __nothrow_applicable<__connect_t, _Sexpr, __state_t&>)
      : __state_(__sexpr_impl<__tag_t>::__get_state(static_cast<_Sexpr&&>(__sndr),
                                                    static_cast<_Receiver&&>(__rcvr)))
      , __child_ops_(__apply(__connect_t{}, static_cast<_Sexpr&&>(__sndr), __state_))
    {}

    STDEXEC_IMMOVABLE(__opstate);

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void start() noexcept
    {
      static_assert(
        noexcept(STDEXEC::__apply(__sexpr_impl<__tag_t>::__start, __child_ops_, __state_)));
      STDEXEC::__apply(__sexpr_impl<__tag_t>::__start, __child_ops_, __state_);
    }

    __state_t     __state_;
    __child_ops_t __child_ops_;
  };

  template <class _Tag>
  struct __sexpr_impl : __sexpr_defaults
  {};

  //! A struct template to aid in creating senders. This struct resembles
  //! P2300's [_`basic-sender`_](https://eel.is/c++draft/exec#snd.expos-24),
  //! but is not an exact implementation.
  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Tag, class _Data, class... _Child>
  struct __sexpr : __tuple<_Tag, _Data, _Child...>
  {
    using sender_concept = sender_tag;

    using __tag_t       = _Tag;
    using __data_t      = _Data;
    using __children_t  = __mlist<_Child...>;
    using __indices_t   = __make_indices<sizeof...(_Child)>;
    using __base_t      = __tuple<_Tag, _Data, _Child...>;
    using __get_attrs_t = __mtypeof<__sexpr_impl<__tag_t>::__get_attrs>;
    using __attrs_t     = __apply_result_t<__get_attrs_t, __base_t const &>;

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto get_env() const noexcept -> __attrs_t
    {
      return __apply(__sexpr_impl<__tag_t>::__get_attrs, __c_upcast<__base_t>(*this));
    }

    template <class _Self, class... _Env>
    static consteval auto get_completion_signatures()
    {
      using namespace __detail;
      static_assert(STDEXEC_IS_BASE_OF(__sexpr, __decay_t<_Self>));
      using __self_t = __copy_cvref_t<_Self, __sexpr>;
      if constexpr (__has_get_completion_signatures_v<__tag_t, __self_t, _Env...>)
      {
        return __sexpr_impl<__tag_t>::template __get_completion_signatures<__self_t, _Env...>();
      }
      else if constexpr (__has_get_completion_signatures_v<__tag_t, __self_t>)
      {
        return __sexpr_impl<__tag_t>::template __get_completion_signatures<__self_t>();
      }
      else if constexpr (sizeof...(_Env) == 0)
      {
        return __throw_dependent_sender_error<_Self>();
      }
      else
      {
        return STDEXEC::__throw_compile_time_error(__unrecognized_sender_error_t<_Self, _Env...>());
      }
    }

    // Non-standard extension:
    template <class _Self, receiver _Receiver>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    static constexpr auto __static_connect(_Self&& __self, _Receiver __rcvr) noexcept(
      __noexcept_of<__sexpr_impl<__tag_t>::__connect, __copy_cvref_t<_Self, __sexpr>, _Receiver>)
      -> __result_of<__sexpr_impl<__tag_t>::__connect, __copy_cvref_t<_Self, __sexpr>, _Receiver>
    {
      static_assert(STDEXEC_IS_BASE_OF(__sexpr, __decay_t<_Self>));
      return __sexpr_impl<__tag_t>::__connect(STDEXEC::__c_upcast<__sexpr>(
                                                static_cast<_Self&&>(__self)),
                                              static_cast<_Receiver&&>(__rcvr));
    }

    template <receiver _Receiver>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto connect(_Receiver __rcvr) && noexcept(
      __noexcept_of<__sexpr_impl<__tag_t>::__connect, __sexpr, _Receiver>)
      -> __result_of<__sexpr_impl<__tag_t>::__connect, __sexpr, _Receiver>
    {
      return __sexpr_impl<__tag_t>::__connect(static_cast<__sexpr&&>(*this),
                                              static_cast<_Receiver&&>(__rcvr));
    }

    template <receiver _Receiver>
      requires __std::copy_constructible<__sexpr>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto connect(_Receiver __rcvr) const & noexcept(
      __noexcept_of<__sexpr_impl<__tag_t>::__connect, __sexpr const &, _Receiver>)
      -> __result_of<__sexpr_impl<__tag_t>::__connect, __sexpr const &, _Receiver>
    {
      return __sexpr_impl<__tag_t>::__connect(*this, static_cast<_Receiver&&>(__rcvr));
    }

    // Non-standard extension:
    template <class _Self, receiver _Receiver>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    static constexpr auto submit(_Self&& __self, _Receiver&& __rcvr) noexcept(
      __noexcept_of<__sexpr_impl<__tag_t>::__submit, __copy_cvref_t<_Self, __sexpr>, _Receiver>)
      -> __result_of<__sexpr_impl<__tag_t>::__submit, __copy_cvref_t<_Self, __sexpr>, _Receiver>
    {
      return __sexpr_impl<__tag_t>::__submit(STDEXEC::__c_upcast<__sexpr>(
                                               static_cast<_Self&&>(__self)),
                                             static_cast<_Receiver&&>(__rcvr));
    }
  };

  template <class _Tag, class _Data, class... _Child>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  __sexpr(_Tag, _Data, _Child...) -> __sexpr<_Tag, _Data, _Child...>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
