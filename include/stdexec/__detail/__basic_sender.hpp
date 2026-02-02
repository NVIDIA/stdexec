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

#include "__completion_signatures_of.hpp"
#include "__concepts.hpp"
#include "__connect.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__operation_states.hpp"
#include "__receivers.hpp"
#include "__sender_introspection.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

#include <cstddef>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // Generic __sender type

#if STDEXEC_EDG()
#  define STDEXEC_SEXPR_DESCRIPTOR_FN(_Descriptor)                                                 \
    ([]<class _Desc = _Descriptor>(_Desc __desc = {}) { return __desc; })
#  define STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child)                                            \
    STDEXEC::__descriptor_fn<_Tag, _Data, _Child>()
#else // ^^^ EDG ^^^ / vvv !EDG vvv
#  define STDEXEC_SEXPR_DESCRIPTOR_FN(_Descriptor) ([] { return _Descriptor(); })
#  define STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child)                                            \
    STDEXEC::__descriptor_fn_v<STDEXEC::__detail::__desc<_Tag, _Data, _Child>>
#endif

#if defined(STDEXEC_DEMANGLE_SENDER_NAMES)
  template <class _Descriptor>
  inline constexpr auto __descriptor_fn_v = _Descriptor{};
#else
  template <class _Descriptor, auto _DescriptorFn = STDEXEC_SEXPR_DESCRIPTOR_FN(_Descriptor)>
  inline constexpr auto __descriptor_fn_v = _DescriptorFn;
#endif

  template <class _Tag, class _Data, class... _Child>
  consteval auto __descriptor_fn() noexcept {
    return __descriptor_fn_v<__detail::__desc<_Tag, _Data, _Child...>>;
  }

  template <class _Tag>
  struct __sexpr_impl;

  template <class _Sexpr, class _Receiver>
  struct __opstate;

  template <class _Tag, class _State, std::size_t _Idx>
  struct __rcvr;

  namespace __detail {
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
      __decay_if_t<__result_of<__sexpr_impl<tag_of_t<_Sexpr>>::get_state, _Sexpr, _Receiver>>;

    template <class _Tag, class _Index, class _State>
    using __env_type_t = __result_of<__sexpr_impl<_Tag>::get_env, _Index, const _State&>;

    template <class _Sexpr>
    using __child_indices_t = __desc_of<_Sexpr>::__indices;

    template <class _Receiver, class _Data>
    struct __state {
      using __receiver_t = _Receiver;
      using __data_t = _Data;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Data __data_;
    };

    template <class _Receiver, class _Data>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __state(_Receiver, _Data) -> __state<_Receiver, _Data>;

    template <class _Tag, class _Indices>
    struct __connect;

    template <class _Sexpr>
    using __connect_t = __connect<tag_of_t<_Sexpr>, __child_indices_t<_Sexpr>>;

    template <class _Tag, std::size_t... _Idx>
    struct __connect<_Tag, __indices<_Idx...>> {
      template <class _State, class... _Child>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
      constexpr auto operator()(_State& __state, __ignore, __ignore, _Child&&... __child) const
        noexcept((__nothrow_connectable<_Child, __rcvr<_Tag, _State, _Idx>> && ...))
          -> __tuple<connect_result_t<_Child, __rcvr<_Tag, _State, _Idx>>...> {
        return __tuple{
          STDEXEC::connect(static_cast<_Child&&>(__child), __rcvr<_Tag, _State, _Idx>{__state})...};
      }
    };

    template <class _Sexpr, class _Receiver>
    concept __connectable_to =
      __applicable<__connect_t<_Sexpr>, _Sexpr, __state_type_t<_Sexpr, _Receiver>&>;

    struct __defaults {
      static constexpr auto get_attrs = //
        [](__ignore, __ignore, const auto&... __child) noexcept -> decltype(auto) {
        if constexpr (sizeof...(__child) == 1) {
          return __fwd_env(STDEXEC::get_env(__child...));
        } else {
          return env<>();
        }
      };

      static constexpr auto get_state = //
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(
          __nothrow_decay_copyable<__data_of<_Sender>>) -> decltype(auto) {
        return __state{
          static_cast<_Receiver&&>(__rcvr), STDEXEC::__get<1>(static_cast<_Sender&&>(__sndr))};
      };

      static constexpr auto get_env = //
        []<class _State>(__ignore, const _State& __state) noexcept
        -> env_of_t<decltype(_State::__rcvr_)> {
        return STDEXEC::get_env(__state.__rcvr_);
      };

      static constexpr auto connect = //
        []<class _Receiver, __connectable_to<_Receiver> _Sender>(
          _Sender&& __sndr,
          _Receiver&& __rcvr) //
        noexcept(__nothrow_constructible_from<__opstate<_Sender, _Receiver>, _Sender, _Receiver>)
        -> __opstate<_Sender, _Receiver> {
        return __opstate<_Sender, _Receiver>(
          static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
      };

      static constexpr auto submit = //
        [] {
        };

      static constexpr auto start = //
        []<class... _ChildOps>(__ignore, _ChildOps&... __ops) noexcept {
          static_assert(sizeof...(_ChildOps) > 0);
          (STDEXEC::start(__ops), ...);
        };

      static constexpr auto complete = //
        []<class _Idx, class _State, class _Set, class... _As>(
          _Idx,
          _State& __state,
          _Set,
          _As&&... __as) noexcept -> void {
        static_assert(_Idx::value == 0, "I don't know how to complete this operation.");
        _Set()(static_cast<_State&&>(__state).__rcvr_, static_cast<_As&&>(__as)...);
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(
          __mnever<tag_of_t<_Sender>>,
          "No customization of get_completion_signatures for this sender tag type.");
      }
    };

    template <class _Tag, class _Self, class... _Env>
    concept __has_get_completion_signatures_impl = requires {
      __sexpr_impl<_Tag>::template get_completion_signatures<_Self, _Env...>();
    };
  } // namespace __detail

  using __sexpr_defaults = __detail::__defaults;

  template <class _Tag, class _State, std::size_t _Idx>
  struct __rcvr {
    using receiver_concept = receiver_t;
    using __index_t = __msize_t<_Idx>;

    template <class... _Args>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_value(_Args&&... __args) noexcept {
      static_assert(
        __noexcept_of<__sexpr_impl<_Tag>::complete, __index_t, _State&, set_value_t, _Args...>);
      __sexpr_impl<_Tag>::complete(
        __index_t(), __state_, STDEXEC::set_value, static_cast<_Args&&>(__args)...);
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_error(_Error&& __err) noexcept {
      static_assert(
        __noexcept_of<__sexpr_impl<_Tag>::complete, __index_t, _State&, set_error_t, _Error>);
      __sexpr_impl<_Tag>::complete(
        __index_t(), __state_, STDEXEC::set_error, static_cast<_Error&&>(__err));
    }

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void set_stopped() noexcept {
      static_assert(__noexcept_of<__sexpr_impl<_Tag>::complete, __index_t, _State&, set_stopped_t>);
      __sexpr_impl<_Tag>::complete(__index_t(), __state_, STDEXEC::set_stopped);
    }

    STDEXEC_ATTRIBUTE(nodiscard, always_inline)
    constexpr auto get_env() const noexcept -> __detail::__env_type_t<_Tag, __index_t, _State> {
      static_assert(__noexcept_of<__sexpr_impl<_Tag>::get_env, __index_t, const _State&>);
      return __sexpr_impl<_Tag>::get_env(__index_t(), const_cast<const _State&>(__state_));
    }

    _State& __state_;
  };

  template <class _Sexpr, class _Receiver>
  struct __opstate {
    using __desc_t = __decay_t<_Sexpr>::__desc_t;
    using __tag_t = __desc_t::__tag;
    using __state_t = __detail::__state_type_t<_Sexpr, _Receiver>;
    using __connect_t = __detail::__connect<__tag_t, __detail::__child_indices_t<_Sexpr>>;
    using __child_ops_t = __apply_result_t<__connect_t, _Sexpr, __state_t&>;

    constexpr explicit __opstate(_Sexpr&& __sndr, _Receiver __rcvr) noexcept(
      noexcept(
        __state_t(__sexpr_impl<__tag_t>::get_state(__declval<_Sexpr>(), __declval<_Receiver>())))
      && __nothrow_applicable<__connect_t, _Sexpr, __state_t&>)
      : __state_(
          __sexpr_impl<__tag_t>::get_state(
            static_cast<_Sexpr&&>(__sndr),
            static_cast<_Receiver&&>(__rcvr)))
      , __child_ops_(__apply(__connect_t{}, static_cast<_Sexpr&&>(__sndr), __state_)) {
    }

    STDEXEC_IMMOVABLE(__opstate);

    STDEXEC_ATTRIBUTE(always_inline)
    constexpr void start() noexcept {
      STDEXEC::__apply(__sexpr_impl<__tag_t>::start, __child_ops_, __state_);
    }

    __state_t __state_;
    __child_ops_t __child_ops_;
  };

  template <class _Tag>
  struct __sexpr_impl : __sexpr_defaults { };

  template <class _Tag, class _Data, class... _Child>
  using __sexpr_t = __sexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  //! A dummy type used only for diagnostic purposes.
  //! See `__sexpr` for the implementation of P2300's _`basic-sender`_.
  template <class _Tag, class _Data, class... _Child>
  struct __basic_sender {
    using __mangled_t = __sexpr_t<_Tag, _Data, __remangle_t<_Child>...>;
  };

#if !defined(STDEXEC_DEMANGLE_SENDER_NAMES)
  namespace {
#endif
    //! A struct template to aid in creating senders. This struct resembles P2300's
    //! [_`basic-sender`_](https://eel.is/c++draft/exec#snd.expos-24), but is not an exact
    //! implementation. Note: The struct named `__basic_sender` is just a dummy type and
    //! is also not _`basic-sender`_.
    template <auto _DescriptorFn>
    struct __sexpr : __minvoke<decltype(_DescriptorFn()), __qq<__tuple>> {
      using sender_concept = sender_t;

      using __desc_t = decltype(_DescriptorFn());
      using __tag_t = __desc_t::__tag;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr auto get_env() const noexcept -> decltype(auto) {
        return __apply(__sexpr_impl<__tag_t>::get_attrs, __c_upcast<__sexpr>(*this));
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        using namespace __detail;
        static_assert(STDEXEC_IS_BASE_OF(__sexpr, __decay_t<_Self>));
        using __self_t = __copy_cvref_t<_Self, __sexpr>;
        if constexpr (__has_get_completion_signatures_impl<__tag_t, __self_t, _Env...>) {
          return __sexpr_impl<__tag_t>::template get_completion_signatures<__self_t, _Env...>();
        } else if constexpr (__has_get_completion_signatures_impl<__tag_t, __self_t>) {
          return __sexpr_impl<__tag_t>::template get_completion_signatures<__self_t>();
        } else if constexpr (sizeof...(_Env) == 0) {
          return __dependent_sender<_Self>();
        } else {
          return __throw_compile_time_error(__unrecognized_sender_error_t<_Self, _Env...>());
        }
      }

      // Non-standard extension:
      template <class _Self, receiver _Receiver>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      static constexpr auto static_connect(_Self&& __self, _Receiver __rcvr) noexcept(
        __noexcept_of<__sexpr_impl<__tag_t>::connect, __copy_cvref_t<_Self, __sexpr>, _Receiver>)
        -> __result_of<__sexpr_impl<__tag_t>::connect, __copy_cvref_t<_Self, __sexpr>, _Receiver> {
        static_assert(STDEXEC_IS_BASE_OF(__sexpr, __decay_t<_Self>));
        return __sexpr_impl<__tag_t>::connect(
          STDEXEC::__c_upcast<__sexpr>(static_cast<_Self&&>(__self)),
          static_cast<_Receiver&&>(__rcvr));
      }

      template <receiver _Receiver>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr auto connect(_Receiver __rcvr) && noexcept(
        __noexcept_of<__sexpr_impl<__tag_t>::connect, __sexpr, _Receiver>)
        -> __result_of<__sexpr_impl<__tag_t>::connect, __sexpr, _Receiver> {
        return __sexpr_impl<__tag_t>::connect(
          static_cast<__sexpr&&>(*this), static_cast<_Receiver&&>(__rcvr));
      }

      template <receiver _Receiver>
        requires __std::copy_constructible<__sexpr>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr auto connect(_Receiver __rcvr) const & noexcept(
        __noexcept_of<__sexpr_impl<__tag_t>::connect, __sexpr const &, _Receiver>)
        -> __result_of<__sexpr_impl<__tag_t>::connect, __sexpr const &, _Receiver> {
        return __sexpr_impl<__tag_t>::connect(*this, static_cast<_Receiver&&>(__rcvr));
      }

      // Non-standard extension:
      template <class _Self, receiver _Receiver>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      static constexpr auto submit(_Self&& __self, _Receiver&& __rcvr) noexcept(
        __noexcept_of<__sexpr_impl<__tag_t>::submit, __copy_cvref_t<_Self, __sexpr>, _Receiver>)
        -> __result_of<__sexpr_impl<__tag_t>::submit, __copy_cvref_t<_Self, __sexpr>, _Receiver> {
        return __sexpr_impl<__tag_t>::submit(
          STDEXEC::__c_upcast<__sexpr>(static_cast<_Self&&>(__self)),
          static_cast<_Receiver&&>(__rcvr));
      }
    };

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __sexpr(_Tag, _Data, _Child...) -> __sexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;
#if !defined(STDEXEC_DEMANGLE_SENDER_NAMES)
  } // anonymous namespace
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __make_sexpr
  //! A tagged function-object
  //! Takes data and children and
  //! returns `__sexpr_t<_Tag, _Data, _Child...>{_Tag(), data, children...}`.
  namespace __detail {
    template <class _Tag>
    struct __make_sexpr_t {
      template <class _Data = __, class... _Child>
      constexpr auto operator()(_Data __data = {}, _Child... __child) const {
        return __sexpr_t<_Tag, _Data, _Child...>{
          {_Tag(), static_cast<_Data&&>(__data), static_cast<_Child&&>(__child)...}
        };
      }
    };
  } // namespace __detail

  template <class _Tag>
  inline constexpr __detail::__make_sexpr_t<_Tag> __make_sexpr{};

  // The __demangle_t utility defined below is used to pretty-print the type names of
  // senders in compiler diagnostics.
  namespace __detail {
    template <class _Tag, class _Data, class... _Child>
    using __basic_sender_t = __basic_sender<_Tag, _Data, __demangle_t<_Child>...>;

    template <auto _Descriptor>
    extern __declfn_t<__minvoke<__result_of<_Descriptor>, __q<__basic_sender_t>>>
      __demangle_v<__sexpr<_Descriptor>>;
  } // namespace __detail
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
