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
#include "__receivers.hpp"
#include "__sender_introspection.hpp"
#include "__tuple.hpp"
#include "__type_traits.hpp"

#include <cstddef>
#include <new>         // IWYU pragma: keep for placement new
#include <type_traits> // IWYU pragma: keep for is_standard_layout
#include <utility>     // for tuple_size/tuple_element

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

#if 1 //defined(STDEXEC_DEMANGLE_SENDER_NAMES) || STDEXEC_MSVC()
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
  struct __op_state;

  template <class _ReceiverId, class _Sexpr, std::size_t _Idx>
  struct __rcvr;

  namespace __detail {
    template <class _Sexpr, class _Receiver>
    struct __connect_fn;

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

    template <class _Tag, class _Sexpr, class _Receiver>
    using __state_type_t =
      __decay_if_t<__result_of<__sexpr_impl<_Tag>::get_state, _Sexpr, _Receiver>>;

    template <class _Tag, class _Index, class _Sexpr, class _Receiver>
    using __env_type_t = __result_of<
      __sexpr_impl<__meval<__msecond, _Index, _Tag>>::get_env,
      _Index,
      const __state_type_t<__meval<__msecond, _Index, _Tag>, _Sexpr, _Receiver>&
    >;

    template <class _Sexpr, class _Receiver>
    concept __connectable = __tup::__applicable_v<__connect_fn<_Sexpr, _Receiver>, _Sexpr>
                         && __mvalid<__state_type_t, tag_of_t<_Sexpr>, _Sexpr, _Receiver>;

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

    struct __defaults {
      static constexpr auto get_attrs =
        [](__ignore, const auto&... __child) noexcept -> decltype(auto) {
        if constexpr (sizeof...(__child) == 1) {
          return __fwd_env(STDEXEC::get_env(__child...));
        } else {
          return env<>();
        }
      };

      static constexpr auto get_env = []<class _State>(__ignore, const _State& __state) noexcept
        -> decltype(STDEXEC::get_env(__state.__rcvr_)) {
        return STDEXEC::get_env(__state.__rcvr_);
      };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(
          __nothrow_decay_copyable<__data_of<_Sender>>) -> decltype(auto) {
        return __state{
          static_cast<_Receiver&&>(__rcvr), STDEXEC::__get<1>(static_cast<_Sender&&>(__sndr))};
      };

      static constexpr auto connect =
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&& __rcvr) noexcept(
          noexcept(__op_state<_Sender, _Receiver>{
            static_cast<_Sender&&>(__sndr),
            static_cast<_Receiver&&>(__rcvr)})) -> __op_state<_Sender, _Receiver>
        requires __connectable<_Sender, _Receiver>
      {
        return __op_state<_Sender, _Receiver>{
          static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)};
      };

      static constexpr auto submit = [] {
      };

      static constexpr auto start =
        []<class _StartTag = start_t, class... _ChildOps>(__ignore, _ChildOps&... __ops) noexcept {
          (_StartTag()(__ops), ...);
        };

      static constexpr auto complete =
        []<class _Index, class _State, class _SetTag, class... _Args>(
          _Index,
          _State& __state,
          _SetTag,
          _Args&&... __args) noexcept {
          static_assert(_Index::value == 0, "I don't know how to complete this operation.");
          _SetTag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(
          __mnever<tag_of_t<_Sender>>,
          "No customization of get_completion_signatures for this sender tag type.");
      }
    };

    template <class _Sexpr, class _Receiver>
    struct __op_base : __immovable {
      using __tag_t = __decay_t<_Sexpr>::__tag_t;
      using __state_t = __state_type_t<__tag_t, _Sexpr, _Receiver>;

      explicit __op_base(_Sexpr&& __sndr, _Receiver&& __rcvr) noexcept(noexcept(
        __state_t(__sexpr_impl<__tag_t>::get_state(__declval<_Sexpr>(), __declval<_Receiver>()))))
        : __state_(
            __sexpr_impl<__tag_t>::get_state(
              static_cast<_Sexpr&&>(__sndr),
              static_cast<_Receiver&&>(__rcvr))) {
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      __state_t __state_;
    };

    template <class _Sexpr, class _Receiver>
    struct __connect_fn {
      template <std::size_t _Idx>
      using __receiver_t = __t<__rcvr<__id<_Receiver>, _Sexpr, _Idx>>;

      template <std::size_t _Idx>
      using __env_t = __detail::__env_type_t<tag_of_t<_Sexpr>, __msize_t<_Idx>, _Sexpr, _Receiver>;

      struct __impl {
        template <std::size_t... _Is, class... _Child>
        auto operator()(__indices<_Is...>, _Child&&... __child) const
          noexcept((__nothrow_connectable<_Child, __receiver_t<_Is>> && ...))
            -> __tuple<connect_result_t<_Child, __receiver_t<_Is>>...> {
          return __tuple{connect(static_cast<_Child&&>(__child), __receiver_t<_Is>{__op_})...};
        }

        __op_state<_Sexpr, _Receiver>* __op_;
      };

      template <class... _Child>
      auto operator()(__ignore, __ignore, _Child&&... __child) const
        noexcept(__nothrow_callable<__impl, __indices_for<_Child...>, _Child...>)
          -> __call_result_t<__impl, __indices_for<_Child...>, _Child...> {
        return __impl{__op_}(__indices_for<_Child...>(), static_cast<_Child&&>(__child)...);
      }

      auto operator()(__ignore, __ignore) const noexcept -> __tuple<> {
        return {};
      }

      __op_state<_Sexpr, _Receiver>* __op_;
    };

    inline constexpr auto __drop_front = []<class _Fn>(_Fn __fn) noexcept {
      return [__fn = std::move(__fn)]<class... _Rest>(auto&&, _Rest&&... __rest) noexcept(
               __nothrow_callable<const _Fn&, _Rest...>) -> __call_result_t<const _Fn&, _Rest...> {
        return __fn(static_cast<_Rest&&>(__rest)...);
      };
    };

    template <class _Tag, class _Self, class... _Env>
    concept __has_get_completion_signatures = requires {
      __sexpr_impl<_Tag>::template get_completion_signatures<_Self, _Env...>();
    };
  } // namespace __detail

  using __sexpr_defaults = __detail::__defaults;

  template <class _ReceiverId, class _Sexpr, std::size_t _Idx>
  struct __rcvr {
    using _Receiver = STDEXEC::__t<_ReceiverId>;

    struct __t {
      using receiver_concept = receiver_t;
      using __id = __rcvr;

      using __index_t = __msize_t<_Idx>;
      using __parent_op_t = __op_state<_Sexpr, _Receiver>;
      using __tag_t = tag_of_t<_Sexpr>;

      template <class... _Args>
      STDEXEC_ATTRIBUTE(always_inline)
      void set_value(_Args&&... __args) noexcept {
        __op_->__complete(__index_t(), STDEXEC::set_value, static_cast<_Args&&>(__args)...);
      }

      template <class _Error>
      STDEXEC_ATTRIBUTE(always_inline)
      void set_error(_Error&& __err) noexcept {
        __op_->__complete(__index_t(), STDEXEC::set_error, static_cast<_Error&&>(__err));
      }

      STDEXEC_ATTRIBUTE(always_inline) void set_stopped() noexcept {
        __op_->__complete(__index_t(), STDEXEC::set_stopped);
      }

      template <class _Index = __msize_t<_Idx>>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      auto get_env() const noexcept -> __detail::__env_type_t<__tag_t, _Index, _Sexpr, _Receiver> {
        return __op_->__get_env(__index_t());
      }

      // A pointer to the parent operation state, which contains the one created with
      // this receiver.
      __parent_op_t* __op_;
    };
  };

  template <class _Sexpr, class _Receiver>
  struct __op_state : __detail::__op_base<_Sexpr, _Receiver> {
    using __desc_t = __decay_t<_Sexpr>::__desc_t;
    using __tag_t = __desc_t::__tag;
    using __data_t = __desc_t::__data;
    using __state_t = __op_state::__op_base::__state_t;
    using __inner_ops_t = __apply_result_t<__detail::__connect_fn<_Sexpr, _Receiver>, _Sexpr>;

    explicit __op_state(_Sexpr&& __sexpr, _Receiver __rcvr) noexcept(
      __nothrow_constructible_from<__detail::__op_base<_Sexpr, _Receiver>, _Sexpr, _Receiver>
      && __nothrow_applicable<__detail::__connect_fn<_Sexpr, _Receiver>, _Sexpr>)
      : __op_state::__op_base{static_cast<_Sexpr&&>(__sexpr), static_cast<_Receiver&&>(__rcvr)}
      , __inner_ops_(__apply(
          __detail::__connect_fn<_Sexpr, _Receiver>{this},
          static_cast<_Sexpr&&>(__sexpr))) {
    }

    STDEXEC_ATTRIBUTE(always_inline) void start() & noexcept {
      using __tag_t = __op_state::__tag_t;
      STDEXEC::__apply(
        [&](auto&... __ops) noexcept { __sexpr_impl<__tag_t>::start(this->__state_, __ops...); },
        __inner_ops_);
    }

    template <class _Index, class _Tag2, class... _Args>
    STDEXEC_ATTRIBUTE(always_inline)
    void __complete(_Index, _Tag2, _Args&&... __args) noexcept {
      using __tag_t = __op_state::__tag_t;
      __sexpr_impl<__tag_t>::complete(
        _Index(), this->__state_, _Tag2(), static_cast<_Args&&>(__args)...);
    }

    template <class _Index>
    STDEXEC_ATTRIBUTE(always_inline)
    auto __get_env(_Index) const noexcept
      -> __detail::__env_type_t<__tag_t, _Index, _Sexpr, _Receiver> {
      return __sexpr_impl<__tag_t>::get_env(_Index(), this->__state_);
    }

    __inner_ops_t __inner_ops_;
  };

  template <class _Tag>
  struct __sexpr_impl : __detail::__defaults {
    using not_specialized = void;
  };

  template <class _Tag, class _Data, class... _Child>
  using __sexpr_t = __sexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;

  //! A dummy type used only for diagnostic purposes.
  //! See `__sexpr` for the implementation of P2300's _`basic-sender`_.
  template <class _Tag, class _Data, class... _Child>
  struct __basic_sender {
    // See MAINTAINERS.md#class-template-parameters for `__id` and `__t`.
    using __id = __basic_sender;
    using __t = __basic_sender;
    using __mangled = __sexpr_t<_Tag, _Data, __remangle_t<_Child>...>;
  };

  namespace {
    //! A struct template to aid in creating senders.
    //! This struct closely resembles P2300's [_`basic-sender`_](https://eel.is/c++draft/exec#snd.expos-24),
    //! but is not an exact implementation.
    //! Note: The struct named `__basic_sender` is just a dummy type and is also not _`basic-sender`_.
    template <auto _DescriptorFn>
    struct __sexpr : __minvoke<decltype(_DescriptorFn()), __qq<__tuple>> {
      using sender_concept = sender_t;

      // See MAINTAINERS.md#class-template-parameters for `__id` and `__t`.
      using __id = __sexpr;
      using __t = __sexpr;
      using __desc_t = decltype(_DescriptorFn());
      using __tag_t = __desc_t::__tag;

      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      constexpr auto get_env() const noexcept -> decltype(auto) {
        return __apply(__detail::__drop_front(__sexpr_impl<__tag_t>::get_attrs), *this);
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        using namespace __detail;
        if constexpr (__has_get_completion_signatures<__tag_t, _Self, _Env...>) {
          return __sexpr_impl<__tag_t>::template get_completion_signatures<_Self, _Env...>();
        } else if constexpr (__has_get_completion_signatures<__tag_t, _Self>) {
          return __sexpr_impl<__tag_t>::template get_completion_signatures<_Self>();
        } else if constexpr (sizeof...(_Env) == 0) {
          return __dependent_sender<_Self>();
        } else {
          return __throw_compile_time_error(__unrecognized_sender_error_t<_Self, _Env...>());
        }
      }

      // Non-standard extension:
      template <class _Self, receiver _Receiver>
      STDEXEC_ATTRIBUTE(nodiscard, always_inline)
      static constexpr auto static_connect(_Self&& __self, _Receiver __rcvr)
        noexcept(__noexcept_of<__sexpr_impl<__tag_t>::connect, _Self, _Receiver>)
          -> __result_of<__sexpr_impl<__tag_t>::connect, _Self, _Receiver> {
        static_assert(__decays_to_derived_from<_Self, __sexpr>);
        return __sexpr_impl<__tag_t>::connect(
          static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
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
      static constexpr auto submit(_Self&& __self, _Receiver&& __rcvr)
        noexcept(__noexcept_of<__sexpr_impl<__tag_t>::submit, _Self, _Receiver>)
          -> __result_of<__sexpr_impl<__tag_t>::submit, _Self, _Receiver> {
        static_assert(__decays_to_derived_from<_Self, __sexpr>);
        return __sexpr_impl<__tag_t>::submit(
          static_cast<_Self&&>(__self), static_cast<_Receiver&&>(__rcvr));
      }
    };

    template <class _Tag, class _Data, class... _Child>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __sexpr(_Tag, _Data, _Child...) -> __sexpr<STDEXEC_SEXPR_DESCRIPTOR(_Tag, _Data, _Child...)>;
  } // anonymous namespace

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
    struct __basic_sender_name {
      template <class _Tag, class _Data, class... _Child>
      using __result = __basic_sender<_Tag, _Data, __demangle_t<_Child>...>;

      template <class _Sender>
      using __f = __minvoke<typename __decay_t<_Sender>::__desc_t, __q<__result>>;
    };

    struct __id_name {
      template <class _Sender>
      using __f = __demangle_t<__id<_Sender>>;
    };

    template <class _Sender>
    extern __mcompose<__cplr, __demangle_fn<_Sender>> __demangle_v<_Sender&>;

    template <class _Sender>
    extern __mcompose<__cprr, __demangle_fn<_Sender>> __demangle_v<_Sender&&>;

    template <class _Sender>
    extern __mcompose<__cpclr, __demangle_fn<_Sender>> __demangle_v<const _Sender&>;

    template <auto _Descriptor>
    extern __basic_sender_name __demangle_v<__sexpr<_Descriptor>>;

    template <__has_id _Sender>
      requires __not_same_as<__id<_Sender>, _Sender>
    extern __id_name __demangle_v<_Sender>;
  } // namespace __detail
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
