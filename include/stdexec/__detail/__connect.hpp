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

// include these after __execution_fwd.hpp
#include "__completion_signatures_of.hpp"
#include "__connect_awaitable.hpp"
#include "__tag_invoke.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.connect]
  namespace __connect
  {
    template <class _Sender, class _Receiver>
    concept __with_static_member = requires(__declfn_t<_Sender&&>   __sndr,
                                            __declfn_t<_Receiver&&> __rcvr) {
      STDEXEC_REMOVE_REFERENCE(_Sender)::__static_connect(__sndr(), __rcvr());
    };

    template <class _Sender, class _Receiver>
    concept __with_member =  //
      requires(__declfn_t<_Sender&&> __sndr, __declfn_t<_Receiver&&> __rcvr) {
        __sndr().connect(__rcvr());
      };

    template <class _Sender, class _Receiver>
    concept __with_co_await = __awaitable<_Sender, __connect_await::__promise<_Sender, _Receiver>>;

    template <class _Sender, class _Receiver>
    concept __with_legacy_tag_invoke = __tag_invocable<connect_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __with_any_connect = __with_static_member<_Sender, _Receiver>
                              || __with_member<_Sender, _Receiver>
                              || __with_co_await<_Sender, _Receiver>
                              || __with_legacy_tag_invoke<_Sender, _Receiver>;

#if !STDEXEC_MSVC()

#  define STDEXEC_CONNECT_DECLFN_FOR(_EXPR) __declfn_t<decltype(_EXPR), noexcept(_EXPR)>

    // A variable template whose type is a function pointer such that the
    // return type and noexcept-ness depend on whether _Sender can be connected
    // to _Receiver via any of the supported customization mechanisms.
    template <class _Sender, class _Receiver, bool _NothrowTransform>
    extern __declfn_t<void> __connect_declfn_v;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>
    extern STDEXEC_CONNECT_DECLFN_FOR(STDEXEC_REMOVE_REFERENCE(_Sender)  //
                                      ::__static_connect(__declval<_Sender>(),
                                                         __declval<_Receiver>()))
      __connect_declfn_v<_Sender, _Receiver, true>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>  //
            || __with_member<_Sender, _Receiver>
    extern STDEXEC_CONNECT_DECLFN_FOR(__declval<_Sender>().connect(__declval<_Receiver>()))
      __connect_declfn_v<_Sender, _Receiver, true>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>  //
            || __with_member<_Sender, _Receiver>         //
            || __with_co_await<_Sender, _Receiver>
    extern __declfn_t<__call_result_t<__connect_awaitable_t, _Sender, _Receiver>, false>
      __connect_declfn_v<_Sender, _Receiver, true>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>  //
            || __with_member<_Sender, _Receiver>         //
            || __with_co_await<_Sender, _Receiver>       //
            || __with_legacy_tag_invoke<_Sender, _Receiver>
    extern STDEXEC_CONNECT_DECLFN_FOR(__tag_invoke(connect,
                                                   __declval<_Sender>(),
                                                   __declval<_Receiver>()))
      __connect_declfn_v<_Sender, _Receiver, true>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>
    extern __declfn_t<decltype(STDEXEC_REMOVE_REFERENCE(_Sender)  //
                               ::__static_connect(__declval<_Sender>(), __declval<_Receiver>())),
                      false>
      __connect_declfn_v<_Sender, _Receiver, false>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>  //
            || __with_member<_Sender, _Receiver>
    extern __declfn_t<decltype(__declval<_Sender>().connect(__declval<_Receiver>())), false>
      __connect_declfn_v<_Sender, _Receiver, false>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>  //
            || __with_member<_Sender, _Receiver>         //
            || __with_co_await<_Sender, _Receiver>
    extern __declfn_t<__call_result_t<__connect_awaitable_t, _Sender, _Receiver>, false>
      __connect_declfn_v<_Sender, _Receiver, false>;

    template <class _Sender, class _Receiver>
      requires __with_static_member<_Sender, _Receiver>  //
            || __with_member<_Sender, _Receiver>         //
            || __with_co_await<_Sender, _Receiver>       //
            || __with_legacy_tag_invoke<_Sender, _Receiver>
    extern __declfn_t<__tag_invoke_result_t<connect_t, _Sender, _Receiver>, false>
      __connect_declfn_v<_Sender, _Receiver, false>;

    template <class _Sender, class _Receiver>
    using __connect_declfn_t =
      decltype(__connect_declfn_v<
               transform_sender_result_t<_Sender, env_of_t<_Receiver>>,
               _Receiver,
               __nothrow_callable<transform_sender_t, _Sender, env_of_t<_Receiver>>>);

#  undef STDEXEC_CONNECT_DECLFN_FOR

#else  // ^^^ !STDEXEC_MSVC() ^^^ / vvv STDEXEC_MSVC() vvv

#  define STDEXEC_CONNECT_DECLFN_FOR(_EXPR) __declfn<decltype(_EXPR), noexcept(_EXPR)>()

    template <bool _NothrowTransform>
    struct __connect_declfn;

    // This version is used for MSVC
    template <>
    struct __connect_declfn<true>
    {
      template <class _Sender, class _Receiver>
      static consteval auto __get() noexcept
      {
        if constexpr (__with_static_member<_Sender, _Receiver>)
        {
          return STDEXEC_CONNECT_DECLFN_FOR(
            STDEXEC_REMOVE_REFERENCE(_Sender)  //
            ::__static_connect(__declval<_Sender>(), __declval<_Receiver>()));
        }
        else if constexpr (__with_member<_Sender, _Receiver>)
        {
          return STDEXEC_CONNECT_DECLFN_FOR(__declval<_Sender>().connect(__declval<_Receiver>()));
        }
        else if constexpr (__with_co_await<_Sender, _Receiver>)
        {
          return __declfn<__call_result_t<__connect_awaitable_t, _Sender, _Receiver>, false>();
        }
        else if constexpr (__with_legacy_tag_invoke<_Sender, _Receiver>)
        {
          return STDEXEC_CONNECT_DECLFN_FOR(
            __tag_invoke(connect, __declval<_Sender>(), __declval<_Receiver>()));
        }
        else
        {
          return __declfn<void, false>();
        }
      }
    };

    template <>
    struct __connect_declfn<false>
    {
      template <class _Sender, class _Receiver>
      static consteval auto __get() noexcept
      {
        if constexpr (__with_static_member<_Sender, _Receiver>)
        {
          return __declfn<decltype(STDEXEC_REMOVE_REFERENCE(_Sender)  //
                                   ::__static_connect(__declval<_Sender>(),
                                                      __declval<_Receiver>())),
                          false>();
        }
        else if constexpr (__with_member<_Sender, _Receiver>)
        {
          return __declfn<decltype(__declval<_Sender>().connect(__declval<_Receiver>())), false>();
        }
        else if constexpr (__with_co_await<_Sender, _Receiver>)
        {
          return __declfn<__call_result_t<__connect_awaitable_t, _Sender, _Receiver>, false>();
        }
        else if constexpr (__with_legacy_tag_invoke<_Sender, _Receiver>)
        {
          return __declfn<__tag_invoke_result_t<connect_t, _Sender, _Receiver>, false>();
        }
        else
        {
          return __declfn<void, false>();
        }
      }
    };

    template <class _Sender, class _Receiver>
    using __connect_declfn_t =
      decltype(__connect_declfn<
               __nothrow_callable<transform_sender_t, _Sender, env_of_t<_Receiver>>>::
                 template __get<transform_sender_result_t<_Sender, env_of_t<_Receiver>>,
                                _Receiver>());

#  undef STDEXEC_CONNECT_DECLFN_FOR

#endif  // STDEXEC_MSVC()
  }  // namespace __connect

  template <class _Sender, class _Receiver>
  concept __connectable_to = requires(__declfn_t<_Sender&&> __sndr, __declfn_t<_Receiver&> __rcvr) {
    { transform_sender(__sndr(), get_env(__rcvr())) } -> __connect::__with_any_connect<_Receiver>;
  };

  /////////////////////////////////////////////////////////////////////////////
  // connect_t

  //! @brief Customization point object that *connects* a sender to a
  //!        receiver, producing an operation state.
  //!
  //! @c connect is the central seam of the sender model. Calling
  //! @c stdexec::connect(sndr, rcvr) does *not* run the sender; it
  //! produces an *operation state* — an opaque, immovable, startable
  //! object that, when subsequently passed to @c stdexec::start, will
  //! eventually deliver exactly one completion signal (@c set_value,
  //! @c set_error, or @c set_stopped) to the receiver.
  //!
  //! Every sender adaptor and consumer ultimately reaches for @c connect.
  //! Most user code does not call it directly — @c sync_wait, @c spawn,
  //! and the various adaptors do — but it is the operation a sender
  //! author *must* support, either by exposing a `.connect(receiver)`
  //! member, by being a coroutine awaitable (so the fallback awaitable
  //! adapter applies), or — historically — via @c tag_invoke (now
  //! deprecated).
  //!
  //! See [exec.connect] in the C++26 working draft for the normative
  //! specification.
  //!
  //! **Lookup.**
  //!
  //! Before dispatching, @c connect transforms the sender via the active
  //! domain's @c transform_sender (passing in @c get_env(rcvr) as the
  //! environment). This is how domain-based customization — e.g. the
  //! GPU scheduler taking over a `then` chain — is implemented.
  //!
  //! @c connect then dispatches by trying, in order:
  //!
  //! 1. A static member: <tt>S::__static_connect(sndr, rcvr)</tt>
  //!    (an stdexec-internal extension point).
  //! 2. A non-static member: <tt>sndr.connect(rcvr)</tt>. This is the
  //!    standard way sender authors customize @c connect in C++26.
  //! 3. The awaitable fallback: if @c sndr is awaitable in stdexec's
  //!    receiver-promise type, an adapter operation state is synthesized.
  //!    This is what makes coroutines work as senders.
  //! 4. <tt>tag_invoke(connect, sndr, rcvr)</tt> — deprecated, retained
  //!    for backwards compatibility.
  //!
  //! **Customization.**
  //!
  //! The standard pattern for sender authors is the `.connect()` member:
  //!
  //! @code{.cpp}
  //! struct my_sender {
  //!   using sender_concept = stdexec::sender_tag;
  //!   using completion_signatures = stdexec::completion_signatures<
  //!     stdexec::set_value_t(int)>;
  //!
  //!   template <stdexec::receiver_of<completion_signatures> R>
  //!   auto connect(R rcvr) && -> my_opstate<R> {
  //!     return my_opstate<R>{std::move(rcvr), ...};
  //!   }
  //! };
  //! @endcode
  //!
  //! The returned object must satisfy @c stdexec::operation_state. In
  //! particular, it must be immovable once constructed (typically by
  //! deleting the move and copy constructors).
  //!
  //! **Concept checks.**
  //!
  //! @c connect(s, r) is only well-formed when both
  //! `sender_in<S, env_of_t<R>>` and `receiver_of<R, completion_signatures_of_t<S, env_of_t<R>>>`
  //! hold. The diagnostics for failures here are intentionally focused —
  //! stdexec emits messages that name the specific completion signal or
  //! environment query the receiver doesn't accept.
  //!
  //! @see stdexec::start             — what you call on the returned operation state
  //! @see stdexec::operation_state   — the concept the result satisfies
  //! @see stdexec::sender_to         — the concept this CPO drives
  //! @see stdexec::set_value         — one of the completions the operation eventually delivers
  //! @see stdexec::transform_sender  — the domain-customization step run before dispatch
  struct connect_t
  {
    //! @brief Connect @c __sndr to @c __rcvr, returning an operation state.
    //!
    //! @tparam _Sender   A type modeling @c stdexec::sender_in for
    //!                   @c env_of_t<_Receiver>.
    //! @tparam _Receiver A type modeling @c stdexec::receiver_of for
    //!                   @c __sndr's completion signatures.
    //!
    //! @param __sndr     The sender describing the asynchronous work.
    //!                   Perfect-forwarded into the operation state.
    //! @param __rcvr     The receiver that will eventually receive a
    //!                   completion. Perfect-forwarded into the operation
    //!                   state.
    //!
    //! @returns An object satisfying @c stdexec::operation_state. Pass it
    //!          to @c stdexec::start to begin the work.
    template <class _Sender,
              class _Receiver,
              class _DeclFn = __connect::__connect_declfn_t<_Sender, _Receiver>>
      requires __connectable_to<_Sender, _Receiver>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
      noexcept(__nothrow_callable<_DeclFn>) -> __call_result_t<_DeclFn>
    {
      decltype(auto) __new_sndr = transform_sender(static_cast<_Sender&&>(__sndr), get_env(__rcvr));
      using __new_sndr_t        = decltype(__new_sndr);

      if constexpr (__connect::__with_static_member<__new_sndr_t, _Receiver>)
      {
        return STDEXEC_REMOVE_REFERENCE(
          __new_sndr_t)::__static_connect(static_cast<__new_sndr_t&&>(__new_sndr),
                                          static_cast<_Receiver&&>(__rcvr));
      }
      else if constexpr (__connect::__with_member<__new_sndr_t, _Receiver>)
      {
        return static_cast<__new_sndr_t&&>(__new_sndr).connect(static_cast<_Receiver&&>(__rcvr));
      }
      else if constexpr (__connect::__with_co_await<__new_sndr_t, _Receiver>)
      {
        return __connect_awaitable(static_cast<__new_sndr_t&&>(__new_sndr),
                                   static_cast<_Receiver&&>(__rcvr));
      }
      else
      {
        return __tag_invoke(*this,
                            static_cast<__new_sndr_t&&>(__new_sndr),
                            static_cast<_Receiver&&>(__rcvr));
      }
    }

    static constexpr auto query(forwarding_query_t) noexcept -> bool
    {
      return false;
    }
  };

  //! @brief The customization point object for connecting a sender to a
  //!        receiver.
  //!
  //! @c connect is an instance of @ref connect_t. See @ref connect_t for
  //! the full description, the lookup order, and customization examples.
  //!
  //! @hideinitializer
  inline constexpr connect_t connect{};

  template <class _Sender, class _Receiver>
  concept __nothrow_connectable = sender_to<_Sender, _Receiver>
                               && __nothrow_callable<connect_t, _Sender, _Receiver>;

}  // namespace STDEXEC

#include "__epilogue.hpp"
