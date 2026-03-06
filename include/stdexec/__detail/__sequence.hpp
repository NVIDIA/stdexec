/*
 * Copyright (c) 2024 NVIDIA Corporation
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

#include "__concepts.hpp"
#include "__connect.hpp"
#include "__receivers.hpp"
#include "__transform_completion_signatures.hpp"
#include "__tuple.hpp"
#include "__variant.hpp"

#include <type_traits>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC
{
  namespace __seq
  {
    template <class... _Senders>
    struct __sndr;
  }  // namespace __seq

  struct __sequence_t
  {
    template <class _Sender>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto operator()(_Sender __sndr) const
      noexcept(STDEXEC::__nothrow_move_constructible<_Sender>) -> _Sender;

    template <class... _Senders>
      requires(sizeof...(_Senders) > 1)
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto operator()(_Senders... sndrs) const
      noexcept(STDEXEC::__nothrow_move_constructible<_Senders...>) -> __seq::__sndr<_Senders...>;
  };

  namespace __seq
  {
    template <class _Receiver>
    struct __opstate_base
    {
      template <class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr void __set_value([[maybe_unused]] Args&&... args) noexcept
      {
        STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr_), static_cast<Args&&>(args)...);
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void __start_next() noexcept
      {
        (*__start_next_)(this);
      }

      _Receiver __rcvr_;
      void (*__start_next_)(__opstate_base*) noexcept = nullptr;
    };

    template <class _Receiver>
    struct __rcvr_base
    {
      using receiver_concept = STDEXEC::receiver_t;

      template <class _Error>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr void set_error(_Error&& err) && noexcept
      {
        STDEXEC::set_error(static_cast<_Receiver&&>(__opstate_->__rcvr_),
                           static_cast<_Error&&>(err));
      }

      STDEXEC_ATTRIBUTE(host, device) void set_stopped() && noexcept
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__opstate_->__rcvr_));
      }

      // TODO: use the predecessor's completion scheduler as the current scheduler here.
      STDEXEC_ATTRIBUTE(nodiscard, host, device)
      constexpr auto get_env() const noexcept -> STDEXEC::env_of_t<_Receiver>
      {
        return STDEXEC::get_env(__opstate_->__rcvr_);
      }

      __opstate_base<_Receiver>* __opstate_;
    };

    template <class _Receiver, bool _IsLast>
    struct __rcvr : __rcvr_base<_Receiver>
    {
      using receiver_concept = STDEXEC::receiver_t;

      template <class... _Args>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr void set_value(_Args&&... __args) && noexcept
      {
        if constexpr (_IsLast)
        {
          this->__opstate_->__set_value(static_cast<_Args&&>(__args)...);
        }
        else
        {
          this->__opstate_->__start_next();
        }
      }
    };

    template <class _Tuple>
    struct __convert_tuple_fn
    {
      template <class... _Ts>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr _Tuple operator()(_Ts&&... __ts) const
        noexcept(STDEXEC::__nothrow_constructible_from<_Tuple, _Ts...>)
      {
        return _Tuple{static_cast<_Ts&&>(__ts)...};
      }
    };

    template <class _Receiver, class... _Senders>
    struct __opstate;

    template <class _Receiver, class _CvSender0, class... _Senders>
    struct __opstate<_Receiver, _CvSender0, _Senders...> : __opstate_base<_Receiver>
    {
      using operation_state_concept = STDEXEC::operation_state_t;

      // We will be connecting the first sender in the opstate constructor, so we don't need to
      // store it in the opstate. The use of `STDEXEC::__ignore` causes the first sender to not
      // be stored.
      using __senders_t = STDEXEC::__tuple<STDEXEC::__ignore, _Senders...>;

      template <bool IsLast>
      using __rcvr_t = __seq::__rcvr<_Receiver, IsLast>;

      template <class _Sender, class IsLast>
      using __child_opstate_t = STDEXEC::connect_result_t<_Sender, __rcvr_t<IsLast::value>>;

      using __mk_child_ops_variant_fn =
        STDEXEC::__mzip_with2<STDEXEC::__q2<__child_opstate_t>, STDEXEC::__qq<STDEXEC::__variant>>;

      using __is_last_mask_t =
        STDEXEC::__mfill_c<sizeof...(_Senders),
                           STDEXEC::__mfalse,
                           STDEXEC::__mbind_back_q<STDEXEC::__mlist, STDEXEC::__mtrue>>;

      using __ops_variant_t = STDEXEC::__minvoke<__mk_child_ops_variant_fn,
                                                 STDEXEC::__tuple<_CvSender0, _Senders...>,
                                                 __is_last_mask_t>;

      template <class _CvSenders>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr explicit __opstate(_Receiver&& __rcvr, _CvSenders&& __sndrs)
        noexcept(::STDEXEC::__nothrow_applicable<__convert_tuple_fn<__senders_t>, _CvSenders>
                 && ::STDEXEC::__nothrow_connectable<::STDEXEC::__tuple_element_t<0, _CvSenders>,
                                                     __rcvr_t<sizeof...(_Senders) == 0>>)
        : __opstate_base<_Receiver>{static_cast<_Receiver&&>(__rcvr)}
        // move all but the first sender into the opstate:
        , __sndrs_{
            STDEXEC::__apply(__convert_tuple_fn<__senders_t>{}, static_cast<_CvSenders&&>(__sndrs))}
      {
        // Below, it looks like we are using `__sndrs` after it has been moved from. This
        // is not the case. `__sndrs` is moved into a tuple type that has `__ignore` for
        // the first element. The result is that the first sender in `__sndrs` is not
        // moved from, but the rest are.
        __ops_.template __emplace_from<0>(STDEXEC::connect,
                                          STDEXEC::__get<0>(static_cast<_CvSenders&&>(__sndrs)),
                                          __rcvr_t<sizeof...(_Senders) == 0>{this});
      }

      template <std::size_t _Remaining>
      static constexpr void __start_next(__opstate_base<_Receiver>* __self_) noexcept
      {
        constexpr auto __nth  = sizeof...(_Senders) - _Remaining;
        auto*          __self = static_cast<__opstate*>(__self_);
        auto&          __sndr = STDEXEC::__get<__nth + 1>(__self->__sndrs_);
        constexpr bool __is_nothrow =
          STDEXEC::__nothrow_connectable<STDEXEC::__m_at_c<__nth, _Senders...>,
                                         __rcvr_t<_Remaining == 1>>;
        STDEXEC_TRY
        {
          auto& __op =
            __self->__ops_.template __emplace_from<__nth + 1>(STDEXEC::connect,
                                                              std::move(__sndr),
                                                              __rcvr_t<(_Remaining == 1)>{__self});
          if constexpr (_Remaining > 1)
          {
            __self->__start_next_ = &__start_next<_Remaining - 1>;
          }
          STDEXEC::start(__op);
        }
        STDEXEC_CATCH_ALL
        {
          if constexpr (__is_nothrow)
          {
            STDEXEC::__std::unreachable();
          }
          else
          {
            STDEXEC::set_error(static_cast<_Receiver&&>(static_cast<__opstate*>(__self_)->__rcvr_),
                               std::current_exception());
          }
        }
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void start() noexcept
      {
        if (sizeof...(_Senders) != 0)
        {
          this->__start_next_ = &__start_next<sizeof...(_Senders)>;
        }
        STDEXEC::start(STDEXEC::__var::__get<0>(__ops_));
      }

      __senders_t     __sndrs_;
      __ops_variant_t __ops_{STDEXEC::__no_init};
    };

    template <class _Sender>
    concept __has_eptr_completion =
      STDEXEC::sender_in<_Sender>
      && __transform_completion_signatures(STDEXEC::get_completion_signatures<_Sender>(),
                                           __ignore_completion(),
                                           __decay_arguments<STDEXEC::set_error_t>(),
                                           __ignore_completion())
           .__contains(STDEXEC::__fn_ptr_t<STDEXEC::set_error_t, std::exception_ptr>());

    template <class _Sender0, class... _Senders>
    struct __sndr<_Sender0, _Senders...>
    {
      using sender_concept = STDEXEC::sender_t;
      using __senders_t    = STDEXEC::__tuple<_Sender0, _Senders...>;

      // Even without an Env, we can sometimes still determine the completion signatures
      // of the sequence sender. If any of the child senders has a
      // set_error(exception_ptr) completion, then the sequence sender has a
      // set_error(exception_ptr) completion. We don't have to ask if any connect call
      // throws.
      template <class _Self, class... _Env>
        requires(sizeof...(_Env) > 0)
             || __has_eptr_completion<STDEXEC::__copy_cvref_t<_Self, _Sender0>>
             || (__has_eptr_completion<_Senders> || ...)
      STDEXEC_ATTRIBUTE(host, device)
      static consteval auto get_completion_signatures()
      {
        if constexpr (!STDEXEC::__decay_copyable<_Self>)
        {
          return STDEXEC::__throw_compile_time_error<
            STDEXEC::_SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
            STDEXEC::_WITH_PRETTY_SENDER_<__sndr<_Sender0, _Senders...>>>();
        }
        else
        {
          using __env_t               = STDEXEC::__mfront<_Env..., STDEXEC::env<>>;
          using __rcvr_t              = STDEXEC::__receiver_archetype<__env_t>;
          constexpr bool __is_nothrow = (STDEXEC::__nothrow_connectable<_Senders, __rcvr_t> && ...);

          // The completions of the sequence sender are the error and stopped completions of all the
          // child senders plus the value completions of the last child sender.
          return __concat_completion_signatures(
            __transform_completion_signatures(
              STDEXEC::get_completion_signatures<STDEXEC::__copy_cvref_t<_Self, _Sender0>,
                                                 _Env...>(),
              __ignore_completion()),
            __transform_completion_signatures(
              STDEXEC::get_completion_signatures<_Senders, _Env...>(),
              __ignore_completion())...,
            STDEXEC::get_completion_signatures<STDEXEC::__mback<_Senders...>, _Env...>(),
            STDEXEC::__eptr_completion_unless_t<__mbool<__is_nothrow>>());
        }
      }

      template <STDEXEC::__decay_copyable _Self, class _Receiver>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        noexcept(STDEXEC::__nothrow_constructible_from<
                 __opstate<_Receiver, STDEXEC::__copy_cvref_t<_Self, _Sender0>, _Senders...>,
                 _Receiver,
                 __copy_cvref_t<_Self, __senders_t>>)
      {
        return __opstate<_Receiver, STDEXEC::__copy_cvref_t<_Self, _Sender0>, _Senders...>{
          static_cast<_Receiver&&>(__rcvr),
          static_cast<_Self&&>(__self).__sndrs_};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      template <std::size_t _Index, class _Self>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto&& __static_get(_Self&& __self) noexcept
      {
        if constexpr (_Index == 0)
        {
          return static_cast<_Self&&>(__self).__tag_;
        }
        else if constexpr (_Index == 1)
        {
          return static_cast<_Self&&>(__self).__ign_;
        }
        else
        {
          return STDEXEC::__get<_Index - 2>(static_cast<_Self&&>(__self).__sndrs_);
        }
      }

      template <std::size_t _Index>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto&& get() && noexcept
      {
        return __static_get<_Index>(static_cast<__sndr&&>(*this));
      }

      template <std::size_t _Index>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto&& get() & noexcept
      {
        return __static_get<_Index>(*this);
      }

      template <std::size_t _Index>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto&& get() const & noexcept
      {
        return __static_get<_Index>(*this);
      }

      STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused) __sequence_t __tag_;
      STDEXEC_ATTRIBUTE(no_unique_address, maybe_unused) STDEXEC::__ __ign_;
      __senders_t __sndrs_;
    };
  }  // namespace __seq

  template <class _Sender>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto __sequence_t::operator()(_Sender __sndr) const
    noexcept(STDEXEC::__nothrow_move_constructible<_Sender>) -> _Sender
  {
    return __sndr;
  }

  template <class... _Senders>
    requires(sizeof...(_Senders) > 1)
  STDEXEC_ATTRIBUTE(host, device)
  constexpr auto __sequence_t::operator()(_Senders... sndrs) const
    noexcept(STDEXEC::__nothrow_move_constructible<_Senders...>) -> __seq::__sndr<_Senders...>
  {
    return __seq::__sndr<_Senders...>{{}, {}, {static_cast<_Senders&&>(sndrs)...}};
  }

  inline constexpr __sequence_t __sequence{};
}  // namespace STDEXEC

namespace exec = experimental::execution;

namespace std
{
  template <class... _Senders>
  struct tuple_size<STDEXEC::__seq::__sndr<_Senders...>>
    : std::integral_constant<std::size_t, sizeof...(_Senders) + 2>
  {};

  template <size_t I, class... _Senders>
  struct tuple_element<I, STDEXEC::__seq::__sndr<_Senders...>>
  {
    using type = STDEXEC::__m_at_c<I, STDEXEC::__sequence_t, STDEXEC::__, _Senders...>;
  };
}  // namespace std

STDEXEC_PRAGMA_POP()
