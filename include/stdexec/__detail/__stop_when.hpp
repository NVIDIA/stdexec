/*
 * Copyright (c) 2025 Ian Petersen
 * Copyright (c) 2025 NVIDIA Corporation
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

#  include "__atomic.hpp"
#  include "__basic_sender.hpp"
#  include "__concepts.hpp"
#  include "__env.hpp"
#  include "__sender_concepts.hpp"
#  include "__sender_introspection.hpp"
#  include "__stop_token.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <functional>
#    include <utility>
#  endif

#  include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.stop.when]
  namespace __stop_when_
  {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    template <class _Token, class _Receiver>
    struct __state
    {
      _Token    __token_;
      _Receiver __rcvr_;
    };

    template <class _Token, class _Receiver>
    __state(_Token, _Receiver) -> __state<_Token, _Receiver>;

    struct __stop_when_t
    {
      template <sender _Sender, unstoppable_token _Token>
      constexpr auto operator()(_Sender&& __sndr, _Token&&) const
        noexcept(__nothrow_move_constructible<_Sender>) -> _Sender
      {
        return static_cast<_Sender&&>(__sndr);
      }

      template <sender _Sender, stoppable_token _Token>
      constexpr auto operator()(_Sender&& __sndr, _Token&& __token) const
        noexcept(__nothrow_decay_copyable<_Sender> && __nothrow_decay_copyable<_Token>)
      {
        return __make_sexpr<__stop_when_t>(static_cast<_Token&&>(__token),
                                           static_cast<_Sender&&>(__sndr));
      }
    };

    template <class _Token1, class _Token2>
    struct __fused_token
    {
      friend constexpr bool operator==(__fused_token const &, __fused_token const &) = default;

      [[nodiscard]]
      bool stop_requested() const noexcept
      {
        return __tkn1_.stop_requested() || __tkn2_.stop_requested();
      }

      [[nodiscard]]
      bool stop_possible() const noexcept
      {
        return __tkn1_.stop_possible() || __tkn2_.stop_possible();
      }

      template <class _Fn>
      struct callback_type : private __immovable
      {
        template <__decays_to<__fused_token> _FusedToken, class _Cb>
          requires __std::constructible_from<_Fn, _Cb>
        explicit callback_type(_FusedToken&& __ftkn, _Cb&& __fn)
          noexcept(__nothrow_constructible_from<_Fn, _Cb>)
          : __fn_(static_cast<_Cb&&>(__fn))
          , __cb1_(static_cast<_FusedToken&&>(__ftkn).__tkn1_, __cb_t(*this))
          , __cb2_(static_cast<_FusedToken&&>(__ftkn).__tkn2_, __cb_t(*this))
        {}

        void operator()() noexcept
        {
          if (!__called_.exchange(true, __std::memory_order_relaxed))
          {
            __fn_();
          }
        }

       private:
        using __cb_t  = std::reference_wrapper<callback_type>;
        using __cb1_t = _Token1::template callback_type<__cb_t>;
        using __cb2_t = _Token2::template callback_type<__cb_t>;

        _Fn                 __fn_;
        __cb1_t             __cb1_;
        __cb2_t             __cb2_;
        __std::atomic<bool> __called_{false};
      };

      _Token1 __tkn1_;
      _Token2 __tkn2_;
    };

    struct __fuse_token_fn
    {
      template <stoppable_token _SenderToken, unstoppable_token _ReceiverToken>
      [[nodiscard]]
      constexpr auto
      operator()(_SenderToken __sndr_token, _ReceiverToken __rcvr_token) const noexcept
        -> _SenderToken
      {
        // when the receiver's stop token is unstoppable, the net token is just the
        // sender's captured token
        return __sndr_token;
      }

      template <stoppable_token _SenderToken, stoppable_token _ReceiverToken>
      [[nodiscard]]
      constexpr auto
      operator()(_SenderToken __sndr_token, _ReceiverToken __rcvr_token) const noexcept
        -> __fused_token<_SenderToken, _ReceiverToken>
      {
        // when the receiver's stop token is stoppable, the net token must be a fused
        // token that responds to signals from both the sender's captured token and the
        // receiver's token
        return __fused_token<_SenderToken, _ReceiverToken>{
          static_cast<_SenderToken&&>(__sndr_token),
          static_cast<_ReceiverToken&&>(__rcvr_token)};
      }
    };

    struct __mk_env2_fn
    {
      template <class _FusedToken, class _Env>
      [[nodiscard]]
      constexpr auto operator()(_FusedToken __fused_token, _Env&& __env) const noexcept
        -> __join_env_t<prop<get_stop_token_t, _FusedToken>, _Env>
      {
        return __env::__join(prop(get_stop_token, static_cast<_FusedToken&&>(__fused_token)),
                             static_cast<_Env&&>(__env));
      }
    };

    struct __stop_when_impl : __sexpr_defaults
    {
      static constexpr auto __get_env = [](__ignore, auto const & __state) noexcept
      {
        return __mk_env2_fn()(__state.__token_, STDEXEC::get_env(__state.__rcvr_));
      };

      static constexpr auto __get_state =
        []<class _Sender, class _Receiver>(_Sender&& __self, _Receiver __rcvr) noexcept
      {
        auto& [__tag, __token, __child] = __self;
        auto __new_token = __fuse_token_fn()(STDEXEC::__forward_like<_Sender>(__token),
                                             get_stop_token(STDEXEC::get_env(__rcvr)));
        return __state{std::move(__new_token), std::move(__rcvr)};
      };

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, __stop_when_t>);
        using __token_t = __decay_t<__data_of<_Sender>>;
        return get_completion_signatures<
          __child_of<_Sender>,
          __call_result_t<__mk_env2_fn,
                          __call_result_t<__fuse_token_fn, __token_t, stop_token_of_t<_Env>>,
                          _Env>...>();
      };
    };
  }  // namespace __stop_when_

  using __stop_when_::__stop_when_t;

  /// @brief The stop-when sender adaptor, which fuses an additional stop token
  ///        into its child sender such that the sender responds to stop
  ///        requests from both the given stop token and the receiver's token
  /// @hideinitializer
  STDEXEC_MODULE_EXPORT_AUTHORING
  inline constexpr __stop_when_t __stop_when{};

  template <>
  struct __sexpr_impl<__stop_when_t> : __stop_when_::__stop_when_impl
  {};
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
