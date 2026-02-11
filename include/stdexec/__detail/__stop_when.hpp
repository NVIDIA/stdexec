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

#include "__execution_fwd.hpp"

#include "__atomic.hpp"
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__env.hpp"
#include "__sender_concepts.hpp"
#include "__sender_introspection.hpp"
#include "__stop_token.hpp"

#include <type_traits>
#include <utility>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.stop.when]
  namespace __stop_when_ {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    template <class _Token, class _Receiver>
    struct __state {
      _Token __token_;
      _Receiver __rcvr_;
    };

    template <class _Token, class _Receiver>
    __state(_Token, _Receiver) -> __state<_Token, _Receiver>;

    struct __stop_when_t {
      template <sender _Sender, unstoppable_token _Token>
      constexpr _Sender&& operator()(_Sender&& __sndr, _Token&&) const noexcept {
        return static_cast<_Sender&&>(__sndr);
      }

      template <sender _Sender, stoppable_token _Token>
      constexpr auto operator()(_Sender&& __sndr, _Token&& __token) const noexcept(
        __nothrow_constructible_from<std::remove_cvref_t<_Sender>, _Sender>
        && __nothrow_constructible_from<std::remove_cvref_t<_Token>, _Token>) {
        return __make_sexpr<__stop_when_t>(
          static_cast<_Token&&>(__token), static_cast<_Sender&&>(__sndr));
      }
    };

    struct __stop_when_impl : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() //
        -> __completion_signatures_of_t<__child_of<_Sender>, _Env...> {
        static_assert(sender_expr_for<_Sender, __stop_when_t>);
        return {};
      };

      static constexpr auto get_env = [](__ignore, const auto& __state) noexcept {
        return __env::__join(
          prop(get_stop_token, __state.__token_), STDEXEC::get_env(__state.__rcvr_));
      };

      template <stoppable_token _Token1, stoppable_token _Token2>
      struct __fused_token {
        _Token1 __tkn1_;
        _Token2 __tkn2_;

        friend constexpr bool operator==(const __fused_token&, const __fused_token&) = default;

        [[nodiscard]]
        bool stop_requested() const noexcept {
          return __tkn1_.stop_requested() || __tkn2_.stop_requested();
        }

        [[nodiscard]]
        bool stop_possible() const noexcept {
          return __tkn1_.stop_possible() || __tkn2_.stop_possible();
        }

        template <class _Fn>
        struct callback_type {
          template <class _C>
            requires __std::constructible_from<_Fn, _C>
          explicit callback_type(__fused_token&& __ftkn, _C&& __fn)
            noexcept(__nothrow_constructible_from<_Fn, _C>)
            : __fn_(static_cast<_C&&>(__fn))
            , __cb1_(std::move(__ftkn.__tkn1_), __cb(this))
            , __cb2_(std::move(__ftkn.__tkn2_), __cb(this)) {
          }

          template <class _C>
            requires __std::constructible_from<_Fn, _C>
          explicit callback_type(const __fused_token& __ftkn, _C&& __fn)
            noexcept(__nothrow_constructible_from<_Fn, _C>)
            : __fn_(static_cast<_C&&>(__fn))
            , __cb1_(__ftkn.__tkn1_, __cb(this))
            , __cb2_(__ftkn.__tkn2_, __cb(this)) {
          }

          callback_type(callback_type&&) = delete;

         private:
          struct __cb {
            callback_type* __self;

            void operator()() noexcept {
              (*__self)();
            }
          };

          using __cb1_t = _Token1::template callback_type<__cb>;
          using __cb2_t = _Token2::template callback_type<__cb>;

          void operator()() noexcept {
            if (!__called_.exchange(true, __std::memory_order_relaxed)) {
              __fn_();
            }
          }

          _Fn __fn_;
          __std::atomic<bool> __called_{false};
          __cb1_t __cb1_;
          __cb2_t __cb2_;
        };
      };

      struct __make_token_fn {
        template <class _SenderToken, class _ReceiverToken>
          requires stoppable_token<std::remove_cvref_t<_SenderToken>>
                && stoppable_token<std::remove_cvref_t<_ReceiverToken>>
        [[nodiscard]]
        auto operator()(_SenderToken&& __sndr_token, _ReceiverToken&& __rcvr_token) const noexcept {
          if constexpr (unstoppable_token<std::remove_cvref_t<_ReceiverToken>>) {
            // when the receiver's stop token is unstoppable, the net token is just
            // the sender's captured token
            return __sndr_token;
          } else {
            // when the receiver's stop token is stoppable, the net token must be
            // a fused token that responds to signals from both the sender's captured
            // token and the receiver's token
            return __fused_token<
              std::remove_cvref_t<_SenderToken>,
              std::remove_cvref_t<_ReceiverToken>
            >{static_cast<_SenderToken&&>(__sndr_token),
              static_cast<_ReceiverToken&&>(__rcvr_token)};
          }
        }
      };

      static constexpr auto get_state =
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver __rcvr) noexcept {
          auto& [__tag, __token, __child] = __self;
          auto __new_token = __make_token_fn{}(
            __forward_like<_Self>(__token), get_stop_token(STDEXEC::get_env(__rcvr)));
          return __state{std::move(__new_token), std::move(__rcvr)};
        };
    };
  } // namespace __stop_when_

  using __stop_when_::__stop_when_t;

  /// @brief The stop-when sender adaptor, which fuses an additional stop token
  ///        into its child sender such that the sender responds to stop
  ///        requests from both the given stop token and the receiver's token
  /// @hideinitializer
  inline constexpr __stop_when_t __stop_when{};

  template <>
  struct __sexpr_impl<__stop_when_t> : __stop_when_::__stop_when_impl { };
} // namespace STDEXEC
