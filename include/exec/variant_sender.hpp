/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../stdexec/__detail/__variant.hpp"
#include "../stdexec/execution.hpp"

namespace exec {
  namespace __var {
    using namespace STDEXEC;

    template <class _Receiver, class... _CvSenders>
    struct __operation_state {
      STDEXEC::__variant<connect_result_t<_CvSenders, _Receiver>...> __variant_{STDEXEC::__no_init};

     public:
      template <class _CvSender>
      constexpr explicit __operation_state(_CvSender&& __sndr, _Receiver&& __rcvr)
        noexcept(__nothrow_connectable<_CvSender, _Receiver>) {
        __variant_.__emplace_from(
          STDEXEC::connect, static_cast<_CvSender&&>(__sndr), static_cast<_Receiver&&>(__rcvr));
      }

      constexpr void start() & noexcept {
        STDEXEC::__visit(STDEXEC::start, __variant_);
      }
    };

    template <class _OpState>
    struct __visitor {
      template <class _Receiver, class _Sender>
      constexpr auto operator()(_Receiver&& __rcvr, _Sender&& __sndr) -> _OpState {
        return _OpState{static_cast<_Sender&&>(__sndr), static_cast<_Receiver&&>(__rcvr)};
      }
    };
  } // namespace __var

  template <class... _Senders>
  struct variant_sender {
    template <class _Self, class _Env>
    using __completions_t = STDEXEC::__mtry_q<STDEXEC::__concat_completion_signatures_t>::__f<
      STDEXEC::__completion_signatures_of_t<STDEXEC::__copy_cvref_t<_Self, _Senders>, _Env>...
    >;

    template <std::size_t _Index>
    using __nth_t = STDEXEC::__m_at_c<_Index, _Senders...>;

    template <class _Self, class _Receiver>
    using __opstate_t =
      __var::__operation_state<_Receiver, STDEXEC::__copy_cvref_t<_Self, _Senders>...>;

    STDEXEC::__variant<_Senders...> __sndrs_{STDEXEC::__no_init};

   public:
    using sender_concept = STDEXEC::sender_t;

    constexpr variant_sender()
      requires std::default_initializable<__nth_t<0>>
    {
      __sndrs_.template emplace<0>();
    }

    template <STDEXEC::__not_decays_to<variant_sender> _Sender>
      requires STDEXEC::__decay_copyable<_Sender>
            && STDEXEC::__one_of<STDEXEC::__decay_t<_Sender>, _Senders...>
    /*implicit*/ variant_sender(_Sender&& __sndr)
      noexcept(STDEXEC::__nothrow_decay_copyable<_Sender>) {
      __sndrs_.template emplace<STDEXEC::__decay_t<_Sender>>(static_cast<_Sender&&>(__sndr));
    }

    [[nodiscard]]
    auto index() const noexcept -> std::size_t {
      return __sndrs_.index();
    }

    template <STDEXEC::__not_decays_to<variant_sender> _Sender>
      requires STDEXEC::__decay_copyable<_Sender>
            && STDEXEC::__one_of<STDEXEC::__decay_t<_Sender>, _Senders...>
    auto operator=(_Sender&& __sndr) noexcept(STDEXEC::__nothrow_decay_copyable<_Sender>)
      -> variant_sender& {
      __sndrs_.template emplace<STDEXEC::__decay_t<_Sender>>(static_cast<_Sender&&>(__sndr));
      return *this;
    }

    template <STDEXEC::__one_of<_Senders...> _Sender, class... _Args>
    auto emplace(_Args&&... __args)
      noexcept(STDEXEC::__nothrow_constructible_from<_Sender, _Args...>) -> _Sender& {
      return __sndrs_.template emplace<_Sender>(static_cast<_Args&&>(__args)...);
    }

    template <std::size_t _Index, class... _Args>
    auto emplace(_Args&&... __args)
      noexcept(STDEXEC::__nothrow_constructible_from<__nth_t<_Index>, _Args...>)
        -> __nth_t<_Index>& {
      return __sndrs_.template emplace<_Index>(static_cast<_Args&&>(__args)...);
    }

    void swap(variant_sender& __other) noexcept {
      static_assert(noexcept(__sndrs_.swap(__other.__sndrs_)));
      __sndrs_.swap(__other.__sndrs_);
    }

    template <STDEXEC::__decays_to<variant_sender> _Self, STDEXEC::receiver _Receiver>
      requires(STDEXEC::sender_to<STDEXEC::__copy_cvref_t<_Self, _Senders>, _Receiver> && ...)
    STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr) noexcept(
      (STDEXEC::__nothrow_connectable<STDEXEC::__copy_cvref_t<_Self, _Senders>, _Receiver> && ...))
      -> __opstate_t<_Self, _Receiver> {
      using __visitor_t = __var::__visitor<__opstate_t<_Self, _Receiver>>;
      return STDEXEC::__visit(
        __visitor_t{}, static_cast<_Self&&>(__self).__sndrs_, static_cast<_Receiver&&>(__rcvr));
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    template <STDEXEC::__decays_to<variant_sender> _Self, class _Env>
    static consteval auto get_completion_signatures() -> __completions_t<_Self, _Env> {
      return {};
    }
  };
} // namespace exec
