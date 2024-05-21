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

#include "stdexec/__detail/__execution_fwd.hpp"

#include "stdexec/__detail/__concepts.hpp"
#include "stdexec/__detail/__env.hpp"
#include "stdexec/__detail/__receivers.hpp"
#include "stdexec/__detail/__senders.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__type_traits.hpp"

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __start_now
  namespace __start_now_ {
    namespace {
      inline constexpr auto __ref = []<class _Ty>(_Ty& __ty) noexcept {
        return [__ty = &__ty]() noexcept -> decltype(auto) {
          return (*__ty);
        };
      };
    } // namespace

    template <class _Ty>
    using __ref_t = decltype(__ref(__declval<_Ty&>()));

    struct __joiner {
      virtual ~__joiner() {}
      virtual void join() const noexcept {}
    };

    template <class _StgRef>
    struct __receiver {
      using receiver_concept = receiver_t;
      using __t = __receiver;
      using __id = __receiver;

      using _Storage = __decay_t<__call_result_t<_StgRef>>;
      using _Env = typename _Storage::__env_t;

      _StgRef __stgref_;

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        auto __joiner = __stgref_().__joiner_.exchange(nullptr);
        if (__joiner) {__joiner->join();}
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept = delete;

      void set_stopped() noexcept {
        auto __joiner = __stgref_().__joiner_.exchange(nullptr);
        if (__joiner) {__joiner->join();}
      }

      // Forward all receiever queries.
      auto get_env() const noexcept -> _Env {
        return __stgref_().__env_;
      }
    };

    static inline const __joiner __empty_joiner_{};

    template <class _StgRef, class _Receiver>
    struct __operation : __joiner {
      using __id = __operation;
      using __t = __operation;

      using _Storage = __decay_t<__call_result_t<_StgRef>>;

      _StgRef __stgref_;
      mutable _Receiver __rcvr_;

      template<class _R>
      __operation(_StgRef __stgref, _R&& __r) : __stgref_(__stgref), __rcvr_((_R&&)__r) {}

      void join() const noexcept override {
        set_value(std::move(__rcvr_));
      }

      template <__decays_to<__operation> _Self>
      STDEXEC_MEMFN_DECL(
        auto start)(this _Self& __self) noexcept //
          -> void {
        const __joiner* expected = &__empty_joiner_;
        if (!__self.__stgref_().__joiner_.compare_exchange_strong(expected, &__self)) {
          __self.join();
        }
      }
    };

    template <class _StgRef>
    struct __sender {
      using sender_concept = sender_t;
      using __id = __sender;
      using __t = __sender;

      using _Storage = __decay_t<__call_result_t<_StgRef>>;

      template <class _Env>
      using __completions_t = completion_signatures<set_value_t(), set_stopped_t()>;

      _StgRef __stgref_;

      template <__decays_to<__sender> _Self, class _Receiver>
        requires receiver_of<_Receiver, __completions_t<env_of_t<_Receiver>>>
      STDEXEC_MEMFN_DECL(
        auto connect)(this _Self&& __self, _Receiver __rcvr) //
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> __operation<_StgRef, std::remove_cvref_t<_Receiver>> {
        return {static_cast<_Self&&>(__self).__stgref_, static_cast<_Receiver&&>(__rcvr)};
      }

      template <class _Env>
      auto get_completion_signatures(_Env&&) -> __completions_t<_Env> {
        return {};
      }
    };

    template <class _SenderId, class _EnvId>
    struct __storage {
      using _Sender = stdexec::__t<_SenderId>;
      using _Env = stdexec::__t<_EnvId>;
      using __receiver_t = __receiver<__ref_t<__storage>>;
      using __sender_t = __sender<__ref_t<const __storage>>;
      using __env_t = __env::__join_t<_Env, __env::__with<inplace_stop_token, get_stop_token_t>>;

      static_assert(sender_to<_Sender, __receiver_t>, "The sender passed to start_now does not satisfy the constraints");

      mutable std::atomic<const __joiner*> __joiner_{&__empty_joiner_};

      STDEXEC_ATTRIBUTE((no_unique_address))
      inplace_stop_source source;
      STDEXEC_ATTRIBUTE((no_unique_address))
      __env_t __env_;
      STDEXEC_ATTRIBUTE((no_unique_address))
      connect_result_t<_Sender, __receiver_t> __op_state_;

      __storage(_Sender&& __sndr, _Env __env)
        : __env_(__env::__join(std::move(__env), __env::__with{source.get_token(), get_stop_token}))
        , __op_state_(connect(static_cast<_Sender&&>(__sndr), __receiver_t{__ref(*this)})) {
        start(__op_state_);
      }

      bool request_stop() noexcept {
        return source.request_stop();
      }

      inplace_stop_token get_token() const noexcept {
        return source.get_token();
      }

      auto join() const noexcept -> __sender_t {
        return __sender_t{__ref(*this)};
      }
    };

    struct start_now_t {
      template <sender _Sender, class _Env = __root_env_t>
      __storage<__id<_Sender>, __id<_Env>> operator()(_Sender&& __sndr, _Env&& __env = {}) const noexcept(false) {
        return __storage<__id<_Sender>, __id<_Env>>{
                  static_cast<_Sender&&>(__sndr), static_cast<_Env&&>(__env)};
      }
    };
  } // namespace __start_now_

  using __start_now_::start_now_t;
  inline constexpr start_now_t start_now{};
} // namespace exec
