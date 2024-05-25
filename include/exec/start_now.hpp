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

#include "async_scope.hpp"

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
    using __ref_t = decltype(__ref(stdexec::__declval<_Ty&>()));

    struct __joiner {
      virtual ~__joiner() {}
      virtual void join() const noexcept {}
    };

    template <class _StgRef>
    struct __receiver {
      using receiver_concept = stdexec::receiver_t;
      using __t = __receiver;
      using __id = __receiver;

      using _Storage = stdexec::__decay_t<stdexec::__call_result_t<_StgRef>>;
      using _Env = typename _Storage::__env_t;

      _StgRef __stgref_;

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        __stgref_().complete();
      }

      template <class _Error>
      void set_error(_Error&& __err) noexcept = delete;

      void set_stopped() noexcept {
        __stgref_().complete();
      }

      // Forward all receiever queries.
      auto get_env() const noexcept -> _Env {
        return __stgref_().__env_;
      }
    };

    inline const __joiner __empty_joiner_{};

    template <class _StgRef, class _Receiver>
    struct __operation : __joiner {
      using __id = __operation;
      using __t = __operation;

      _StgRef __stgref_;
      mutable _Receiver __rcvr_;

      template<class _R>
      __operation(_StgRef __stgref, _R&& __r) : __stgref_(__stgref), __rcvr_((_R&&)__r) {}

      void join() const noexcept override {
        stdexec::set_value(std::move(__rcvr_));
      }

      void start() noexcept {
        const __joiner* expected = &__empty_joiner_;
        if (!__stgref_().__joiner_.compare_exchange_strong(expected, this)) {
          join();
        }
      }
    };

    template <class _StgRef>
    struct __sender {
      using sender_concept = stdexec::sender_t;
      using __id = __sender;
      using __t = __sender;

      using __completions_t = stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

      _StgRef __stgref_;

      using connect_t = stdexec::connect_t;
      template <stdexec::__decays_to<__sender> _Self, stdexec::receiver _Receiver>
        requires stdexec::receiver_of<_Receiver, __completions_t>
      STDEXEC_MEMFN_DECL(
        auto connect)(this _Self&& __self, _Receiver __rcvr) //
        noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          -> __operation<_StgRef, std::remove_cvref_t<_Receiver>> {
        return {static_cast<_Self&&>(__self).__stgref_, static_cast<_Receiver&&>(__rcvr)};
      }

      auto get_completion_signatures(stdexec::__ignore = {}) -> __completions_t {
        return {};
      }
    };

    template <class _EnvId, class _AsyncScopeId, class... _SenderId>
    struct __storage {
      using _Env = stdexec::__t<_EnvId>;
      using _AsyncScope = stdexec::__t<_AsyncScopeId>;
      using __receiver_t = __receiver<__ref_t<__storage>>;
      using __sender_t = __sender<__ref_t<const __storage>>;
      using __env_t = stdexec::__env::__join_t<stdexec::__env::__with<stdexec::inplace_stop_token, stdexec::get_stop_token_t>, _Env>;
      template<class _S>
      using __nested_t = decltype(stdexec::__declval<_AsyncScope&>().nest(stdexec::__declval<_S&&>()));

      mutable std::atomic<const __joiner*> __joiner_;
      mutable std::atomic<int> __pending_;

      STDEXEC_ATTRIBUTE((no_unique_address))
      stdexec::inplace_stop_source __source_;
      STDEXEC_ATTRIBUTE((no_unique_address))
      __env_t __env_;
      STDEXEC_ATTRIBUTE((no_unique_address))
      stdexec::__decayed_tuple<stdexec::connect_result_t<__nested_t<stdexec::__t<_SenderId>>, __receiver_t>...> __op_state_;

      template<class _Sender>
      auto __construct(_Sender&& __sndr) noexcept {
        return [&, this](){return stdexec::connect(static_cast<_Sender&&>(__sndr), __receiver_t{__ref(*this)});};
      }

      __storage(_Env __env, _AsyncScope& __scope, stdexec::__t<_SenderId>... __sndr)
        : __joiner_(&__empty_joiner_)
        , __pending_(0)
        , __source_()
        , __env_(stdexec::__env::__join(stdexec::__env::__with{__source_.get_token(), stdexec::get_stop_token}, std::move(__env)))
        , __op_state_(stdexec::__conv{__construct(__scope.nest(__sndr))}...) {
        __pending_ = sizeof...(_SenderId);
        stdexec::__apply([](auto&... __op_state) noexcept { bool arr[]{(stdexec::start(__op_state), true)...}; (void)arr; }, __op_state_);
      }

      bool request_stop() noexcept {
        return __source_.request_stop();
      }

      stdexec::inplace_stop_token get_token() const noexcept {
        return __source_.get_token();
      }

      [[nodiscard]] auto async_wait() const noexcept -> __sender_t {
        return __sender_t{__ref(*this)};
      }

    private:
      friend struct __receiver<__ref_t<__storage>>;
      void complete() noexcept {
        if (--__pending_ == 0) {
          auto __joiner = __joiner_.exchange(nullptr);
          if (__joiner) {__joiner->join();}
        }
      }
    };

    template <class _Env, class _AsyncScope, class... _Sender>
    using __storage_t = __storage<
      stdexec::__id<std::remove_cvref_t<_Env>>, 
      stdexec::__id<std::remove_cvref_t<_AsyncScope>>, 
      stdexec::__id<std::remove_cvref_t<_Sender>>...>;

    struct start_now_t {
      template <stdexec::queryable _Env, exec::__scope::__async_scope _AsyncScope, stdexec::sender... _Sender>
        requires (!exec::__scope::__async_scope<_Env>) && (!stdexec::sender<_Env>) && (!stdexec::sender<_AsyncScope>)
      __storage_t<_Env, _AsyncScope, _Sender...> operator()(_Env&& __env, _AsyncScope& __scope, _Sender&&... __sndr) const 
        noexcept(
          std::is_nothrow_move_constructible_v<_Env> &&
          (std::is_nothrow_move_constructible_v<_Sender> && ... && true)) {
        static_assert(stdexec::unstoppable_token<stdexec::stop_token_of_t<_Env>>, "start_now() requires that the given environment does not have a stoppable token");
        static_assert((stdexec::__nofail_sender<_Sender> && ... && true), "start_now() requires that the given senders have no set_error(..) completions");
        using __receiver_t = __receiver<__ref_t<__storage_t<_Env, _AsyncScope, _Sender...>>>;
        static_assert((stdexec::sender_to<_Sender, __receiver_t> && ... && true), "The senders passed to start_now do not satisfy the constraints");
        return __storage_t<_Env, _AsyncScope, _Sender...>{
                  static_cast<_Env&&>(__env), std::ref(__scope), static_cast<_Sender&&>(__sndr)...};
      }
      template <exec::__scope::__async_scope _AsyncScope, stdexec::sender... _Sender>
        requires (!stdexec::sender<_AsyncScope>)
      __storage_t<stdexec::__root_env_t, _AsyncScope, _Sender...> operator()(_AsyncScope& __scope, _Sender&&... __sndr) const 
        noexcept(
          (std::is_nothrow_move_constructible_v<_Sender> && ... && true)) {
        static_assert((stdexec::__nofail_sender<_Sender> && ... && true), "start_now() requires that the given senders have no set_error(..) completions");
        using __receiver_t = __receiver<__ref_t<__storage_t<stdexec::__root_env_t, _AsyncScope, _Sender...>>>;
        static_assert((stdexec::sender_to<_Sender, __receiver_t> && ... && true), "The senders passed to start_now do not satisfy the constraints");
        return __storage_t<stdexec::__root_env_t, _AsyncScope, _Sender...>{
                  stdexec::__root_env_t{}, std::ref(__scope), static_cast<_Sender&&>(__sndr)...};
      }
    };
  } // namespace __start_now_

  using __start_now_::start_now_t;
  inline constexpr start_now_t start_now{};
} // namespace exec
