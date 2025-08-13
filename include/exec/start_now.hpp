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

#include "../stdexec/__detail/__execution_fwd.hpp"

#include "../stdexec/__detail/__concepts.hpp"
#include "../stdexec/__detail/__env.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__senders.hpp"
#include "../stdexec/__detail/__meta.hpp"

#include "async_scope.hpp"

#include <atomic>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace exec {
  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __start_now
  namespace __start_now_ {
    inline constexpr auto __mkenv =
      []<class _Env>(_Env&& __env, stdexec::inplace_stop_source& __source) {
        return stdexec::__env::__join(
          stdexec::prop{stdexec::get_stop_token, __source.get_token()}, static_cast<_Env&&>(__env));
      };

    template <class _Env>
    using __env_t = stdexec::__result_of<__mkenv, _Env, stdexec::inplace_stop_source&>;

    struct __joiner : stdexec::__immovable {
      void (*__op_)(void*) noexcept = nullptr;
      void* __ptr_ = nullptr;

      void join() const noexcept {
        if (__op_) {
          __op_(__ptr_);
        }
      }
    };

    inline constexpr __joiner __empty_joiner_{};

    template <class _EnvId>
    struct __storage_base : stdexec::__immovable {
      using _Env = stdexec::__t<_EnvId>;

      mutable std::atomic<std::size_t> __pending_;
      mutable std::atomic<const __joiner*> __joiner_{&__empty_joiner_};
      stdexec::inplace_stop_source __source_{};
      __env_t<_Env> __env_;

      __storage_base(_Env&& __env, std::size_t __pending)
        : __pending_(__pending)
        , __env_(__mkenv(static_cast<_Env&&>(__env), __source_)) {
      }

      void __complete() noexcept {
        if (__pending_.fetch_sub(1) == 1) {
          auto __joiner = __joiner_.exchange(nullptr);
          if (__joiner) {
            __joiner->join();
          }
        }
      }
    };

    template <class _EnvId>
    struct __receiver {
      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;

        __storage_base<_EnvId>* __stg_;

        template <class... _As>
        void set_value(_As&&...) noexcept {
          __stg_->__complete();
        }

        template <class _Error>
        void set_error(_Error&&) noexcept = delete;

        void set_stopped() noexcept {
          __stg_->__complete();
        }

        // Forward all receiever queries.
        auto get_env() const noexcept -> decltype(auto) {
          return (__stg_->__env_);
        }
      };
    };

    template <class _EnvId, class _Receiver>
    struct __operation : __joiner {
      using __id = __operation;
      using __t = __operation;

      const __storage_base<_EnvId>* __stg_;
      _Receiver __rcvr_;

      static void __join(void* __ptr) noexcept {
        auto& __op = *static_cast<__operation*>(__ptr);
        stdexec::set_value(static_cast<_Receiver&&>(__op.__rcvr_));
      }

      __operation(const __storage_base<_EnvId>* __stg, _Receiver&& __rcvr)
        : __joiner{{}, __join, this}
        , __stg_(__stg)
        , __rcvr_(static_cast<_Receiver&&>(__rcvr)) {
      }

      void start() & noexcept {
        const __joiner* expected = &__empty_joiner_;
        if (!__stg_->__joiner_.compare_exchange_strong(expected, this)) {
          join();
        }
      }
    };

    template <class _EnvId>
    struct __sender {
      struct __t {
        using sender_concept = stdexec::sender_t;
        using __id = __sender;

        using __completions_t =
          stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_stopped_t()>;

        const __storage_base<_EnvId>* __stg_;

        using connect_t = stdexec::connect_t;

        template <stdexec::receiver_of<__completions_t> _Receiver>
        auto
          connect(_Receiver __rcvr) const noexcept(stdexec::__nothrow_move_constructible<_Receiver>)
            -> __operation<_EnvId, _Receiver> {
          return {__stg_, static_cast<_Receiver&&>(__rcvr)};
        }

        auto get_completion_signatures(stdexec::__ignore = {}) -> __completions_t {
          return {};
        }
      };
    };

    template <class _EnvId, class _AsyncScopeId, class... _SenderIds>
    struct __storage : private __storage_base<_EnvId> {
     private:
      using _Env = stdexec::__t<_EnvId>;
      using _AsyncScope = stdexec::__t<_AsyncScopeId>;
      using __receiver_t = stdexec::__t<__receiver<_EnvId>>;
      using __sender_t = stdexec::__t<__sender<_EnvId>>;
      template <class _Sender>
      using __nested_t = nest_result_t<_AsyncScope, _Sender>;

      stdexec::__tuple_for<
        stdexec::connect_result_t<__nested_t<stdexec::__cvref_t<_SenderIds>>, __receiver_t>...
      >
        __op_state_;

     public:
      __storage(_Env&& __env, _AsyncScope& __scope, stdexec::__cvref_t<_SenderIds>&&... __sndr)
        : __storage_base<_EnvId>(static_cast<_Env&&>(__env), sizeof...(__sndr))
        , __op_state_{stdexec::connect(
            __scope.nest(static_cast<stdexec::__cvref_t<_SenderIds>&&>(__sndr)),
            __receiver_t{this})...} {
        // Start all of the child operations
        __op_state_.for_each(stdexec::start, __op_state_);
      }

      auto request_stop() noexcept -> bool {
        return this->__source_.request_stop();
      }

      [[nodiscard]]
      auto get_token() const noexcept -> stdexec::inplace_stop_token {
        return this->__source_.get_token();
      }

      [[nodiscard]]
      auto async_wait() const noexcept -> __sender_t {
        return __sender_t{this};
      }
    };

    template <class _Env, class _AsyncScope, class... _Sender>
    using __storage_t =
      __storage<stdexec::__id<_Env>, stdexec::__id<_AsyncScope>, stdexec::__cvref_id<_Sender>...>;

    struct start_now_t {
      template <
        stdexec::queryable _Env,
        exec::__scope::__async_scope _AsyncScope,
        stdexec::sender... _Sender
      >
      auto operator()(_Env __env, _AsyncScope& __scope, _Sender&&... __sndr) const noexcept(
        stdexec::__nothrow_move_constructible<std::unwrap_reference_t<_Env>>
        && (stdexec::__nothrow_move_constructible<_Sender> && ...)) {
        using __local_env_t = stdexec::__as_root_env_t<std::unwrap_reference_t<_Env>>;
        static_assert(
          !stdexec::sender<_Env>,
          "The first argument to start_now() must be either an environment or an async_scope");
        static_assert(
          stdexec::unstoppable_token<stdexec::stop_token_of_t<__local_env_t>>,
          "start_now() requires that the given environment does not have a stoppable token");
        static_assert(
          (stdexec::__nofail_sender<_Sender, __local_env_t> && ...),
          "start_now() requires that the given senders have no set_error(..) completions");
        using __receiver_t = stdexec::__t<__receiver<stdexec::__id<_Env>>>;
        static_assert(
          (stdexec::sender_to<_Sender, __receiver_t> && ...),
          "The senders passed to start_now do not satisfy the constraints");
        return __storage_t<__local_env_t, _AsyncScope, _Sender...>{
          stdexec::__as_root_env(static_cast<std::unwrap_reference_t<_Env>&&>(__env)),
          __scope,
          static_cast<_Sender&&>(__sndr)...};
      }

      template <exec::__scope::__async_scope _AsyncScope, stdexec::sender... _Sender>
      auto operator()(_AsyncScope& __scope, _Sender&&... __sndr) const
        noexcept((stdexec::__nothrow_move_constructible<_Sender> && ...)) {
        static_assert(
          !stdexec::sender<_AsyncScope>,
          "The first argument to start_now() must be either an environment or an async_scope");
        static_assert(
          (stdexec::__nofail_sender<_Sender, stdexec::__root_env> && ...),
          "start_now() requires that the given senders have no set_error(..) completions");
        using __receiver_t = stdexec::__t<__receiver<stdexec::__root_env>>;
        static_assert(
          (stdexec::sender_to<_Sender, __receiver_t> && ...),
          "The senders passed to start_now do not satisfy the constraints");
        return __storage_t<stdexec::__root_env, _AsyncScope, _Sender...>{
          stdexec::__root_env{}, __scope, static_cast<_Sender&&>(__sndr)...};
      }
    };
  } // namespace __start_now_

  using __start_now_::start_now_t;
  inline constexpr start_now_t start_now{};
} // namespace exec

STDEXEC_PRAGMA_POP()
