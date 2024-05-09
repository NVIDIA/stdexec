/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__detail/__as_awaitable.hpp"
#include "__detail/__basic_sender.hpp"
#include "__detail/__bulk.hpp"
#include "__detail/__completion_signatures.hpp"
#include "__detail/__connect_awaitable.hpp"
#include "__detail/__continue_on.hpp"
#include "__detail/__cpo.hpp"
#include "__detail/__debug.hpp"
#include "__detail/__domain.hpp"
#include "__detail/__ensure_started.hpp"
#include "__detail/__env.hpp"
#include "__detail/__execute.hpp"
#include "__detail/__inline_scheduler.hpp"
#include "__detail/__into_variant.hpp"
#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__intrusive_slist.hpp"
#include "__detail/__just.hpp"
#include "__detail/__let.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__on.hpp"
#include "__detail/__operation_states.hpp"
#include "__detail/__read_env.hpp"
#include "__detail/__receivers.hpp"
#include "__detail/__receiver_adaptor.hpp"
#include "__detail/__run_loop.hpp"
#include "__detail/__schedule_from.hpp"
#include "__detail/__schedulers.hpp"
#include "__detail/__senders.hpp"
#include "__detail/__sender_adaptor_closure.hpp"
#include "__detail/__split.hpp"
#include "__detail/__start_detached.hpp"
#include "__detail/__start_on.hpp"
#include "__detail/__stopped_as_error.hpp"
#include "__detail/__stopped_as_optional.hpp"
#include "__detail/__submit.hpp"
#include "__detail/__then.hpp"
#include "__detail/__transfer_just.hpp"
#include "__detail/__transform_sender.hpp"
#include "__detail/__transform_completion_signatures.hpp"
#include "__detail/__type_traits.hpp"
#include "__detail/__upon_error.hpp"
#include "__detail/__upon_stopped.hpp"
#include "__detail/__utility.hpp"
#include "__detail/__when_all.hpp"
#include "__detail/__with_awaitable_senders.hpp"
#include "__detail/__write_env.hpp"

#include "functional.hpp"
#include "concepts.hpp"
#include "coroutine.hpp"
#include "stop_token.hpp"

#include <atomic>
#include <cassert>
#include <concepts>
#include <stdexcept>
#include <memory>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <variant>
#include <cstddef>
#include <exception>
#include <utility>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wundefined-inline")
STDEXEC_PRAGMA_IGNORE_GNU("-Wsubobject-linkage")
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

STDEXEC_PRAGMA_IGNORE_EDG(1302)
STDEXEC_PRAGMA_IGNORE_EDG(497)
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Tag = set_value_t>
  concept __completes_on =
    __decays_to<__call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to<__call_result_t<get_scheduler_t, _Env>, _Scheduler>;

  // NOT TO SPEC
  template <class _Tag, const auto& _Predicate>
  concept tag_category = //
    requires {
      typename __mbool<bool{_Predicate(_Tag{})}>;
      requires bool { _Predicate(_Tag{}) };
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    inline auto __make_env(run_loop& __loop) noexcept {
      return __env::__with(__loop.get_scheduler(), get_scheduler, get_delegatee_scheduler);
    }

    struct __env : __result_of<__make_env, run_loop&> {
      __env();

      explicit __env(run_loop& __loop) noexcept
        : __result_of<__make_env, run_loop&>{__sync_wait::__make_env(__loop)} {
      }
    };

    // What should sync_wait(just_stopped()) return?
    template <class _Sender, class _Continuation>
    using __sync_wait_result_impl = //
      __try_value_types_of_t<
        _Sender,
        __env,
        __transform<__q<__decay_t>, _Continuation>,
        __q<__msingle>>;

    template <class _Sender>
    using __sync_wait_result_t = __mtry_eval<__sync_wait_result_impl, _Sender, __q<std::tuple>>;

    template <class _Sender>
    using __sync_wait_with_variant_result_t =
      __mtry_eval<__sync_wait_result_impl, __result_of<into_variant, _Sender>, __q<__midentity>>;

    template <class... _Values>
    struct __state {
      using _Tuple = std::tuple<_Values...>;
      std::variant<std::monostate, _Tuple, std::exception_ptr, set_stopped_t> __data_{};
    };

    template <class... _Values>
    struct __receiver {
      struct __t {
        using receiver_concept = receiver_t;
        using __id = __receiver;
        __state<_Values...>* __state_;
        run_loop* __loop_;

        template <class _Error>
        void __set_error(_Error __err) noexcept {
          if constexpr (__decays_to<_Error, std::exception_ptr>)
            __state_->__data_.template emplace<2>(static_cast<_Error&&>(__err));
          else if constexpr (__decays_to<_Error, std::error_code>)
            __state_->__data_.template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
          else
            __state_->__data_.template emplace<2>(
              std::make_exception_ptr(static_cast<_Error&&>(__err)));
          __loop_->finish();
        }

        template <class... _As>
          requires constructible_from<std::tuple<_Values...>, _As...>
        STDEXEC_MEMFN_DECL(
          void set_value)(this __t&& __rcvr, _As&&... __as) noexcept {
          try {
            __rcvr.__state_->__data_.template emplace<1>(static_cast<_As&&>(__as)...);
            __rcvr.__loop_->finish();
          } catch (...) {
            __rcvr.__set_error(std::current_exception());
          }
        }

        template <class _Error>
        STDEXEC_MEMFN_DECL(void set_error)(this __t&& __rcvr, _Error __err) noexcept {
          __rcvr.__set_error(static_cast<_Error&&>(__err));
        }

        STDEXEC_MEMFN_DECL(void set_stopped)(this __t&& __rcvr) noexcept {
          __rcvr.__state_->__data_.template emplace<3>(set_stopped_t{});
          __rcvr.__loop_->finish();
        }

        STDEXEC_MEMFN_DECL(auto get_env)(this const __t& __rcvr) noexcept -> __env {
          return __env(*__rcvr.__loop_);
        }
      };
    };

    template <class _Sender>
    using __receiver_t = __t<__sync_wait_result_impl<_Sender, __q<__receiver>>>;

    // These are for hiding the metaprogramming in diagnostics
    template <class _Sender>
    struct __sync_receiver_for {
      using __t = __receiver_t<_Sender>;
    };
    template <class _Sender>
    using __sync_receiver_for_t = __t<__sync_receiver_for<_Sender>>;

    template <class _Sender>
    struct __value_tuple_for {
      using __t = __sync_wait_result_t<_Sender>;
    };
    template <class _Sender>
    using __value_tuple_for_t = __t<__value_tuple_for<_Sender>>;

    template <class _Sender>
    struct __variant_for {
      using __t = __sync_wait_with_variant_result_t<_Sender>;
    };
    template <class _Sender>
    using __variant_for_t = __t<__variant_for<_Sender>>;

    inline constexpr __mstring __sync_wait_context_diag = //
      "In stdexec::sync_wait()..."_mstr;
    inline constexpr __mstring __too_many_successful_completions_diag =
      "The argument to stdexec::sync_wait() is a sender that can complete successfully in more "
      "than one way. Use stdexec::sync_wait_with_variant() instead."_mstr;

    template <__mstring _Context, __mstring _Diagnostic>
    struct _INVALID_ARGUMENT_TO_SYNC_WAIT_;

    template <__mstring _Diagnostic>
    using __invalid_argument_to_sync_wait =
      _INVALID_ARGUMENT_TO_SYNC_WAIT_<__sync_wait_context_diag, _Diagnostic>;

    template <__mstring _Diagnostic, class _Sender, class _Env = __env>
    using __sync_wait_error = __mexception<
      __invalid_argument_to_sync_wait<_Diagnostic>,
      _WITH_SENDER_<_Sender>,
      _WITH_ENVIRONMENT_<_Env>>;

    template <class _Sender, class>
    using __too_many_successful_completions_error =
      __sync_wait_error<__too_many_successful_completions_diag, _Sender>;

    template <class _Sender>
    concept __valid_sync_wait_argument = __ok<__minvoke<
      __mtry_catch_q<__single_value_variant_sender_t, __q<__too_many_successful_completions_error>>,
      _Sender,
      __env>>;

#if STDEXEC_NVHPC()
    // It requires some hoop-jumping to get the NVHPC compiler to report a meaningful
    // diagnostic for SFINAE failures.
    template <class _Sender>
    auto __diagnose_error() {
      if constexpr (!sender_in<_Sender, __env>) {
        using _Completions = __completion_signatures_of_t<_Sender, __env>;
        if constexpr (__merror<_Completions>) {
          return _Completions();
        } else {
          constexpr __mstring __diag =
            "The stdexec::sender_in<Sender, Environment> concept check has failed."_mstr;
          return __sync_wait_error<__diag, _Sender>();
        }
      } else if constexpr (!__valid_sync_wait_argument<_Sender>) {
        return __sync_wait_error<__too_many_successful_completions_diag, _Sender>();
      } else if constexpr (!sender_to<_Sender, __sync_receiver_for_t<_Sender>>) {
        constexpr __mstring __diag =
          "Failed to connect the given sender to sync_wait's internal receiver. "
          "The stdexec::connect(Sender, Receiver) expression is ill-formed."_mstr;
        return __sync_wait_error<__diag, _Sender>();
      } else {
        constexpr __mstring __diag = "Unknown concept check failure."_mstr;
        return __sync_wait_error<__diag, _Sender>();
      }
    }

    template <class _Sender>
    using __error_description_t = decltype(__sync_wait::__diagnose_error<_Sender>());
#endif

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      template <sender_in<__env> _Sender>
        requires __valid_sync_wait_argument<_Sender>
              && __has_implementation_for<sync_wait_t, __early_domain_of_t<_Sender>, _Sender>
      auto operator()(_Sender&& __sndr) const -> std::optional<__value_tuple_for_t<_Sender>> {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr));
      }

#if STDEXEC_NVHPC()
      // This is needed to get sensible diagnostics from nvc++
      template <class _Sender, class _Error = __error_description_t<_Sender>>
      auto operator()(_Sender&&, [[maybe_unused]] _Error __diagnostic = {}) const
        -> std::optional<std::tuple<int>> = delete;
#endif

      using _Sender = __0;
      using __legacy_customizations_t = __types<
        // For legacy reasons:
        tag_invoke_t(
          sync_wait_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(sync_wait_t, _Sender)>;

      // clang-format off
      /// @brief Synchronously wait for the result of a sender, blocking the
      ///         current thread.
      ///
      /// `sync_wait` connects and starts the given sender, and then drives a
      ///         `run_loop` instance until the sender completes. Additional work
      ///         can be delegated to the `run_loop` by scheduling work on the
      ///         scheduler returned by calling `get_delegatee_scheduler` on the
      ///         receiver's environment.
      ///
      /// @pre The sender must have a exactly one value completion signature. That
      ///         is, it can only complete successfully in one way, with a single
      ///         set of values.
      ///
      /// @retval success Returns an engaged `std::optional` containing the result
      ///         values in a `std::tuple`.
      /// @retval canceled Returns an empty `std::optional`.
      /// @retval error Throws the error.
      ///
      /// @throws std::rethrow_exception(error) if the error has type
      ///         `std::exception_ptr`.
      /// @throws std::system_error(error) if the error has type
      ///         `std::error_code`.
      /// @throws error otherwise
      // clang-format on
      template <class _Sender>
        requires sender_to<_Sender, __sync_receiver_for_t<_Sender>>
      auto apply_sender(_Sender&& __sndr) const -> std::optional<__sync_wait_result_t<_Sender>> {
        using state_t = __sync_wait_result_impl<_Sender, __q<__state>>;
        state_t __state{};
        run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state =
          connect(static_cast<_Sender&&>(__sndr), __receiver_t<_Sender>{&__state, &__loop});
        start(__op_state);

        // Wait for the variant to be filled in.
        __loop.run();

        if (__state.__data_.index() == 2)
          std::rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return std::nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait_with_variant]
    struct sync_wait_with_variant_t {
      struct __impl;

      template <sender_in<__env> _Sender>
        requires __callable<
          apply_sender_t,
          __early_domain_of_t<_Sender>,
          sync_wait_with_variant_t,
          _Sender>
      auto operator()(_Sender&& __sndr) const -> std::optional<__variant_for_t<_Sender>> {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr));
      }

#if STDEXEC_NVHPC()
      template <
        class _Sender,
        class _Error = __error_description_t<__result_of<into_variant, _Sender>>>
      auto operator()(_Sender&&, [[maybe_unused]] _Error __diagnostic = {}) const
        -> std::optional<std::tuple<std::variant<std::tuple<>>>> = delete;
#endif

      using _Sender = __0;
      using __legacy_customizations_t = __types<
        // For legacy reasons:
        tag_invoke_t(
          sync_wait_with_variant_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(sync_wait_with_variant_t, _Sender)>;

      template <class _Sender>
        requires __callable<sync_wait_t, __result_of<into_variant, _Sender>>
      auto apply_sender(_Sender&& __sndr) const -> std::optional<__variant_for_t<_Sender>> {
        if (auto __opt_values = sync_wait_t()(into_variant(static_cast<_Sender&&>(__sndr)))) {
          return std::move(std::get<0>(*__opt_values));
        }
        return std::nullopt;
      }
    };
  } // namespace __sync_wait

  using __sync_wait::sync_wait_t;
  inline constexpr sync_wait_t sync_wait{};

  using __sync_wait::sync_wait_with_variant_t;
  inline constexpr sync_wait_with_variant_t sync_wait_with_variant{};

  //////////////////////////////////////////////////////////////////////////////////////////////////
  struct __ignore_sender {
    using sender_concept = sender_t;

    template <sender _Sender>
    constexpr __ignore_sender(_Sender&&) noexcept {
    }
  };

  template <auto _Reason = "You cannot pipe one sender into another."_mstr>
  struct _CANNOT_PIPE_INTO_A_SENDER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_CANNOT_PIPE_INTO_A_SENDER_<>, _WITH_SENDER_<_Sender>>;
} // namespace stdexec

#if STDEXEC_MSVC() && _MSC_VER >= 1939
namespace stdexec {
  // MSVCBUG https://developercommunity.visualstudio.com/t/destroy-coroutine-from-final_suspend-r/10096047

  // Prior to Visual Studio 17.9 (Feb, 2024), aka MSVC 19.39, MSVC incorrectly allocates the return
  // buffer for await_suspend calls within the suspended coroutine frame. When the suspended
  // coroutine is destroyed within await_suspend, the continuation coroutine handle is not only used
  // after free, but also overwritten by the debug malloc implementation when NRVO is in play.

  // This workaround delays the destruction of the suspended coroutine by wrapping the continuation
  // in another coroutine which destroys the former and transfers execution to the original
  // continuation.

  // The wrapping coroutine is thread-local and is reused within the thread for each
  // destroy-and-continue sequence. The wrapping coroutine itself is destroyed at thread exit.

  namespace __destroy_and_continue_msvc {
    struct __task {
      struct promise_type {
        __task get_return_object() noexcept {
          return {__coro::coroutine_handle<promise_type>::from_promise(*this)};
        }

        static std::suspend_never initial_suspend() noexcept {
          return {};
        }

        static std::suspend_never final_suspend() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
          return {};
        }

        static void return_void() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
        }

        static void unhandled_exception() noexcept {
          STDEXEC_ASSERT(!"Should never get here");
        }
      };

      __coro::coroutine_handle<> __coro_;
    };

    struct __continue_t {
      static constexpr bool await_ready() noexcept {
        return false;
      }

      __coro::coroutine_handle<> await_suspend(__coro::coroutine_handle<>) noexcept {
        return __continue_;
      }

      static void await_resume() noexcept {
      }

      __coro::coroutine_handle<> __continue_;
    };

    struct __context {
      __coro::coroutine_handle<> __destroy_;
      __coro::coroutine_handle<> __continue_;
    };

    inline __task __co_impl(__context& __c) {
      while (true) {
        co_await __continue_t{__c.__continue_};
        __c.__destroy_.destroy();
      }
    }

    struct __context_and_coro {
      __context_and_coro() {
        __context_.__continue_ = __coro::noop_coroutine();
        __coro_ = __co_impl(__context_).__coro_;
      }

      ~__context_and_coro() {
        __coro_.destroy();
      }

      __context __context_;
      __coro::coroutine_handle<> __coro_;
    };

    inline __coro::coroutine_handle<>
      __impl(__coro::coroutine_handle<> __destroy, __coro::coroutine_handle<> __continue) {
      static thread_local __context_and_coro __c;
      __c.__context_.__destroy_ = __destroy;
      __c.__context_.__continue_ = __continue;
      return __c.__coro_;
    }
  } // namespace __destroy_and_continue_msvc
} // namespace stdexec

#  define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue)                                      \
    (::stdexec::__destroy_and_continue_msvc::__impl(__destroy, __continue))
#else
#  define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue) (__destroy.destroy(), __continue)
#endif

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <stdexec::sender _Sender>
  requires stdexec::__ok<stdexec::__bad_pipe_sink_t<_Sender>>
auto operator|(stdexec::__ignore_sender, _Sender&&) noexcept -> stdexec::__ignore_sender;

#include "__detail/__p2300.hpp"

STDEXEC_PRAGMA_POP()
