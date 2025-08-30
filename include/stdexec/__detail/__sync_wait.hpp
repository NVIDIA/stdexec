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
#include "__concepts.hpp"
#include "__debug.hpp" // IWYU pragma: keep
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__into_variant.hpp"
#include "__meta.hpp"
#include "__senders.hpp"
#include "__receivers.hpp"
#include "__transform_completion_signatures.hpp"
#include "__transform_sender.hpp"
#include "__run_loop.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <system_error>
#include <optional>
#include <tuple>
#include <variant>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    struct __env {
      using __t = __env;
      using __id = __env;

      run_loop* __loop_ = nullptr;

      [[nodiscard]]
      auto query(get_scheduler_t) const noexcept -> run_loop::__scheduler {
        return __loop_->get_scheduler();
      }

      [[nodiscard]]
      auto query(get_delegation_scheduler_t) const noexcept -> run_loop::__scheduler {
        return __loop_->get_scheduler();
      }

      [[nodiscard]]
      constexpr auto query(__root_t) const noexcept -> bool {
        return true;
      }

      // static constexpr auto query(__debug::__is_debug_env_t) noexcept -> bool
      // {
      //   return true;
      // }
    };

    // What should sync_wait(just_stopped()) return?
    template <class _Sender, class _Continuation>
    using __sync_wait_result_impl = __value_types_of_t<
      _Sender,
      __env,
      __mtransform<__q<__decay_t>, _Continuation>,
      __q<__msingle>
    >;

    template <class _Sender>
    using __sync_wait_result_t = __mtry_eval<__sync_wait_result_impl, _Sender, __qq<std::tuple>>;

    template <class _Sender>
    using __sync_wait_with_variant_result_t =
      __mtry_eval<__sync_wait_result_impl, __result_of<into_variant, _Sender>, __q<__midentity>>;

    struct __state {
      std::exception_ptr __eptr_;
      run_loop __loop_;
    };

    template <class... _Values>
    struct __receiver {
      struct __t {
        using receiver_concept = receiver_t;
        using __id = __receiver;
        __state* __state_;
        std::optional<std::tuple<_Values...>>* __values_;

        template <class... _As>
          requires constructible_from<std::tuple<_Values...>, _As...>
        void set_value(_As&&... __as) noexcept {
          STDEXEC_TRY {
            __values_->emplace(static_cast<_As&&>(__as)...);
          }
          STDEXEC_CATCH_ALL {
            __state_->__eptr_ = std::current_exception();
          }
          __state_->__loop_.finish();
        }

        template <class _Error>
        void set_error(_Error __err) noexcept {
          if constexpr (__same_as<_Error, std::exception_ptr>) {
            STDEXEC_ASSERT(__err != nullptr); // std::exception_ptr must not be null.
            __state_->__eptr_ = static_cast<_Error&&>(__err);
          } else if constexpr (__same_as<_Error, std::error_code>) {
            __state_->__eptr_ = std::make_exception_ptr(std::system_error(__err));
          } else {
            __state_->__eptr_ = std::make_exception_ptr(static_cast<_Error&&>(__err));
          }
          __state_->__loop_.finish();
        }

        void set_stopped() noexcept {
          __state_->__loop_.finish();
        }

        [[nodiscard]]
        auto get_env() const noexcept -> __env {
          return __env{&__state_->__loop_};
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

    inline constexpr __mstring __sync_wait_context_diag = "In stdexec::sync_wait()..."_mstr;
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
      _WITH_ENVIRONMENT_<_Env>
    >;

    template <class _Sender, class>
    using __too_many_successful_completions_error =
      __sync_wait_error<__too_many_successful_completions_diag, _Sender>;

    template <class _Sender>
    concept __valid_sync_wait_argument = __ok<__minvoke<
      __mtry_catch_q<__single_value_variant_sender_t, __q<__too_many_successful_completions_error>>,
      _Sender,
      __env
    >>;

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      template <class _Sender>
      auto operator()(_Sender&& __sndr) const {
        if constexpr (!sender_in<_Sender, __env>) {
          stdexec::__diagnose_sender_concept_failure<_Sender, __env>();
        } else {
          using __early_domain_t = __early_domain_of_t<_Sender>;
          using __domain_t = __late_domain_of_t<_Sender, __env, __early_domain_t>;
          constexpr auto __success_completion_count =
            __v<value_types_of_t<_Sender, __env, __types, __msize::__f>>;
          static_assert(
            __success_completion_count != 0,
            "The argument to stdexec::sync_wait() is a sender that cannot complete successfully. "
            "stdexec::sync_wait() requires a sender that can complete successfully in exactly one "
            "way. In other words, the sender's completion signatures must include exactly one "
            "signature of the form `set_value_t(value-types...)`.");
          static_assert(
            __success_completion_count <= 1,
            "The sender passed to stdexec::sync_wait() can complete successfully in "
            "more than one way. Use stdexec::sync_wait_with_variant() instead.");
          if constexpr (1 == __success_completion_count) {
            using __sync_wait_receiver = __receiver_t<_Sender>;
            constexpr bool __no_custom_sync_wait = __same_as<__domain_t, default_domain>;
            if constexpr (__no_custom_sync_wait && sender_to<_Sender, __sync_wait_receiver>) {
              // using __connect_result = connect_result_t<_Sender, __sync_wait_receiver>;
              // if constexpr (!operation_state<__connect_result>) {
              //   static_assert(
              //     operation_state<__connect_result>,
              //     "The `connect` member function of the sender passed to stdexec::sync_wait() does "
              //     "not return an operation state. An operation state is required to have a "
              //     "no-throw .start() member function.");
              // } else
              {
                // success path, dispatch to the default domain's sync_wait
                return default_domain().apply_sender(*this, static_cast<_Sender&&>(__sndr));
              }
            } else if constexpr (__no_custom_sync_wait) {
              static_assert(
                sender_to<_Sender, __sync_wait_receiver>,
                STDEXEC_ERROR_SYNC_WAIT_CANNOT_CONNECT_SENDER_TO_RECEIVER);
            } else if constexpr (!__has_implementation_for<sync_wait_t, __domain_t, _Sender>) {
              static_assert(
                __has_implementation_for<sync_wait_t, __domain_t, _Sender>,
                "The sender passed to stdexec::sync_wait() has a domain that does not provide a "
                "usable implementation for sync_wait().");
            } else {
              // success path, dispatch to the custom domain's sync_wait
              return stdexec::apply_sender(__domain_t(), *this, static_cast<_Sender&&>(__sndr));
            }
          }
        }
      }

      // clang-format off
      /// @brief Synchronously wait for the result of a sender, blocking the
      ///         current thread.
      ///
      /// `sync_wait` connects and starts the given sender, and then drives a
      ///         `run_loop` instance until the sender completes. Additional work
      ///         can be delegated to the `run_loop` by scheduling work on the
      ///         scheduler returned by calling `get_delegation_scheduler` on the
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

      template <sender_in<__env> _Sender>
      auto apply_sender(_Sender&& __sndr) const -> std::optional<__sync_wait_result_t<_Sender>> {
        __state __local_state{};
        std::optional<__sync_wait_result_t<_Sender>> __result{};

        // Launch the sender with a continuation that will fill in the __result optional or set the
        // exception_ptr in __local_state.
        [[maybe_unused]]
        auto __op = stdexec::connect(
          static_cast<_Sender&&>(__sndr), __receiver_t<_Sender>{&__local_state, &__result});
        stdexec::start(__op);

        // Wait for the variant to be filled in.
        __local_state.__loop_.run();

        if (__local_state.__eptr_) {
          std::rethrow_exception(static_cast<std::exception_ptr&&>(__local_state.__eptr_));
        }

        return __result;
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
          _Sender
        >
      auto operator()(_Sender&& __sndr) const -> decltype(auto) {
        using __result_t = __call_result_t<
          apply_sender_t,
          __early_domain_of_t<_Sender>,
          sync_wait_with_variant_t,
          _Sender
        >;
        static_assert(__is_instance_of<__result_t, std::optional>);
        using __variant_t = typename __result_t::value_type;
        static_assert(__is_instance_of<__variant_t, std::variant>);

        using _Domain = __late_domain_of_t<_Sender, __env>;
        return stdexec::apply_sender(_Domain(), *this, static_cast<_Sender&&>(__sndr));
      }

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
} // namespace stdexec
