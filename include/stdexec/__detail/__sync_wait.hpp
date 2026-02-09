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
#include "__receivers.hpp"
#include "__run_loop.hpp"
#include "__senders.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

#include <exception>
#include <optional>
#include <system_error>
#include <tuple>
#include <variant>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    struct sync_wait_t;

    struct __env {
      run_loop* __loop_ = nullptr;

      [[nodiscard]]
      constexpr auto query(get_scheduler_t) const noexcept -> run_loop::scheduler {
        return __loop_->get_scheduler();
      }

      [[nodiscard]]
      constexpr auto query(get_delegation_scheduler_t) const noexcept -> run_loop::scheduler {
        return __loop_->get_scheduler();
      }

      [[nodiscard]]
      constexpr auto query(__root_t) const noexcept -> bool {
        return true;
      }
    };

    // What should sync_wait(just_stopped()) return?
    template <class _CvSender, class _Continuation>
    using __result_t = __value_types_of_t<
      _CvSender,
      __env,
      __mtransform<__q<__decay_t>, _Continuation>,
      __q<__msingle>
    >;

    template <class _CvSender>
    using __sync_wait_result_t = __result_t<_CvSender, __qq<std::tuple>>;

    template <class _CvSender>
    using __sync_wait_with_variant_result_t =
      __result_t<__result_of<into_variant, _CvSender>, __q<__midentity>>;

    struct __state {
      std::exception_ptr __eptr_;
      run_loop __loop_;
    };

    template <class... _Values>
    struct __receiver {
      using receiver_concept = receiver_t;

      template <class... _As>
      constexpr void set_value(_As&&... __as) noexcept {
        static_assert(__std::constructible_from<std::tuple<_Values...>, _As...>);
        STDEXEC_TRY {
          __values_->emplace(static_cast<_As&&>(__as)...);
        }
        STDEXEC_CATCH_ALL {
          __state_->__eptr_ = std::current_exception();
        }
        __state_->__loop_.finish();
      }

      template <class _Error>
      constexpr void set_error(_Error __err) noexcept {
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

      constexpr void set_stopped() noexcept {
        __state_->__loop_.finish();
      }

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> __env {
        return __env{&__state_->__loop_};
      }
      __state* __state_;
      std::optional<std::tuple<_Values...>>* __values_;
    };

    template <class _CvSender>
    using __receiver_t = __result_t<_CvSender, __q<__receiver>>;

    // These are for hiding the metaprogramming in diagnostics
    template <class _CvSender>
    struct __sync_receiver_for {
      using __t = __receiver_t<_CvSender>;
    };
    template <class _CvSender>
    using __sync_receiver_for_t = __t<__sync_receiver_for<_CvSender>>;

    template <class _CvSender>
    struct __value_tuple_for {
      using __t = __sync_wait_result_t<_CvSender>;
    };
    template <class _CvSender>
    using __value_tuple_for_t = __t<__value_tuple_for<_CvSender>>;

    template <class _CvSender>
    struct __variant {
      using __t = __sync_wait_with_variant_result_t<_CvSender>;
    };
    template <class _CvSender>
    using __variant_for_t = __t<__variant<_CvSender>>;

    struct _SENDER_HAS_TOO_MANY_SUCCESSFUL_COMPLETIONS_ { };
    struct _USE_SYNC_WAIT_WITH_VARIANT_INSTEAD_ { };

    template <class _Reason, class _CvSender, class _Env = __env>
    using __sync_wait_error_t = __mexception<
      _WHAT_(_INVALID_ARGUMENT_),
      _WHERE_(_IN_ALGORITHM_, sync_wait_t),
      _WHY_(_Reason),
      _WITH_PRETTY_SENDER_<_CvSender>,
      _WITH_ENVIRONMENT_(_Env),
      _TO_FIX_THIS_ERROR_(_USE_SYNC_WAIT_WITH_VARIANT_INSTEAD_)
    >;

    template <class _CvSender, class>
    using __too_many_successful_completions_error_t =
      __sync_wait_error_t<_SENDER_HAS_TOO_MANY_SUCCESSFUL_COMPLETIONS_, _CvSender>;

    template <class _CvSender>
    concept __valid_sync_wait_argument = __ok<__minvoke<
      __mtry_catch_q<__single_value_variant_sender_t, __q<__too_many_successful_completions_error_t>>,
      _CvSender,
      __env
    >>;

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    struct sync_wait_t {
      template <sender_in<__env> _CvSender>
      auto operator()(_CvSender&& __sndr) const {
        using __domain_t = __completion_domain_of_t<set_value_t, _CvSender, __env>;
        constexpr auto __success_completion_count =
          __count_of<set_value_t, _CvSender, __env>::value;

        static_assert(
          __success_completion_count != 0,
          "The argument to STDEXEC::sync_wait() is a sender that cannot complete successfully. "
          "STDEXEC::sync_wait() requires a sender that can complete successfully in exactly one "
          "way. In other words, the sender's completion signatures must include exactly one "
          "signature of the form `set_value_t(value-types...)`.");

        static_assert(
          __success_completion_count <= 1,
          "The sender passed to STDEXEC::sync_wait() can complete successfully in "
          "more than one way. Use STDEXEC::sync_wait_with_variant() instead.");

        if constexpr (1 == __success_completion_count) {
          if constexpr (__same_as<__domain_t, default_domain>) {
            if constexpr (sender_to<_CvSender, __receiver_t<_CvSender>>) {
              using __opstate_t = connect_result_t<_CvSender, __receiver_t<_CvSender>>;
              if constexpr (operation_state<__opstate_t>) {
                // success path, dispatch to the default domain's sync_wait
                return default_domain().apply_sender(*this, static_cast<_CvSender&&>(__sndr));
              } else {
                static_assert(
                  operation_state<__opstate_t>,
                  "The `connect` member function of the sender passed to STDEXEC::sync_wait() "
                  "does not return an operation state. An operation state is required to have a "
                  "no-throw .start() member function.");
              }
            } else {
              // This shoud generate a useful error message about why the sender cannot
              // be connected to the receiver:
              connect(static_cast<_CvSender&&>(__sndr), __receiver_t<_CvSender>{});
              static_assert(
                sender_to<_CvSender, __receiver_t<_CvSender>>,
                STDEXEC_ERROR_SYNC_WAIT_CANNOT_CONNECT_SENDER_TO_RECEIVER);
            }
          } else if constexpr (!__has_implementation_for<sync_wait_t, __domain_t, _CvSender>) {
            static_assert(
              __has_implementation_for<sync_wait_t, __domain_t, _CvSender>,
              "The sender passed to STDEXEC::sync_wait() has a domain that does not provide a "
              "usable implementation for sync_wait().");
          } else {
            // success path, dispatch to the custom domain's sync_wait
            return STDEXEC::apply_sender(__domain_t(), *this, static_cast<_CvSender&&>(__sndr));
          }
        }
      }

      template <class _CvSender>
      constexpr auto operator()(_CvSender&&) const {
        STDEXEC::__diagnose_sender_concept_failure<__demangle_t<_CvSender>, __env>();
        // dummy return type to silence follow-on errors
        return std::optional<std::tuple<int>>{};
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

      template <sender_in<__env> _CvSender>
      STDEXEC_CONSTEXPR_CXX23 auto apply_sender(_CvSender&& __sndr) const //
        -> std::optional<__value_tuple_for_t<_CvSender>> {
        __state __local_state{};
        std::optional<__value_tuple_for_t<_CvSender>> __result{};

        // Launch the sender with a continuation that will fill in the __result optional or set the
        // exception_ptr in __local_state.
        [[maybe_unused]]
        auto __op = STDEXEC::connect(
          static_cast<_CvSender&&>(__sndr), __receiver_t<_CvSender>{&__local_state, &__result});
        STDEXEC::start(__op);

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

      template <sender_in<__env> _CvSender>
        requires __callable<
          apply_sender_t,
          __completion_domain_of_t<set_value_t, _CvSender, __env>,
          sync_wait_with_variant_t,
          _CvSender
        >
      auto operator()(_CvSender&& __sndr) const -> decltype(auto) {
        using __result_t = __call_result_t<
          apply_sender_t,
          __completion_domain_of_t<set_value_t, _CvSender, __env>,
          sync_wait_with_variant_t,
          _CvSender
        >;
        static_assert(__is_instance_of<__result_t, std::optional>);
        using __variant_t = __result_t::value_type;
        static_assert(__is_instance_of<__variant_t, std::variant>);

        using _Domain = __completion_domain_of_t<set_value_t, _CvSender, __env>;
        return STDEXEC::apply_sender(_Domain(), *this, static_cast<_CvSender&&>(__sndr));
      }

      template <class _CvSender>
        requires __callable<sync_wait_t, __result_of<into_variant, _CvSender>>
      auto apply_sender(_CvSender&& __sndr) const -> std::optional<__variant_for_t<_CvSender>> {
        if (auto __opt_values = sync_wait_t()(into_variant(static_cast<_CvSender&&>(__sndr)))) {
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
} // namespace STDEXEC
