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

#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/execution.hpp"

namespace exec {
  struct AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT;
  struct A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS;

  struct just_from_t;
  struct just_error_from_t;
  struct just_stopped_from_t;

  namespace detail {
    auto _just_from(just_from_t*) -> stdexec::set_value_t;
    auto _just_from(just_error_from_t*) -> stdexec::set_error_t;
    auto _just_from(just_stopped_from_t*) -> stdexec::set_stopped_t;
  } // namespace detail

  template <class JustTag>
  struct _just_from { // NOLINT(bugprone-crtp-constructor-accessibility)
   private:
    friend JustTag;
    using _set_tag_t = decltype(detail::_just_from(static_cast<JustTag*>(nullptr)));

    using _diag_t = stdexec::__if_c<
      STDEXEC_IS_SAME(_set_tag_t, stdexec::set_error_t),
      AN_ERROR_COMPLETION_MUST_HAVE_EXACTLY_ONE_ERROR_ARGUMENT,
      A_STOPPED_COMPLETION_MUST_HAVE_NO_ARGUMENTS
    >;

    template <class... Ts>
    using _error_t = stdexec::_ERROR_<
      stdexec::_WHAT_<>(_diag_t),
      stdexec::_WHERE_(stdexec::_IN_ALGORITHM_, JustTag),
      stdexec::_WITH_COMPLETION_SIGNATURE_<_set_tag_t(Ts...)>
    >;

    struct _probe_fn {
      template <class... Ts>
      auto operator()(Ts&&... ts) const noexcept -> _error_t<Ts...>;

      template <class... Ts>
        requires stdexec::__sigs::__is_compl_sig<_set_tag_t(Ts...)>
      auto operator()(Ts&&...) const noexcept -> stdexec::completion_signatures<_set_tag_t(Ts...)> {
        return {};
      }
    };

    template <class Rcvr>
    struct _complete_fn {
      Rcvr& _rcvr;

      template <class... Ts>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      void operator()(Ts&&... ts) const noexcept {
        _set_tag_t()(static_cast<Rcvr&&>(_rcvr), static_cast<Ts&&>(ts)...);
      }
    };

    template <class Rcvr, class Fn>
    struct _opstate {
      using operation_state_concept = stdexec::operation_state_t;

      Rcvr _rcvr;
      Fn _fn;

      STDEXEC_ATTRIBUTE(host, device) void start() & noexcept {
        if constexpr (stdexec::__nothrow_callable<Fn, _complete_fn<Rcvr>>) {
          static_cast<Fn&&>(_fn)(_complete_fn<Rcvr>{_rcvr});
        } else {
          STDEXEC_TRY {
            static_cast<Fn&&>(_fn)(_complete_fn<Rcvr>{_rcvr});
          }
          STDEXEC_CATCH_ALL {
            stdexec::set_error(static_cast<Rcvr&&>(_rcvr), std::current_exception());
          }
        }
      }
    };

    struct _nothrow_completions {
      template <class Fn>
      using __f = stdexec::__call_result_t<Fn, _probe_fn>;
    };

    struct _throw_completions {
      template <class Fn>
      using __f = stdexec::__concat_completion_signatures<
        stdexec::__call_result_t<Fn, _probe_fn>,
        stdexec::__eptr_completion
      >;
    };

    template <class Fn>
    using _completions = stdexec::__minvoke_if_c<
      stdexec::__nothrow_callable<Fn, _probe_fn>,
      _nothrow_completions,
      _throw_completions,
      Fn
    >;

    template <class Fn>
    struct _sndr_base {
      using sender_concept = stdexec::sender_t;
      using completion_signatures = _completions<Fn>;

      STDEXEC_ATTRIBUTE(no_unique_address) JustTag _tag;
      Fn _fn;

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      auto connect(Rcvr rcvr) && noexcept(stdexec::__nothrow_decay_copyable<Rcvr, Fn>)
        -> _opstate<Rcvr, Fn> {
        return _opstate<Rcvr, Fn>{static_cast<Rcvr&&>(rcvr), static_cast<Fn&&>(_fn)};
      }

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      auto connect(Rcvr rcvr) const & noexcept(stdexec::__nothrow_decay_copyable<Rcvr, Fn const &>)
        -> _opstate<Rcvr, Fn> {
        return _opstate<Rcvr, Fn>{static_cast<Rcvr&&>(rcvr), _fn};
      }

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      auto submit(Rcvr rcvr) && noexcept -> void {
        auto op = static_cast<_sndr_base&&>(*this).connect(static_cast<Rcvr&&>(rcvr));
        stdexec::start(op);
      }

      template <class Rcvr>
      STDEXEC_ATTRIBUTE(host, device)
      auto submit(Rcvr rcvr) const & noexcept -> void {
        auto op = this->connect(static_cast<Rcvr&&>(rcvr));
        stdexec::start(op);
      }
    };

    template <class Fn, class Tag = JustTag>
    using _sndr = typename Tag::template _sndr<Fn>;

   public:
    template <class Fn>
    STDEXEC_ATTRIBUTE(always_inline, host, device)
    auto operator()(Fn fn) const noexcept {
      if constexpr (stdexec::__callable<Fn, _probe_fn>) {
        using _completions = stdexec::__call_result_t<Fn, _probe_fn>;
        static_assert(
          stdexec::__sigs::__is_completion_signatures<_completions>,
          "The function passed to just_from, just_error_from, and just_stopped_from must return an "
          "instance of a specialization of stdexec::completion_signatures<>.");
        return _sndr<Fn>{
          {{}, static_cast<Fn&&>(fn)}
        };
      } else {
        static_assert(
          stdexec::__callable<Fn, _probe_fn>,
          "The function passed to just_from, just_error_from, and just_stopped_from must be "
          "callable with a sink function.");
      }
    }
  };

  //! @brief `just_from(fn)` creates a sender that completes inline by passing a "sink" function to
  //! `fn`. Calling the sink function with arguments sends the arguments as values to the receiver.
  //!
  //! @post The sink function passed to `fn` must be called exactly once.
  //!
  //! @param fn The callable to be invoked when the sender is started.
  //!
  //! @par
  //! The function passed to `just_from` must return an instance of a specialization of
  //! `stdexec::completion_signatures<>` that describes the ways the sink function might be
  //! invoked. The sink function returns such a specialization of `stdexec::completion_signatures<>`
  //! corresponding to the arguments passed to it, but if your function uses the sink function
  //! in several different ways, you must specify the return type explicitly.
  //!
  //! @par Example:
  //! @code
  //! // The following sender is equivalent to just(42, 3.14):
  //! auto sndr = exec::just_from([](auto sink) { return sink(42, 3.14); });
  //! @endcode
  //!
  //! @par Example:
  //! @code
  //! // A just_from sender can have multiple completion signatures:
  //! auto sndr = exec::just_from(
  //!   [](auto sink) {
  //!     if (some-condition) {
  //!       sink(42);
  //!     } else {
  //!       sink(3.14);
  //!     }
  //!     return stdexec::completion_signatures<stdexec::set_value_t(int),
  //!                                           stdexec::set_value_t(double)>{};
  //!   });
  //! @endcode
  struct just_from_t : _just_from<just_from_t> {
    template <class _Fn>
    struct _sndr : _just_from::_sndr_base<_Fn> { };
  };

  //! @brief `just_error_from(fn)` creates a sender that completes inline by passing a "sink"
  //! function to `fn`. Calling the sink function with an argument sends that argument as an error
  //! to the receiver.
  //!
  //! @sa just_from
  struct just_error_from_t : _just_from<just_error_from_t> {
    template <class _Fn>
    struct _sndr : _just_from::_sndr_base<_Fn> { };
  };

  //! @brief `just_stopped_from(fn)` creates a sender that completes inline by passing a "sink"
  //! function to `fn`. Calling the sink function with no arguments sends a stopped signal to the
  //! receiver.
  //!
  //! @sa just_from
  struct just_stopped_from_t : _just_from<just_stopped_from_t> {
    template <class _Fn>
    struct _sndr : _just_from::_sndr_base<_Fn> { };
  };

  inline constexpr just_from_t just_from{};
  inline constexpr just_error_from_t just_error_from{};
  inline constexpr just_stopped_from_t just_stopped_from{};

} // namespace exec
