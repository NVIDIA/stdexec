/*
 * Copyright (c) 2022-2024 NVIDIA Corporation
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

#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__tag_invoke.hpp"

#include "../functional.hpp"

#include <exception>

namespace STDEXEC {
  enum class __disposition {
    __value,
    __error,
    __stopped
  };

  namespace __detail {
    template <__disposition _Disposition>
    struct __completion_tag {
      static constexpr STDEXEC::__disposition __disposition = _Disposition;

      template <STDEXEC::__disposition _OtherDisposition>
      constexpr bool operator==(__completion_tag<_OtherDisposition>) const noexcept {
        return _Disposition == _OtherDisposition;
      }
    };
  } // namespace __detail

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  namespace __rcvrs {
    template <class _Receiver, class... _As>
    concept __set_value_member = requires(_Receiver &&__rcvr, _As &&...__args) {
      static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__args)...);
    };

    struct set_value_t : __detail::__completion_tag<__disposition::__value> {
      template <class _Fn, class... _As>
      using __f = __minvoke<_Fn, _As...>;

      template <class _Receiver, class... _As>
        requires __set_value_member<_Receiver, _As...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr void operator()(_Receiver &&__rcvr, _As &&...__as) const noexcept {
        static_assert(
          noexcept(static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__as)...)),
          "set_value member functions must be noexcept");
        static_assert(
          __same_as<
            decltype(static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__as)...)),
            void
          >,
          "set_value member functions must return void");
        static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__as)...);
      }

      template <class _Receiver, class... _As>
        requires __set_value_member<_Receiver, _As...>
              || __tag_invocable<set_value_t, _Receiver, _As...>
      [[deprecated("the use of tag_invoke for set_value is deprecated")]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) //
        constexpr void operator()(_Receiver &&__rcvr, _As &&...__as) const noexcept {
        static_assert(__nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) __tag_invoke(*this, static_cast<_Receiver &&>(__rcvr), static_cast<_As &&>(__as)...);
      }
    };

    template <class _Receiver, class _Error>
    concept __set_error_member = requires(_Receiver &&__rcvr, _Error &&__err) {
      static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err));
    };

    struct set_error_t : __detail::__completion_tag<__disposition::__error> {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 1)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class _Error>
        requires __set_error_member<_Receiver, _Error>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr void operator()(_Receiver &&__rcvr, _Error &&__err) const noexcept {
        static_assert(
          noexcept(static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err))),
          "set_error member functions must be noexcept");
        static_assert(
          __same_as<
            decltype(static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err))),
            void
          >,
          "set_error member functions must return void");
        static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err));
      }

      template <class _Receiver, class _Error>
        requires __set_error_member<_Receiver, _Error>
              || __tag_invocable<set_error_t, _Receiver, _Error>
      [[deprecated("the use of tag_invoke for set_error is deprecated")]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) //
        constexpr void operator()(_Receiver &&__rcvr, _Error &&__err) const noexcept {
        static_assert(__nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
        (void)
          __tag_invoke(*this, static_cast<_Receiver &&>(__rcvr), static_cast<_Error &&>(__err));
      }
    };

    template <class _Receiver>
    concept __set_stopped_member = requires(_Receiver &&__rcvr) {
      static_cast<_Receiver &&>(__rcvr).set_stopped();
    };

    struct set_stopped_t : __detail::__completion_tag<__disposition::__stopped> {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 0)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver>
        requires __set_stopped_member<_Receiver>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      constexpr void operator()(_Receiver &&__rcvr) const noexcept {
        static_assert(
          noexcept(static_cast<_Receiver &&>(__rcvr).set_stopped()),
          "set_stopped member functions must be noexcept");
        static_assert(
          __same_as<decltype(static_cast<_Receiver &&>(__rcvr).set_stopped()), void>,
          "set_stopped member functions must return void");
        static_cast<_Receiver &&>(__rcvr).set_stopped();
      }

      template <class _Receiver>
        requires __set_stopped_member<_Receiver> || __tag_invocable<set_stopped_t, _Receiver>
      [[deprecated("the use of tag_invoke for set_stopped is deprecated")]]
      STDEXEC_ATTRIBUTE(host, device, always_inline) //
        constexpr void operator()(_Receiver &&__rcvr) const noexcept {
        static_assert(__nothrow_tag_invocable<set_stopped_t, _Receiver>);
        (void) __tag_invoke(*this, static_cast<_Receiver &&>(__rcvr));
      }
    };
  } // namespace __rcvrs

  using __rcvrs::set_value_t;
  using __rcvrs::set_error_t;
  using __rcvrs::set_stopped_t;
  inline constexpr set_value_t set_value{};
  inline constexpr set_error_t set_error{};
  inline constexpr set_stopped_t set_stopped{};

  struct receiver_t {
    using receiver_concept = receiver_t; // NOT TO SPEC
  };

  namespace __detail {
    template <class _Receiver>
    concept __enable_receiver =
      (STDEXEC_PP_WHEN(
        STDEXEC_EDG(),
        requires { typename _Receiver::receiver_concept; } &&)
         __std::derived_from<typename _Receiver::receiver_concept, receiver_t>)
      || requires { typename _Receiver::is_receiver; } // back-compat, NOT TO SPEC
      || STDEXEC_IS_BASE_OF(receiver_t, _Receiver);    // NOT TO SPEC, for receiver_adaptor
  } // namespace __detail

  template <class _Receiver>
  inline constexpr bool enable_receiver = __detail::__enable_receiver<_Receiver>; // NOT TO SPEC

  template <class _Receiver>
  concept receiver = enable_receiver<__decay_t<_Receiver>>
                  && environment_provider<__cref_t<_Receiver>>
                  && __nothrow_move_constructible<__decay_t<_Receiver>>
                  && __std::constructible_from<__decay_t<_Receiver>, _Receiver>;

  struct _THE_RECEIVER_DOES_NOT_ACCEPT_ALL_OF_THE_SENDERS_COMPLETION_SIGNALS_ { };

  namespace __detail {
    template <class _Receiver, class _Tag, class... _Args>
    constexpr auto __try_completion(_Tag (*)(_Args...)) -> __mexception<
      _WHAT_(_CONCEPT_CHECK_FAILURE_),
      _WHY_(_THE_RECEIVER_DOES_NOT_ACCEPT_ALL_OF_THE_SENDERS_COMPLETION_SIGNALS_),
      _UNHANDLED_COMPLETION_SIGNAL_<_Tag(_Args...)>,
      _WITH_RECEIVER_(_Receiver)
    >;

    template <class _Receiver, class _Tag, class... _Args>
      requires __callable<_Tag, _Receiver, _Args...>
    auto __try_completion(_Tag (*)(_Args...)) -> __msuccess;

    template <class _Receiver, class... _Sigs>
    constexpr auto __try_completions(completion_signatures<_Sigs...> *) -> decltype((
      __msuccess(),
      ...,
      __detail::__try_completion<__decay_t<_Receiver>>(static_cast<_Sigs *>(nullptr))));
  } // namespace __detail

  template <class _Receiver, class _Completions>
  concept receiver_of = receiver<_Receiver> && requires(_Completions *__completions) {
    { __detail::__try_completions<_Receiver>(__completions) } -> __ok;
  };

  /// A utility for calling set_value with the result of a function invocation:
  template <class _Receiver, class _Fun, class... _As>
  STDEXEC_ATTRIBUTE(host, device)
  constexpr void __set_value_from(_Receiver &&__rcvr, _Fun &&__fun, _As &&...__as) noexcept {
    STDEXEC_TRY {
      if constexpr (__std::same_as<void, __invoke_result_t<_Fun, _As...>>) {
        __invoke(static_cast<_Fun &&>(__fun), static_cast<_As &&>(__as)...);
        STDEXEC::set_value(static_cast<_Receiver &&>(__rcvr));
      } else {
        STDEXEC::set_value(
          static_cast<_Receiver &&>(__rcvr),
          __invoke(static_cast<_Fun &&>(__fun), static_cast<_As &&>(__as)...));
      }
    }
    STDEXEC_CATCH_ALL {
      if constexpr (!__nothrow_invocable<_Fun, _As...>) {
        STDEXEC::set_error(static_cast<_Receiver &&>(__rcvr), std::current_exception());
      }
    }
  }

  template <class _Tag, class _Receiver>
  constexpr auto __mk_completion_fn(_Tag, _Receiver &__rcvr) noexcept {
    return [&]<class... _Args>(_Args &&...__args) noexcept {
      _Tag()(static_cast<_Receiver &&>(__rcvr), static_cast<_Args &&>(__args)...);
    };
  }

  template <class _Env>
  struct __receiver_archetype {
    using receiver_concept = receiver_t;

    template <class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_value(_Args &&...) noexcept {
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_error(_Error &&) noexcept {
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_stopped() noexcept {
    }

    STDEXEC_ATTRIBUTE(nodiscard, noreturn, host, device)
    _Env get_env() const noexcept {
      STDEXEC_ASSERT(false);
      STDEXEC_TERMINATE();
    }
  };
} // namespace STDEXEC
