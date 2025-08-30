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

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  namespace __rcvrs {
    template <class _Receiver, class... _As>
    concept __set_value_member = requires(_Receiver &&__rcvr, _As &&...__args) {
      static_cast<_Receiver &&>(__rcvr).set_value(static_cast<_As &&>(__args)...);
    };

    struct set_value_t {
      template <class _Fn, class... _As>
      using __f = __minvoke<_Fn, _As...>;

      template <class _Receiver, class... _As>
        requires __set_value_member<_Receiver, _As...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      void operator()(_Receiver &&__rcvr, _As &&...__as) const noexcept {
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
        requires(!__set_value_member<_Receiver, _As...>)
             && tag_invocable<set_value_t, _Receiver, _As...>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      void operator()(_Receiver &&__rcvr, _As &&...__as) const noexcept {
        static_assert(nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) tag_invoke(*this, static_cast<_Receiver &&>(__rcvr), static_cast<_As &&>(__as)...);
      }
    };

    template <class _Receiver, class _Error>
    concept __set_error_member = requires(_Receiver &&__rcvr, _Error &&__err) {
      static_cast<_Receiver &&>(__rcvr).set_error(static_cast<_Error &&>(__err));
    };

    struct set_error_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 1)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class _Error>
        requires __set_error_member<_Receiver, _Error>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      void operator()(_Receiver &&__rcvr, _Error &&__err) const noexcept {
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
        requires(!__set_error_member<_Receiver, _Error>)
             && tag_invocable<set_error_t, _Receiver, _Error>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      void operator()(_Receiver &&__rcvr, _Error &&__err) const noexcept {
        static_assert(nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
        (void) tag_invoke(*this, static_cast<_Receiver &&>(__rcvr), static_cast<_Error &&>(__err));
      }
    };

    template <class _Receiver>
    concept __set_stopped_member = requires(_Receiver &&__rcvr) {
      static_cast<_Receiver &&>(__rcvr).set_stopped();
    };

    struct set_stopped_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 0)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver>
        requires __set_stopped_member<_Receiver>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      void operator()(_Receiver &&__rcvr) const noexcept {
        static_assert(
          noexcept(static_cast<_Receiver &&>(__rcvr).set_stopped()),
          "set_stopped member functions must be noexcept");
        static_assert(
          __same_as<decltype(static_cast<_Receiver &&>(__rcvr).set_stopped()), void>,
          "set_stopped member functions must return void");
        static_cast<_Receiver &&>(__rcvr).set_stopped();
      }

      template <class _Receiver>
        requires(!__set_stopped_member<_Receiver>) && tag_invocable<set_stopped_t, _Receiver>
      STDEXEC_ATTRIBUTE(host, device, always_inline)
      void operator()(_Receiver &&__rcvr) const noexcept {
        static_assert(nothrow_tag_invocable<set_stopped_t, _Receiver>);
        (void) tag_invoke(*this, static_cast<_Receiver &&>(__rcvr));
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
      (STDEXEC_WHEN(
        STDEXEC_EDG(),
        requires { typename _Receiver::receiver_concept; } &&)
         derived_from<typename _Receiver::receiver_concept, receiver_t>)
      || requires { typename _Receiver::is_receiver; } // back-compat, NOT TO SPEC
      || STDEXEC_IS_BASE_OF(receiver_t, _Receiver);    // NOT TO SPEC, for receiver_adaptor
  } // namespace __detail

  template <class _Receiver>
  inline constexpr bool enable_receiver = __detail::__enable_receiver<_Receiver>; // NOT TO SPEC

  template <class _Receiver>
  concept receiver = enable_receiver<__decay_t<_Receiver>>
                  && environment_provider<__cref_t<_Receiver>>
                  && move_constructible<__decay_t<_Receiver>>
                  && constructible_from<__decay_t<_Receiver>, _Receiver>;

  namespace __detail {
    template <class _Receiver, class _Tag, class... _Args>
    auto __try_completion(_Tag (*)(_Args...))
      -> __mexception<_MISSING_COMPLETION_SIGNAL_<_Tag(_Args...)>, _WITH_RECEIVER_<_Receiver>>;

    template <class _Receiver, class _Tag, class... _Args>
      requires __callable<_Tag, _Receiver, _Args...>
    auto __try_completion(_Tag (*)(_Args...)) -> __msuccess;

    template <class _Receiver, class... _Sigs>
    auto __try_completions(completion_signatures<_Sigs...> *) -> decltype((
      __msuccess(),
      ...,
      __detail::__try_completion<__decay_t<_Receiver>>(static_cast<_Sigs *>(nullptr))));
  } // namespace __detail

  template <class _Receiver, class _Completions>
  concept receiver_of = receiver<_Receiver> && requires(_Completions *__completions) {
    { __detail::__try_completions<_Receiver>(__completions) } -> __ok;
  };

  template <class _Receiver, class _Sender>
  concept __receiver_from =
    receiver_of<_Receiver, __completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;

  /// A utility for calling set_value with the result of a function invocation:
  template <class _Receiver, class _Fun, class... _As>
  STDEXEC_ATTRIBUTE(host, device)
  void __set_value_invoke(_Receiver &&__rcvr, _Fun &&__fun, _As &&...__as) noexcept {
    STDEXEC_TRY {
      if constexpr (same_as<void, __invoke_result_t<_Fun, _As...>>) {
        __invoke(static_cast<_Fun &&>(__fun), static_cast<_As &&>(__as)...);
        stdexec::set_value(static_cast<_Receiver &&>(__rcvr));
      } else {
        stdexec::set_value(
          static_cast<_Receiver &&>(__rcvr),
          __invoke(static_cast<_Fun &&>(__fun), static_cast<_As &&>(__as)...));
      }
    }
    STDEXEC_CATCH_ALL {
      if constexpr (!__nothrow_invocable<_Fun, _As...>) {
        stdexec::set_error(static_cast<_Receiver &&>(__rcvr), std::current_exception());
      }
    }
  }

  template <class _Tag, class _Receiver>
  auto __mk_completion_fn(_Tag, _Receiver &__rcvr) noexcept {
    return [&]<class... _Args>(_Args &&...__args) noexcept {
      _Tag()(static_cast<_Receiver &&>(__rcvr), static_cast<_Args &&>(__args)...);
    };
  }
} // namespace stdexec
