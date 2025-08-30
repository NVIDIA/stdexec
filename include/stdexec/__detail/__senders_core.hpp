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
#include "__awaitable.hpp"
#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__domain.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  struct sender_t {
    using sender_concept = sender_t;
  };

  namespace __detail {
    template <class _Sender>
    concept __enable_sender = derived_from<typename _Sender::sender_concept, sender_t>
                           || requires { typename _Sender::is_sender; } // NOT TO SPEC back compat
                           || __awaitable<_Sender, __env::__promise<env<>>>;
  } // namespace __detail

  template <class _Sender>
  inline constexpr bool enable_sender = __detail::__enable_sender<_Sender>;

  template <class _Sender>
  concept sender = enable_sender<__decay_t<_Sender>> && environment_provider<__cref_t<_Sender>>
                && __detail::__consistent_completion_domains<_Sender>
                && move_constructible<__decay_t<_Sender>>
                && constructible_from<__decay_t<_Sender>, _Sender>;

  template <class _Sender, class... _Env>
  concept sender_in =
    (sizeof...(_Env) <= 1) && sender<_Sender> && requires(_Sender &&__sndr, _Env &&...__env) {
      {
        get_completion_signatures(static_cast<_Sender &&>(__sndr), static_cast<_Env &&>(__env)...)
      } -> __valid_completion_signatures;
    };

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Sender, class _Receiver>
  concept sender_to = receiver<_Receiver> && sender_in<_Sender, env_of_t<_Receiver>>
                   && __receiver_from<_Receiver, _Sender>
                   && requires(_Sender &&__sndr, _Receiver &&__rcvr) {
                        connect(static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr));
                      };

  template <class _Sender, class _Receiver>
  using connect_result_t = __call_result_t<connect_t, _Sender, _Receiver>;

  // Used to report a meaningful error message when the sender_in<Sndr, Env>
  // concept check fails.
  template <class _Sender, class... _Env>
  auto __diagnose_sender_concept_failure() {
    if constexpr (!enable_sender<__decay_t<_Sender>>) {
      static_assert(enable_sender<_Sender>, STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE);
    } else if constexpr (!__detail::__consistent_completion_domains<_Sender>) {
      static_assert(
        __detail::__consistent_completion_domains<_Sender>,
        "The completion schedulers of the sender do not have "
        "consistent domains. This is likely a "
        "bug in the sender implementation.");
    } else if constexpr (!move_constructible<__decay_t<_Sender>>) {
      static_assert(
        move_constructible<__decay_t<_Sender>>, "The sender type is not move-constructible.");
    } else if constexpr (!constructible_from<__decay_t<_Sender>, _Sender>) {
      static_assert(
        constructible_from<__decay_t<_Sender>, _Sender>,
        "The sender cannot be decay-copied. Did you forget a std::move?");
    } else {
      using _Completions = __completion_signatures_of_t<_Sender, _Env...>;
      if constexpr (__same_as<_Completions, __unrecognized_sender_error<_Sender, _Env...>>) {
        static_assert(__mnever<_Completions>, STDEXEC_ERROR_CANNOT_COMPUTE_COMPLETION_SIGNATURES);
      } else if constexpr (__merror<_Completions>) {
        static_assert(
          !__merror<_Completions>, STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_RETURNED_AN_ERROR);
      } else {
        static_assert(
          __valid_completion_signatures<_Completions>,
          STDEXEC_ERROR_GET_COMPLETION_SIGNATURES_HAS_INVALID_RETURN_TYPE);
      }
#if STDEXEC_MSVC() || STDEXEC_NVHPC()
      // MSVC and NVHPC need more encouragement to print the type of the
      // error.
      _Completions __what = 0;
#endif
    }
  }
} // namespace stdexec
