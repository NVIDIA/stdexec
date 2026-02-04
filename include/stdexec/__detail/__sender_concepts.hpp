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
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__get_completion_signatures.hpp"
#include "__meta.hpp"
#include "__receivers.hpp"
#include "__type_traits.hpp"

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  struct sender_t {
    // NOT TO SPEC:
    using sender_concept = sender_t;
  };

  namespace __detail {
    template <class _Sender>
    concept __enable_sender = __std::derived_from<typename _Sender::sender_concept, sender_t>
                           || requires { typename _Sender::is_sender; } // NOT TO SPEC back compat
                           || __awaitable<_Sender, __detail::__promise<env<>>>;
  } // namespace __detail

  template <class _Sender>
  inline constexpr bool enable_sender = __detail::__enable_sender<_Sender>;

  // [exec.snd.concepts]
  template <class _Sender>
  concept sender = enable_sender<__decay_t<_Sender>>       //
                && environment_provider<__cref_t<_Sender>> //
                && __std::move_constructible<__decay_t<_Sender>>
                && __std::constructible_from<__decay_t<_Sender>, _Sender>;

  template <auto _Completions>
  concept __constant_completion_signatures = __valid_completion_signatures<decltype(_Completions)>;

  template <class _Sender, class... _Env>
  concept sender_in =
    (sizeof...(_Env) <= 1) //
    && sender<_Sender>     //
    && __constant_completion_signatures<STDEXEC::get_completion_signatures<_Sender, _Env...>()>;

  template <class _Receiver, class _Sender>
  concept __receiver_from =
    receiver_of<_Receiver, __completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Sender, class _Receiver>
  concept __sender_to = receiver<_Receiver>                     //
                     && sender_in<_Sender, env_of_t<_Receiver>> //
                     && __receiver_from<_Receiver, _Sender>;

  template <class _Sender, class _Receiver>
  concept sender_to = __sender_to<_Sender, _Receiver> //
                   && requires(_Sender &&__sndr, _Receiver &&__rcvr) {
                        connect(static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr));
                      };

  template <class _Sender>
  concept dependent_sender = sender<_Sender> && __is_dependent_sender<_Sender>;

  template <class _Sender, class... _Env>
  using __single_sender_value_t = __value_types_t<
    __completion_signatures_of_t<_Sender, _Env...>,
    __qq<__msingle>,
    __qq<__msingle>
  >;

  template <class _Sender, class... _Env>
  using __single_value_variant_sender_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env...>, __qq<__mlist>, __qq<__msingle>>;

  template <class _Tag, class _Sender, class... _Env>
  concept __sends = sender_in<_Sender, _Env...> //
                 && __count_of<_Tag, _Sender, _Env...>::value != 0;

  template <class _Tag, class _Sender, class... _Env>
  concept __never_sends = sender_in<_Sender, _Env...> //
                       && __count_of<_Tag, _Sender, _Env...>::value == 0;

  template <class _Sender, class... _Env>
  concept __single_value_sender = sender_in<_Sender, _Env...> //
                               && requires { typename __single_sender_value_t<_Sender, _Env...>; };

  template <class _Sender, class... _Env>
  concept __single_value_variant_sender =
    sender_in<_Sender, _Env...> //
    && requires { typename __single_value_variant_sender_t<_Sender, _Env...>; };

  // Used to report a meaningful error message when the sender_in<Sndr, Env>
  // concept check fails.
  template <class _Sender, class... _Env>
  constexpr auto __diagnose_sender_concept_failure() noexcept {
    using __sndr_t = __remangle_t<_Sender>;
    if constexpr (!enable_sender<__decay_t<__sndr_t>>) {
      static_assert(enable_sender<__sndr_t>, STDEXEC_ERROR_ENABLE_SENDER_IS_FALSE);
    } else if constexpr (!__std::move_constructible<__decay_t<__sndr_t>>) {
      static_assert(
        __std::move_constructible<__decay_t<__sndr_t>>,
        "The sender type is not move-constructible.");
    } else if constexpr (!__std::constructible_from<__decay_t<__sndr_t>, __sndr_t>) {
      static_assert(
        __decay_copyable<__sndr_t>,
        "The sender cannot be decay-copied. Did you forget a std::move?");
    } else {
      using _Completions = __completion_signatures_of_t<__sndr_t, _Env...>;
      if constexpr (__same_as<_Completions, __unrecognized_sender_error_t<__sndr_t, _Env...>>) {
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
} // namespace STDEXEC
