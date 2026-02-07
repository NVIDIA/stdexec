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

#include "__completion_signatures.hpp"
#include "__concepts.hpp"
#include "__env.hpp" // IWYU pragma: keep for env<>
#include "__meta.hpp"
#include "__query.hpp"
#include "__sender_concepts.hpp"

#include <exception> // IWYU pragma: keep for std::terminate

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // Some utilities for debugging senders
  namespace __queries {
    struct __debug_env_t : __query<__debug_env_t> {
      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
    };
  } // namespace __queries

  namespace __debug {
    struct _COMPLETION_SIGNATURES_MISMATCH_ { };

    template <class _Sig>
    struct _COMPLETION_SIGNATURE_ { };

    template <class... _Sigs>
    struct _IS_NOT_ONE_OF_ { };

    template <class _Sender>
    struct _SIGNAL_SENT_BY_SENDER_ { };

    template <class _Warning>
    [[deprecated(
      "The sender claims to send a particular set of completions,"
      " but in actual fact it completes with a result that is not"
      " one of the declared completion signatures.")]] constexpr //
      STDEXEC_ATTRIBUTE(host, device)                            //
      void _ATTENTION_() noexcept {
    }

    template <class _Env>
    using __env_t = env<prop<__queries::__debug_env_t, bool>, _Env>;

    template <class _CvSender, class _Env, class... _Sigs>
    struct __receiver {
      using receiver_concept = receiver_t;

      template <class _Which>
      STDEXEC_ATTRIBUTE(host, device)
      static void __complete() noexcept {
        if constexpr (__none_of<_Which, _Sigs...>) {
          using __what_t = _WARNING_<
            _COMPLETION_SIGNATURES_MISMATCH_,
            _COMPLETION_SIGNATURE_<_Which>,
            _IS_NOT_ONE_OF_<_Sigs...>,
            _SIGNAL_SENT_BY_SENDER_<__demangle_t<_CvSender>>
          >;
          __debug::_ATTENTION_<__what_t>();
        }
        STDEXEC_TERMINATE();
      }

      template <class... _Args>
      STDEXEC_ATTRIBUTE(host, device)
      void set_value(_Args&&...) noexcept {
        __complete<set_value_t(_Args...)>();
      }

      template <class _Error>
      STDEXEC_ATTRIBUTE(host, device)
      void set_error(_Error&&) noexcept {
        __complete<set_error_t(_Error)>();
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr void set_stopped() noexcept {
        __complete<set_stopped_t()>();
      }

      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto get_env() const noexcept -> __env_t<_Env> {
        STDEXEC_TERMINATE();
      }
    };

    template <class _CvSender, class _Env, class _Sigs>
    using __receiver_t = __mapply<
      __mtransform<
        __mcompose<__q1<std::remove_pointer_t>, __q1<__cmplsigs::__normalize_sig_t>>,
        __mbind_front_q<__receiver, _CvSender, _Env>
      >,
      _Sigs
    >;

    struct __opstate {
      constexpr void start() & noexcept {
      }
    };
  } // namespace __debug

  ////////////////////////////////////////////////////////////////////////////
  // `__debug_sender`
  // ===============

  // Understanding why a particular sender doesn't connect to a particular
  // receiver is nigh impossible in the current design due to limitations in
  // how the compiler reports overload resolution failure in the presence of
  // constraints. `__debug_sender` is a utility to assist with the process. It
  // gives you the deep template instantiation backtrace that you need to
  // understand where in a chain of senders the problem is occurring.

  // ```c++
  // template <class _Sigs, class _Env = env<>, class _Sender>
  //   void __debug_sender(_Sender&& __sndr, _Env = {});

  // template <class _Env = env<>, class _Sender>
  //   void __debug_sender(_Sender&& __sndr, _Env = {});
  // ```

  // **Usage:**

  // To find out where in a chain of senders a sender is failing to connect
  // to a receiver, pass it to `__debug_sender`, optionally with an
  // environment argument; e.g. `__debug_sender(sndr [, env])`

  // To find out why a sender will not connect to a receiver of a particular
  // signature, specify the set of completion signatures as an explicit template
  // argument that names an instantiation of `completion_signatures`; e.g.:
  // `__debug_sender<completion_signatures<set_value_t(int)>>(sndr [, env])`.

  // **How it works:**

  // The `__debug_sender` function `connect`'s the sender to a
  // `__debug_receiver`, whose environment is augmented with a special
  // `__is_debug_env_t` query. An additional fall-back overload is added to
  // the `connect` CPO that recognizes receivers whose environments respond to
  // that query and lets them through. Then in a non-immediate context, it
  // looks for a `tag_invoke(connect_t...)` overload for the input sender and
  // receiver. This will recurse until it hits the `tag_invoke` call that is
  // causing the failure.

  // At least with clang, this gives me a nice backtrace, at the bottom of
  // which is the faulty `tag_invoke` overload with a mention of the
  // constraint that failed.
  template <class _Sigs, class _CvSender, class _Env = env<>>
  constexpr void __debug_sender(_CvSender&& __sndr, const _Env& = {}) {
    if constexpr (!__is_debug_env<_Env>) {
      if constexpr (sender_in<_CvSender, _Env>) {
        using __receiver_t = __debug::__receiver_t<_CvSender, _Env, _Sigs>;
        using __operation_t = connect_result_t<_CvSender, __receiver_t>;
        //static_assert(receiver_of<__receiver_t, _Sigs>);
        if constexpr (!__std::same_as<__operation_t, __debug::__opstate>) {
          if (__mnever<_CvSender>) {
            auto __op = connect(static_cast<_CvSender&&>(__sndr), __receiver_t{});
            STDEXEC::start(__op);
          }
        }
      } else {
        STDEXEC::__diagnose_sender_concept_failure<__demangle_t<_CvSender>, _Env>();
      }
    }
  }

  template <class _CvSender, class _Env = env<>>
  constexpr void __debug_sender(_CvSender&& __sndr, const _Env& __env = {}) {
    if constexpr (!__is_debug_env<_Env>) {
      if constexpr (sender_in<_CvSender, _Env>) {
        using __completions_t = __completion_signatures_of_t<_CvSender, __debug::__env_t<_Env>>;
        __debug_sender<__completions_t>(static_cast<_CvSender&&>(__sndr), __env);
      } else {
        STDEXEC::__diagnose_sender_concept_failure<__demangle_t<_CvSender>, _Env>();
      }
    }
  }
} // namespace STDEXEC
