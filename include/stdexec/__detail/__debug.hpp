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

#include "__concepts.hpp"
#include "__completion_signatures.hpp"
#include "__diagnostics.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__senders_core.hpp"

#include <exception> // IWYU pragma: keep for std::terminate

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // Some utilities for debugging senders
  namespace __debug {
    struct __is_debug_env_t {
      static constexpr auto query(forwarding_query_t) noexcept -> bool {
        return true;
      }
      template <class _Env>
        requires __env::__queryable<_Env, __is_debug_env_t>
      auto
        operator()(const _Env&) const noexcept -> __env::__query_result_t<_Env, __is_debug_env_t>;
    };

    template <class _Env>
    using __debug_env_t = env<prop<__is_debug_env_t, bool>, _Env>;

    template <class _Env>
    concept __is_debug_env = __callable<__is_debug_env_t, _Env>;

    struct __completion_signatures { };

#if STDEXEC_MSVC()
    // MSVCBUG https://developercommunity.visualstudio.com/t/Explicit-variable-template-specialisatio/10360032
    // MSVCBUG https://developercommunity.visualstudio.com/t/Non-function-type-interpreted-as-functio/10447831

    template <class _Sig>
    struct __normalize_sig;

    template <class _Tag, class... _Args>
    struct __normalize_sig<_Tag(_Args...)> {
      using __type = _Tag (*)(_Args&&...);
    };

    template <class _Sig>
    using __normalize_sig_t = typename __normalize_sig<_Sig>::__type;
#else
    template <class _Sig>
    extern int __normalize_sig;

    template <class _Tag, class... _Args>
    extern _Tag (*__normalize_sig<_Tag(_Args...)>)(_Args&&...);

    template <class _Sig>
    using __normalize_sig_t = decltype(__normalize_sig<_Sig>);
#endif

    template <class... _Sigs>
    struct __valid_completions {
      template <class... _Args>
        requires __one_of<set_value_t (*)(_Args&&...), _Sigs...>
      STDEXEC_ATTRIBUTE(host, device)
      void set_value(_Args&&...) noexcept {
        STDEXEC_TERMINATE();
      }

      template <class _Error>
        requires __one_of<set_error_t (*)(_Error&&), _Sigs...>
      STDEXEC_ATTRIBUTE(host, device)
      void set_error(_Error&&) noexcept {
        STDEXEC_TERMINATE();
      }

      STDEXEC_ATTRIBUTE(host, device)
      void set_stopped() noexcept
        requires __one_of<set_stopped_t (*)(), _Sigs...>
      {
        STDEXEC_TERMINATE();
      }
    };

    template <class _CvrefSenderId, class _Env, class _Completions>
    struct __debug_receiver {
      using __t = __debug_receiver;
      using __id = __debug_receiver;
      using receiver_concept = receiver_t;
    };

    template <class _CvrefSenderId, class _Env, class... _Sigs>
    struct __debug_receiver<_CvrefSenderId, _Env, completion_signatures<_Sigs...>>
      : __valid_completions<__normalize_sig_t<_Sigs>...> {
      using __t = __debug_receiver;
      using __id = __debug_receiver;
      using receiver_concept = receiver_t;

      STDEXEC_ATTRIBUTE(host, device) auto get_env() const noexcept -> __debug_env_t<_Env> {
        STDEXEC_TERMINATE();
      }
    };

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
      " one of the declared completion signatures.")]] STDEXEC_ATTRIBUTE(host, device) void _ATTENTION_() noexcept {
    }

    template <class _Sig>
    struct __invalid_completion {
      struct __t {
        template <class _CvrefSenderId, class _Env, class... _Sigs>
        // BUGBUG this works around a recently (aug 2023) introduced regression in nvc++
          requires(!__one_of<_Sig, _Sigs...>)
        __t(__debug_receiver<_CvrefSenderId, _Env, completion_signatures<_Sigs...>>&&) noexcept {
          using _SenderId = __decay_t<_CvrefSenderId>;
          using _Sender = stdexec::__t<_SenderId>;
          using _What = _WARNING_<
            _COMPLETION_SIGNATURES_MISMATCH_,
            _COMPLETION_SIGNATURE_<_Sig>,
            _IS_NOT_ONE_OF_<_Sigs...>,
            _SIGNAL_SENT_BY_SENDER_<__name_of<_Sender>>
          >;
          __debug::_ATTENTION_<_What>();
        }
      };
    };

    template <__completion_tag _Tag, class... _Args>
    STDEXEC_ATTRIBUTE(host, device)
    void tag_invoke(_Tag, __t<__invalid_completion<_Tag(_Args...)>>, _Args&&...) noexcept {
    }

    struct __debug_operation {
      void start() & noexcept {
      }
    };

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
    template <class _Sigs, class _Env = env<>, class _Sender>
    void __debug_sender(_Sender&& __sndr, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env>) {
        if constexpr (sender_in<_Sender, _Env>) {
          using _Receiver = __debug_receiver<__cvref_id<_Sender>, _Env, _Sigs>;
          using _Operation = connect_result_t<_Sender, _Receiver>;
          //static_assert(receiver_of<_Receiver, _Sigs>);
          if constexpr (!same_as<_Operation, __debug_operation>) {
            if (sizeof(_Sender) == ~0u) { // never true
              auto __op = connect(static_cast<_Sender&&>(__sndr), _Receiver{});
              stdexec::start(__op);
            }
          }
        } else {
          stdexec::__diagnose_sender_concept_failure<_Sender, _Env>();
        }
      }
    }

    template <class _Env = env<>, class _Sender>
    void __debug_sender(_Sender&& __sndr, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env>) {
        if constexpr (sender_in<_Sender, _Env>) {
          using _Sigs = __completion_signatures_of_t<_Sender, __debug_env_t<_Env>>;
          using _Receiver = __debug_receiver<__cvref_id<_Sender>, _Env, _Sigs>;
          if constexpr (!same_as<_Sigs, __debug::__completion_signatures>) {
            using _Operation = connect_result_t<_Sender, _Receiver>;
            //static_assert(receiver_of<_Receiver, _Sigs>);
            if constexpr (!same_as<_Operation, __debug_operation>) {
              if (sizeof(_Sender) == ~0ul) { // never true
                auto __op = connect(static_cast<_Sender&&>(__sndr), _Receiver{});
                stdexec::start(__op);
              }
            }
          }
        } else {
          __diagnose_sender_concept_failure<_Sender, _Env>();
        }
      }
    }
  } // namespace __debug

  using __debug::__is_debug_env;
  using __debug::__debug_sender;
} // namespace stdexec
