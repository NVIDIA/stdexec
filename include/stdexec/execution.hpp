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

#include <atomic>
#include <cassert>
#include <concepts>
#include <condition_variable>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <variant>

#include "__detail/__execution_fwd.hpp"

#include "__detail/__env.hpp"
#include "__detail/__domain.hpp"
#include "__detail/__intrusive_ptr.hpp"
#include "__detail/__meta.hpp"
#include "__detail/__scope.hpp"
#include "__detail/__basic_sender.hpp"
#include "functional.hpp"
#include "concepts.hpp"
#include "coroutine.hpp"
#include "stop_token.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wpragmas")
STDEXEC_PRAGMA_IGNORE_GNU("-Wunknown-warning-option")
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
    __decays_to< __call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to< __call_result_t<get_scheduler_t, _Env>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  namespace __receivers {
    struct set_value_t {
      template <class _Fn, class... _Args>
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class... _As>
        requires tag_invocable<set_value_t, _Receiver, _As...>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        void
        operator()(_Receiver&& __rcvr, _As&&... __as) const noexcept {
        static_assert(nothrow_tag_invocable<set_value_t, _Receiver, _As...>);
        (void) tag_invoke(set_value_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
      }
    };

    struct set_error_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 1)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver, class _Error>
        requires tag_invocable<set_error_t, _Receiver, _Error>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        void
        operator()(_Receiver&& __rcvr, _Error&& __err) const noexcept {
        static_assert(nothrow_tag_invocable<set_error_t, _Receiver, _Error>);
        (void) tag_invoke(set_error_t{}, (_Receiver&&) __rcvr, (_Error&&) __err);
      }
    };

    struct set_stopped_t {
      template <class _Fn, class... _Args>
        requires(sizeof...(_Args) == 0)
      using __f = __minvoke<_Fn, _Args...>;

      template <class _Receiver>
        requires tag_invocable<set_stopped_t, _Receiver>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        void
        operator()(_Receiver&& __rcvr) const noexcept {
        static_assert(nothrow_tag_invocable<set_stopped_t, _Receiver>);
        (void) tag_invoke(set_stopped_t{}, (_Receiver&&) __rcvr);
      }
    };
  } // namespace __receivers

  using __receivers::set_value_t;
  using __receivers::set_error_t;
  using __receivers::set_stopped_t;
  inline constexpr set_value_t set_value{};
  inline constexpr set_error_t set_error{};
  inline constexpr set_stopped_t set_stopped{};

  inline constexpr struct __try_call_t {
    template <class _Receiver, class _Fun, class... _Args>
      requires __callable<_Fun, _Args...>
    void operator()(_Receiver&& __rcvr, _Fun __fun, _Args&&... __args) const noexcept {
      if constexpr (__nothrow_callable<_Fun, _Args...>) {
        ((_Fun&&) __fun)((_Args&&) __args...);
      } else {
        try {
          ((_Fun&&) __fun)((_Args&&) __args...);
        } catch (...) {
          set_error((_Receiver&&) __rcvr, std::current_exception());
        }
      }
    }
  } __try_call{};

  namespace __error__ {
    inline constexpr __mstring __unrecognized_sender_type_diagnostic =
      "The given type cannot be used as a sender with the given environment "
      "because the attempt to compute the completion signatures failed."__csz;

    template <__mstring _Diagnostic = __unrecognized_sender_type_diagnostic>
    struct _UNRECOGNIZED_SENDER_TYPE_;

    template <class _Sender>
    struct _WITH_SENDER_;

    template <class... _Senders>
    struct _WITH_SENDERS_;

    template <class _Env>
    struct _WITH_ENVIRONMENT_;

    template <class _Ty>
    struct _WITH_TYPE_;

    template <class _Receiver>
    struct _WITH_RECEIVER_;

    template <class _Sig>
    struct _MISSING_COMPLETION_SIGNAL_;
  }

  using __error__::_UNRECOGNIZED_SENDER_TYPE_;
  using __error__::_WITH_ENVIRONMENT_;
  using __error__::_WITH_TYPE_;
  using __error__::_WITH_RECEIVER_;
  using __error__::_MISSING_COMPLETION_SIGNAL_;

  template <class _Sender>
  using _WITH_SENDER_ = __error__::_WITH_SENDER_<__name_of<_Sender>>;

  template <class... _Senders>
  using _WITH_SENDERS_ = __error__::_WITH_SENDERS_<__name_of<_Senders>...>;

  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __compl_sigs {
#if STDEXEC_NVHPC()
    template <class _Ty = __q<__types>, class... _Args>
    __types<__minvoke<_Ty, _Args...>> __test(set_value_t (*)(_Args...), set_value_t = {}, _Ty = {});
    template <class _Ty = __q<__types>, class _Error>
    __types<__minvoke<_Ty, _Error>> __test(set_error_t (*)(_Error), set_error_t = {}, _Ty = {});
    template <class _Ty = __q<__types>>
    __types<__minvoke<_Ty>> __test(set_stopped_t (*)(), set_stopped_t = {}, _Ty = {});
    __types<> __test(__ignore, __ignore, __ignore = {});

    template <class _Sig>
    inline constexpr bool __is_compl_sig = false;
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;

#else

    template <same_as<set_value_t> _Tag, class _Ty = __q<__types>, class... _Args>
    __types<__minvoke<_Ty, _Args...>> __test(_Tag (*)(_Args...));
    template <same_as<set_error_t> _Tag, class _Ty = __q<__types>, class _Error>
    __types<__minvoke<_Ty, _Error>> __test(_Tag (*)(_Error));
    template <same_as<set_stopped_t> _Tag, class _Ty = __q<__types>>
    __types<__minvoke<_Ty>> __test(_Tag (*)());
    template <class, class = void>
    __types<> __test(...);
    template <class _Tag, class _Ty = void, class... _Args>
    void __test(_Tag (*)(_Args...) noexcept) = delete;
#endif

#if STDEXEC_NVHPC()
    template <class _Sig>
    concept __completion_signature = __compl_sigs::__is_compl_sig<_Sig>;

    template <class _Sig, class _Tag, class _Ty = __q<__types>>
    using __signal_args_t = decltype(__compl_sigs::__test((_Sig*) nullptr, _Tag{}, _Ty{}));
#else
    template <class _Sig>
    concept __completion_signature = __typename<decltype(__compl_sigs::__test((_Sig*) nullptr))>;

    template <class _Sig, class _Tag, class _Ty = __q<__types>>
    using __signal_args_t = decltype(__compl_sigs::__test<_Tag, _Ty>((_Sig*) nullptr));
#endif
  } // namespace __compl_sigs

  using __compl_sigs::__completion_signature;

  template <__compl_sigs::__completion_signature... _Sigs>
  struct completion_signatures {
    // Uncomment this to see where completion_signatures is
    // erroneously getting instantiated:
    //static_assert(sizeof...(_Sigs) == -1u);
  };

  namespace __compl_sigs {
    template <class _TaggedTuple, __completion_tag _Tag, class... _Ts>
    auto __as_tagged_tuple_(_Tag (*)(_Ts...), _TaggedTuple*)
      -> __mconst<__minvoke<_TaggedTuple, _Tag, _Ts...>>;

    template <class _Sig, class _TaggedTuple>
    using __as_tagged_tuple =
      decltype(__compl_sigs::__as_tagged_tuple_((_Sig*) nullptr, (_TaggedTuple*) nullptr));

    template <class _TaggedTuple, class _Variant, class... _Sigs>
    auto __for_all_sigs_(completion_signatures<_Sigs...>*, _TaggedTuple*, _Variant*)
      -> __mconst< __minvoke< _Variant, __minvoke<__as_tagged_tuple<_Sigs, _TaggedTuple>>...>>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __for_all_sigs =                      //
      __minvoke<                                //
        decltype(__compl_sigs::__for_all_sigs_( //
          (_Completions*) nullptr,
          (_TaggedTuple*) nullptr,
          (_Variant*) nullptr))>;

    template <class _Completions, class _TaggedTuple, class _Variant>
    using __maybe_for_all_sigs = __meval<__for_all_sigs, _Completions, _TaggedTuple, _Variant>;
  } // namespace __compl_sigs

  template <class _Completions>
  concept __valid_completion_signatures = //
    __ok<_Completions> && __is_instance_of<_Completions, completion_signatures>;

  template <class _Completions>
  using __invalid_completion_signatures_t = //
    __mbool<!__valid_completion_signatures<_Completions>>;

  template <__mstring _Msg = "Expected an instance of template completion_signatures<>"__csz>
  struct _INVALID_COMPLETION_SIGNATURES_TYPE_ {
    template <class... _Completions>
    using __f = //
      __mexception<
        _INVALID_COMPLETION_SIGNATURES_TYPE_<>,
        __minvoke<
          __mfind_if<
            __q<__invalid_completion_signatures_t>,
            __mcompose<__q<_WITH_TYPE_>, __q<__mfront>>>,
          _Completions...>>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_impl_t = //
    __minvoke<
      __if_c<
        (__valid_completion_signatures<_Completions> && ...),
        __mconcat<__munique<__q<completion_signatures>>>,
        _INVALID_COMPLETION_SIGNATURES_TYPE_<>>,
      _Completions...>;

  template <class... _Completions>
  struct __concat_completion_signatures_ {
    using __t = __meval<__concat_completion_signatures_impl_t, _Completions...>;
  };

  template <class... _Completions>
  using __concat_completion_signatures_t = __t<__concat_completion_signatures_<_Completions...>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template <class _Receiver, class _Tag, class... _Args>
  auto __try_completion(_Tag (*)(_Args...))
    -> __mexception<_MISSING_COMPLETION_SIGNAL_<_Tag(_Args...)>, _WITH_RECEIVER_<_Receiver>>;

  template <class _Receiver, class _Tag, class... _Args>
    requires nothrow_tag_invocable<_Tag, _Receiver, _Args...>
  __msuccess __try_completion(_Tag (*)(_Args...));

  template <class _Receiver, class... _Sigs>
  auto __try_completions(completion_signatures<_Sigs...>*)
    -> decltype((__msuccess(), ..., stdexec::__try_completion<_Receiver>((_Sigs*) nullptr)));

  template <class _Sender, class _Env>
  using __unrecognized_sender_error = //
    __mexception< _UNRECOGNIZED_SENDER_TYPE_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>>;

  template <class _Sender, class _Env>
  using __completion_signatures_of_t = __call_result_t<get_completion_signatures_t, _Sender, _Env>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  struct receiver_t {
    using receiver_concept = receiver_t; // NOT TO SPEC
  };

  namespace __detail {
    template <class _Receiver>
    concept __enable_receiver =                                            //
      (STDEXEC_NVHPC(requires { typename _Receiver::receiver_concept; }&&) //
       derived_from<typename _Receiver::receiver_concept, receiver_t>)
      || requires { typename _Receiver::is_receiver; } // back-compat, NOT TO SPEC
      || STDEXEC_IS_BASE_OF(receiver_t, _Receiver);    // NOT TO SPEC, for receiver_adaptor
  }                                                    // namespace __detail

  template <class _Receiver>
  inline constexpr bool enable_receiver = __detail::__enable_receiver<_Receiver>; // NOT TO SPEC

  template <class _Receiver>
  concept receiver =
    enable_receiver<__decay_t<_Receiver>> &&     //
    environment_provider<__cref_t<_Receiver>> && //
    move_constructible<__decay_t<_Receiver>> &&  //
    constructible_from<__decay_t<_Receiver>, _Receiver>;

  template <class _Receiver, class _Completions>
  concept receiver_of =    //
    receiver<_Receiver> && //
    requires(_Completions* __completions) {
      { stdexec::__try_completions<__decay_t<_Receiver>>(__completions) } -> __ok;
    };

  template <class _Receiver, class _Sender>
  concept __receiver_from =
    receiver_of< _Receiver, __completion_signatures_of_t<_Sender, env_of_t<_Receiver>>>;

  /////////////////////////////////////////////////////////////////////////////
  // Some utilities for debugging senders
  namespace __debug {
    struct __is_debug_env_t {
      friend constexpr bool tag_invoke(forwarding_query_t, const __is_debug_env_t&) noexcept {
        return true;
      }
      template <class _Env>
        requires tag_invocable<__is_debug_env_t, const _Env&>
      auto operator()(const _Env&) const noexcept
        -> tag_invoke_result_t<__is_debug_env_t, const _Env&>;
    };
    template <class _Env>
    using __debug_env_t = __make_env_t<_Env, __with<__is_debug_env_t, bool>>;

    template <class _Env>
    concept __is_debug_env = tag_invocable<__debug::__is_debug_env_t, _Env>;

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
      template <derived_from<__valid_completions> _Self, class _Tag, class... _Args>
        requires __one_of<_Tag (*)(_Args&&...), _Sigs...>
      STDEXEC_ATTRIBUTE((host, device)) friend void tag_invoke(_Tag, _Self&&, _Args&&...) noexcept {
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
    struct __debug_receiver<_CvrefSenderId, _Env, completion_signatures<_Sigs...>> //
      : __valid_completions<__normalize_sig_t<_Sigs>...> {
      using __t = __debug_receiver;
      using __id = __debug_receiver;
      using receiver_concept = receiver_t;

      template <same_as<get_env_t> _Tag>
      STDEXEC_ATTRIBUTE((host, device))
      friend __debug_env_t<_Env> tag_invoke(_Tag, __debug_receiver) noexcept {
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
      " one of the declared completion signatures.")]] STDEXEC_ATTRIBUTE((host, device)) void _ATTENTION_() noexcept {
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
          using _What = //
            _WARNING_<  //
              _COMPLETION_SIGNATURES_MISMATCH_,
              _COMPLETION_SIGNATURE_<_Sig>,
              _IS_NOT_ONE_OF_<_Sigs...>,
              _SIGNAL_SENT_BY_SENDER_<__name_of<_Sender>>>;
          __debug::_ATTENTION_<_What>();
        }
      };
    };

    template <__completion_tag _Tag, class... _Args>
    STDEXEC_ATTRIBUTE((host, device))
    void tag_invoke(_Tag, __t<__invalid_completion<_Tag(_Args...)>>, _Args&&...) noexcept {
    }

    struct __debug_operation {
      template <same_as<start_t> _Tag>
      friend void tag_invoke(_Tag, __debug_operation&) noexcept {
      }
    };

    ////////////////////////////////////////////////////////////////////////////
    // `__debug_sender`
    // ===============
    //
    // Understanding why a particular sender doesn't connect to a particular
    // receiver is nigh impossible in the current design due to limitations in
    // how the compiler reports overload resolution failure in the presence of
    // constraints. `__debug_sender` is a utility to assist with the process. It
    // gives you the deep template instantiation backtrace that you need to
    // understand where in a chain of senders the problem is occurring.
    //
    // ```c++
    // template <class _Sigs, class _Env = empty_env, class _Sender>
    //   void __debug_sender(_Sender&& __sndr, _Env = {});
    //
    // template <class _Env = empty_env, class _Sender>
    //   void __debug_sender(_Sender&& __sndr, _Env = {});
    // ```
    //
    // **Usage:**
    //
    // To find out where in a chain of senders a sender is failing to connect
    // to a receiver, pass it to `__debug_sender`, optionally with an
    // environment argument; e.g. `__debug_sender(sndr [, env])`
    //
    // To find out why a sender will not connect to a receiver of a particular
    // signature, specify the set of completion signatures as an explicit template
    // argument that names an instantiation of `completion_signatures`; e.g.:
    // `__debug_sender<completion_signatures<set_value_t(int)>>(sndr [, env])`.
    //
    // **How it works:**
    //
    // The `__debug_sender` function `connect`'s the sender to a
    // `__debug_receiver`, whose environment is augmented with a special
    // `__is_debug_env_t` query. An additional fall-back overload is added to
    // the `connect` CPO that recognizes receivers whose environments respond to
    // that query and lets them through. Then in a non-immediate context, it
    // looks for a `tag_invoke(connect_t...)` overload for the input sender and
    // receiver. This will recurse until it hits the `tag_invoke` call that is
    // causing the failure.
    //
    // At least with clang, this gives me a nice backtrace, at the bottom of
    // which is the faulty `tag_invoke` overload with a mention of the
    // constraint that failed.
    template <class _Sigs, class _Env = empty_env, class _Sender>
    void __debug_sender(_Sender&& __sndr, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env>) {
        if (sizeof(_Sender) == ~0) { // never true
          using _Receiver = __debug_receiver<__cvref_id<_Sender>, _Env, _Sigs>;
          using _Operation = connect_result_t<_Sender, _Receiver>;
          //static_assert(receiver_of<_Receiver, _Sigs>);
          if constexpr (!same_as<_Operation, __debug_operation>) {
            auto __op = connect((_Sender&&) __sndr, _Receiver{});
            start(__op);
          }
        }
      }
    }

    template <class _Env = empty_env, class _Sender>
    void __debug_sender(_Sender&& __sndr, const _Env& = {}) {
      if constexpr (!__is_debug_env<_Env>) {
        if (sizeof(_Sender) == ~0) { // never true
          using _Sigs = __completion_signatures_of_t<_Sender, __debug_env_t<_Env>>;
          if constexpr (!same_as<_Sigs, __debug::__completion_signatures>) {
            using _Receiver = __debug_receiver<__cvref_id<_Sender>, _Env, _Sigs>;
            using _Operation = connect_result_t<_Sender, _Receiver>;
            //static_assert(receiver_of<_Receiver, _Sigs>);
            if constexpr (!same_as<_Operation, __debug_operation>) {
              auto __op = connect((_Sender&&) __sndr, _Receiver{});
              start(__op);
            }
          }
        }
      }
    }
  } // namespace __debug

  using __debug::__is_debug_env;
  using __debug::__debug_sender;

  /////////////////////////////////////////////////////////////////////////////
  // dependent_domain
  struct dependent_domain {
    template <sender_expr _Sender, class _Env>
      requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
    STDEXEC_ATTRIBUTE((always_inline)) decltype(auto)
      transform_sender(_Sender&& __sndr, const _Env& __env) const;
  };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  namespace __domain {
    struct __transform_env {
      template <class _Domain, class _Sender, class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      /*constexpr*/ decltype(auto)
        operator()(_Domain __dom, _Sender&& __sndr, _Env&& __env) const noexcept {
        if constexpr (__domain::__has_transform_env<_Domain, _Sender, _Env>) {
          return __dom.transform_env((_Sender&&) __sndr, (_Env&&) __env);
        } else {
          return default_domain().transform_env((_Sender&&) __sndr, (_Env&&) __env);
        }
      }
    };

    struct __transform_sender_1 {
      template <class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      /*constexpr*/ decltype(auto)
        operator()(_Domain __dom, _Sender&& __sndr, const _Env&... __env) const {
        if constexpr (__domain::__has_transform_sender<_Domain, _Sender, _Env...>) {
          return __dom.transform_sender((_Sender&&) __sndr, __env...);
        } else {
          return default_domain().transform_sender((_Sender&&) __sndr, __env...);
        }
      }
    };

    template <class _Ty, class _Uy>
    concept __decay_same_as = same_as<__decay_t<_Ty>, __decay_t<_Uy>>;

    struct __transform_sender {
      template <class _Self = __transform_sender, class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      /*constexpr*/ decltype(auto)
        operator()(_Domain __dom, _Sender&& __sndr, const _Env&... __env) const {
        using _Sender2 = __call_result_t<__transform_sender_1, _Domain, _Sender, const _Env&...>;
        // If the transformation doesn't change the sender's type, then do not
        // apply the transform recursively.
        if constexpr (__decay_same_as<_Sender, _Sender2>) {
          return __transform_sender_1()(__dom, (_Sender&&) __sndr, __env...);
        } else {
          // We transformed the sender and got back a different sender. Transform that one too.
          return _Self()(
            __dom, __transform_sender_1()(__dom, (_Sender&&) __sndr, __env...), __env...);
        }
      }
    };

    struct __transform_dependent_sender {
      // If we are doing a lazy customization of a type whose domain is value-dependent (e.g.,
      // let_value), first transform the sender to determine the domain. Then continue transforming
      // the sender with the requested domain.
      template <class _Domain, sender_expr _Sender, class _Env>
        requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
      /*constexpr*/ decltype(auto)
        operator()(_Domain __dom, _Sender&& __sndr, const _Env& __env) const {
        static_assert(__none_of<_Domain, dependent_domain>);
        return __transform_sender()(
          __dom, dependent_domain().transform_sender((_Sender&&) __sndr, __env), __env);
      }
    };
  } // namespace __domain

  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  inline constexpr struct transform_sender_t
    : __domain::__transform_sender
    , __domain::__transform_dependent_sender {
    using __domain::__transform_sender::operator();
    using __domain::__transform_dependent_sender::operator();
  } transform_sender{};

  template <class _Domain, class _Sender, class... _Env>
  using transform_sender_result_t = __call_result_t<transform_sender_t, _Domain, _Sender, _Env...>;

  inline constexpr __domain::__transform_env transform_env{};

  struct _CHILD_SENDERS_WITH_DIFFERENT_DOMAINS_ { };

  template <sender_expr _Sender, class _Env>
    requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
  decltype(auto) dependent_domain::transform_sender(_Sender&& __sndr, const _Env& __env) const {
    // apply any algorithm-specific transformation to the environment
    const auto& __env2 = transform_env(*this, (_Sender&&) __sndr, __env);

    // recursively transform the sender to determine the domain
    return __sexpr_apply(
      (_Sender&&) __sndr,
      [&]<class _Tag, class _Data, class... _Childs>(_Tag, _Data&& __data, _Childs&&... __childs) {
        // TODO: propagate meta-exceptions here:
        auto __sndr2 = __make_sexpr<_Tag>(
          (_Data&&) __data, __domain::__transform_sender()(*this, (_Childs&&) __childs, __env2)...);
        using _Sender2 = decltype(__sndr2);

        auto __domain2 = __sexpr_apply(__sndr2, __domain::__common_domain_fn());
        using _Domain2 = decltype(__domain2);

        if constexpr (same_as<_Domain2, __none_such>) {
          return __mexception<_CHILD_SENDERS_WITH_DIFFERENT_DOMAINS_, _WITH_SENDER_<_Sender2>>();
        } else {
          return __domain::__transform_sender()(__domain2, std::move(__sndr2), __env);
        }
        STDEXEC_UNREACHABLE();
      });
  }

  // A helper for use when building sender trees where each node must be transformed.
  template <class _Domain, class _Env>
  auto __make_transformer(_Domain, const _Env& __env) {
    return [&]<class _Tag>(_Tag) {
      return [&]<class... _Args>(_Args&&... __args) -> decltype(auto) {
        return stdexec::transform_sender(_Domain(), _Tag()((_Args&&) __args...), __env);
      };
    };
  }

  /////////////////////////////////////////////////////////////////////////////
  template <class _Tag, class _Domain, class _Sender, class... _Args>
  concept __has_implementation_for =
    __domain::__has_apply_sender<_Domain, _Tag, _Sender, _Args...>
    || __domain::__has_apply_sender<default_domain, _Tag, _Sender, _Args...>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.apply_sender]
  inline constexpr struct apply_sender_t {
    template <class _Domain, class _Tag, class _Sender, class... _Args>
      requires __has_implementation_for<_Tag, _Domain, _Sender, _Args...>
    STDEXEC_ATTRIBUTE((always_inline))
      /*constexpr*/ decltype(auto)
        operator()(_Domain __dom, _Tag, _Sender&& __sndr, _Args&&... __args) const {
      if constexpr (__domain::__has_apply_sender<_Domain, _Tag, _Sender, _Args...>) {
        return __dom.apply_sender(_Tag(), (_Sender&&) __sndr, (_Args&&) __args...);
      } else {
        return default_domain().apply_sender(_Tag(), (_Sender&&) __sndr, (_Args&&) __args...);
      }
    }
  } apply_sender{};

  template <class _Domain, class _Tag, class _Sender, class... _Args>
  using apply_sender_result_t = __call_result_t<apply_sender_t, _Domain, _Tag, _Sender, _Args...>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.sndtraits]
  namespace __get_completion_signatures {
    template <class _Sender, class _Env>
    using __tfx_sender =
      transform_sender_result_t<__late_domain_of_t<_Sender, _Env>, _Sender, _Env>;

    template <class _Sender, class _Env>
    concept __with_tag_invoke = //
      tag_invocable<get_completion_signatures_t, __tfx_sender<_Sender, _Env>, _Env>;

    template <class _Sender, class _Env>
    using __member_alias_t = //
      typename __decay_t<__tfx_sender<_Sender, _Env>>::completion_signatures;

    template <class _Sender, class _Env = empty_env>
    concept __with_member_alias = __mvalid<__member_alias_t, _Sender, _Env>;

    struct get_completion_signatures_t {
      template <class _Sender, class _Env>
      static auto __impl() {
        static_assert(sizeof(_Sender), "Incomplete type used with get_completion_signatures");
        static_assert(sizeof(_Env), "Incomplete type used with get_completion_signatures");

        if constexpr (__with_tag_invoke<_Sender, _Env>) {
          using _TfxSender = __tfx_sender<_Sender, _Env>;
          using _Result = tag_invoke_result_t<get_completion_signatures_t, _TfxSender, _Env>;
          return (_Result(*)()) nullptr;
        } else if constexpr (__with_member_alias<_Sender, _Env>) {
          using _Result = __member_alias_t<_Sender, _Env>;
          return (_Result(*)()) nullptr;
        } else if constexpr (__awaitable<_Sender, __env_promise<_Env>>) {
          using _Result = __await_result_t<_Sender, __env_promise<_Env>>;
          return (completion_signatures<
                  // set_value_t() or set_value_t(T)
                  __minvoke<__remove<void, __qf<set_value_t>>, _Result>,
                  set_error_t(std::exception_ptr),
                  set_stopped_t()>(*)()) nullptr;
        } else if constexpr (__is_debug_env<_Env>) {
          using __tag_invoke::tag_invoke;
          // This ought to cause a hard error that indicates where the problem is.
          using _Completions
            [[maybe_unused]] = tag_invoke_result_t<get_completion_signatures_t, _Sender, _Env>;
          return (__debug::__completion_signatures(*)()) nullptr;
        } else {
          using _Result = __mexception<
            _UNRECOGNIZED_SENDER_TYPE_<>,
            _WITH_SENDER_<_Sender>,
            _WITH_ENVIRONMENT_<_Env>>;
          return (_Result(*)()) nullptr;
        }
      }

      // NOT TO SPEC: if we're unable to compute the completion signatures,
      // return an error type instead of SFINAE.
      template <class _Sender, class _Env = __default_env>
      constexpr auto operator()(_Sender&&, const _Env&) const noexcept
        -> decltype(__impl<_Sender, _Env>()()) {
        return {};
      }
    };
  } // namespace __get_completion_signatures

  using __get_completion_signatures::get_completion_signatures_t;
  inline constexpr get_completion_signatures_t get_completion_signatures{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  struct sender_t {
    using sender_concept = sender_t;
  };

  namespace __detail {
    template <class _Sender>
    concept __enable_sender = //
      derived_from<typename _Sender::sender_concept, sender_t>
      || requires { typename _Sender::is_sender; } // NOT TO SPEC back compat
      || __awaitable<_Sender, __env_promise<empty_env>>;
  } // namespace __detail

  template <class _Sender>
  inline constexpr bool enable_sender = __detail::__enable_sender<_Sender>;

  template <class _Sender, class _Env = empty_env>
  concept sender =
    enable_sender<__decay_t<_Sender>> &&                  //
    environment_provider<__cref_t<_Sender>> &&            //
    __detail::__consistent_completion_domains<_Sender> && //
    move_constructible<__decay_t<_Sender>> &&             //
    constructible_from<__decay_t<_Sender>, _Sender>;

  template <class _Sender, class _Env = empty_env>
  concept sender_in =  //
    sender<_Sender> && //
    requires(_Sender&& __sndr, _Env&& __env) {
      {
        get_completion_signatures((_Sender&&) __sndr, (_Env&&) __env)
      } -> __valid_completion_signatures;
    };

#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
  // __checked_completion_signatures is for catching logic bugs in a typed
  // sender's metadata. If sender<S> and sender_in<S, Ctx> are both true, then they
  // had better report the same metadata. This completion signatures wrapper
  // enforces that at compile time.
  template <class _Sender, class _Env>
  auto __checked_completion_signatures(_Sender&& __sndr, const _Env& __env) noexcept {
    using __completions_t = __completion_signatures_of_t<_Sender, _Env>;
    stdexec::__debug_sender<__completions_t>((_Sender&&) __sndr, __env);
    return __completions_t{};
  }

  template <class _Sender, class _Env = empty_env>
    requires sender_in<_Sender, _Env>
  using completion_signatures_of_t =
    decltype(stdexec::__checked_completion_signatures(__declval<_Sender>(), __declval<_Env>()));
#else
  template <class _Sender, class _Env = empty_env>
    requires sender_in<_Sender, _Env>
  using completion_signatures_of_t = __completion_signatures_of_t<_Sender, _Env>;
#endif

  struct __not_a_variant {
    __not_a_variant() = delete;
  };
  template <class... _Ts>
  using __variant = //
    __minvoke<
      __if_c<
        sizeof...(_Ts) != 0,
        __transform<__q<__decay_t>, __munique<__q<std::variant>>>,
        __mconst<__not_a_variant>>,
      _Ts...>;

  using __nullable_variant_t = __munique<__mbind_front_q<std::variant, std::monostate>>;

  template <class... _Ts>
  using __decayed_tuple = __meval<std::tuple, __decay_t<_Ts>...>;

  template <class _Tag, class _Tuple>
  struct __select_completions_for {
    template <same_as<_Tag> _Tag2, class... _Args>
    using __f = __minvoke<_Tag2, _Tuple, _Args...>;
  };

  template <class _Tuple>
  struct __invoke_completions {
    template <class _Tag, class... _Args>
    using __f = __minvoke<_Tag, _Tuple, _Args...>;
  };

  template <class _Tag, class _Tuple>
  using __select_completions_for_or = //
    __with_default< __select_completions_for<_Tag, _Tuple>, __>;

  template <class _Tag, class _Completions>
  using __only_gather_signal = //
    __compl_sigs::__maybe_for_all_sigs<
      _Completions,
      __select_completions_for_or<_Tag, __qf<_Tag>>,
      __remove<__, __q<completion_signatures>>>;

  template <class _Tag, class _Completions, class _Tuple, class _Variant>
  using __gather_signal = //
    __compl_sigs::__maybe_for_all_sigs<
      __only_gather_signal<_Tag, _Completions>,
      __invoke_completions<_Tuple>,
      _Variant>;

  template <class _Tag, class _Sender, class _Env, class _Tuple, class _Variant>
  using __gather_completions_for = //
    __meval<                       //
      __gather_signal,
      _Tag,
      __completion_signatures_of_t<_Sender, _Env>,
      _Tuple,
      _Variant>;

  template <                             //
    class _Sender,                       //
    class _Env = __default_env,          //
    class _Tuple = __q<__decayed_tuple>, //
    class _Variant = __q<__variant>>
  using __try_value_types_of_t = //
    __gather_completions_for<set_value_t, _Sender, _Env, _Tuple, _Variant>;

  template <                             //
    class _Sender,                       //
    class _Env = __default_env,          //
    class _Tuple = __q<__decayed_tuple>, //
    class _Variant = __q<__variant>>
    requires sender_in<_Sender, _Env>
  using __value_types_of_t = //
    __msuccess_or_t<__try_value_types_of_t<_Sender, _Env, _Tuple, _Variant>>;

  template <class _Sender, class _Env = __default_env, class _Variant = __q<__variant>>
  using __try_error_types_of_t =
    __gather_completions_for<set_error_t, _Sender, _Env, __q<__midentity>, _Variant>;

  template <class _Sender, class _Env = __default_env, class _Variant = __q<__variant>>
    requires sender_in<_Sender, _Env>
  using __error_types_of_t = __msuccess_or_t<__try_error_types_of_t<_Sender, _Env, _Variant>>;

  template <                                            //
    class _Sender,                                      //
    class _Env = __default_env,                         //
    template <class...> class _Tuple = __decayed_tuple, //
    template <class...> class _Variant = __variant>
    requires sender_in<_Sender, _Env>
  using value_types_of_t = __value_types_of_t<_Sender, _Env, __q<_Tuple>, __q<_Variant>>;

  template <class _Sender, class _Env = __default_env, template <class...> class _Variant = __variant>
    requires sender_in<_Sender, _Env>
  using error_types_of_t = __error_types_of_t<_Sender, _Env, __q<_Variant>>;

  template <class _Tag, class _Sender, class _Env = __default_env>
  using __try_count_of = //
    __compl_sigs::__maybe_for_all_sigs<
      __completion_signatures_of_t<_Sender, _Env>,
      __q<__mfront>,
      __mcount<_Tag>>;

  template <class _Tag, class _Sender, class _Env = __default_env>
    requires sender_in<_Sender, _Env>
  using __count_of = __msuccess_or_t<__try_count_of<_Tag, _Sender, _Env>>;

  template <class _Tag, class _Sender, class _Env = __default_env>
    requires __mvalid<__count_of, _Tag, _Sender, _Env>
  inline constexpr bool __sends = (__v<__count_of<_Tag, _Sender, _Env>> != 0);

  template <class _Sender, class _Env = __default_env>
    requires __mvalid<__count_of, set_stopped_t, _Sender, _Env>
  inline constexpr bool sends_stopped = __sends<set_stopped_t, _Sender, _Env>;

  template <class _Sender, class _Env = __default_env>
  using __single_sender_value_t =
    __value_types_of_t<_Sender, _Env, __msingle_or<void>, __q<__msingle>>;

  template <class _Sender, class _Env = __default_env>
  using __single_value_variant_sender_t = value_types_of_t<_Sender, _Env, __types, __msingle>;

  template <class _Sender, class _Env = __default_env>
  concept __single_typed_sender =
    sender_in<_Sender, _Env> && __mvalid<__single_sender_value_t, _Sender, _Env>;

  template <class _Sender, class _Env = __default_env>
  concept __single_value_variant_sender =
    sender_in<_Sender, _Env> && __mvalid<__single_value_variant_sender_t, _Sender, _Env>;

  template <class... Errs>
  using __nofail = __mbool<sizeof...(Errs) == 0>;

  template <class _Sender, class _Env = __default_env>
  concept __nofail_sender =
    sender_in<_Sender, _Env> && (__v<error_types_of_t<_Sender, _Env, __nofail>>);

  /////////////////////////////////////////////////////////////////////////////
  namespace __compl_sigs {
    template <class... _Args>
    using __default_set_value = completion_signatures<set_value_t(_Args...)>;

    template <class _Error>
    using __default_set_error = completion_signatures<set_error_t(_Error)>;

    template <__valid_completion_signatures... _Sigs>
    using __ensure_concat_ = __minvoke<__mconcat<__q<completion_signatures>>, _Sigs...>;

    template <class... _Sigs>
    using __ensure_concat = __mtry_eval<__ensure_concat_, _Sigs...>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __compl_sigs_impl = //
      __concat_completion_signatures_t<
        _Sigs,
        __mtry_eval<__try_value_types_of_t, _Sender, _Env, _SetVal, __q<__ensure_concat>>,
        __mtry_eval<__try_error_types_of_t, _Sender, _Env, __transform<_SetErr, __q<__ensure_concat>>>,
        __if<__try_count_of<set_stopped_t, _Sender, _Env>, _SetStp, completion_signatures<>>>;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
      requires __mvalid<__compl_sigs_impl, _Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>
    extern __compl_sigs_impl<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp> __compl_sigs_v;

    template <class _Sender, class _Env, class _Sigs, class _SetVal, class _SetErr, class _SetStp>
    using __compl_sigs_t =
      decltype(__compl_sigs_v<_Sender, _Env, _Sigs, _SetVal, _SetErr, _SetStp>);

    template <                                                    //
      class _Sender,                                              //
      class _Env = __default_env,                                 //
      class _Sigs = completion_signatures<>,                      //
      class _SetValue = __q<__default_set_value>,                 //
      class _SetError = __q<__default_set_error>,                 //
      class _SetStopped = completion_signatures<set_stopped_t()>> //
    using __try_make_completion_signatures =                      //
      __meval< __compl_sigs_t, _Sender, _Env, _Sigs, _SetValue, _SetError, _SetStopped>;
  } // namespace __compl_sigs

  using __compl_sigs::__try_make_completion_signatures;

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC
  //
  // make_completion_signatures
  // ==========================
  //
  // `make_completion_signatures` takes a sender, and environment, and a bunch
  // of other template arguments for munging the completion signatures of a
  // sender in interesting ways.
  //
  //  ```c++
  //  template <class... Args>
  //    using __default_set_value = completion_signatures<set_value_t(Args...)>;
  //
  //  template <class Err>
  //    using __default_set_error = completion_signatures<set_error_t(Err)>;
  //
  //  template <
  //    sender Sndr,
  //    class Env = __default_env,
  //    class AddlSigs = completion_signatures<>,
  //    template <class...> class SetValue = __default_set_value,
  //    template <class> class SetError = __default_set_error,
  //    class SetStopped = completion_signatures<set_stopped_t()>>
  //      requires sender_in<Sndr, Env>
  //  using make_completion_signatures =
  //    completion_signatures< ... >;
  //  ```
  //
  //  * `SetValue` : an alias template that accepts a set of value types and
  //    returns an instance of `completion_signatures`.
  //  * `SetError` : an alias template that accepts an error types and returns a
  //    an instance of `completion_signatures`.
  //  * `SetStopped` : an instantiation of `completion_signatures` with a list
  //    of completion signatures `Sigs...` to the added to the list if the
  //    sender can complete with a stopped signal.
  //  * `AddlSigs` : an instantiation of `completion_signatures` with a list of
  //    completion signatures `Sigs...` to the added to the list
  //    unconditionally.
  //
  //  `make_completion_signatures` does the following:
  //  * Let `VCs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `value_types_of_t<Sndr, Env, SetValue,
  //    __typelist>`, and let `Vs...` be the concatenation of the packs that are
  //    template arguments to each `completion_signature` in `VCs...`.
  //  * Let `ECs...` be a pack of the `completion_signatures` types in the
  //    `__typelist` named by `error_types_of_t<Sndr, Env, __errorlist>`, where
  //    `__errorlist` is an alias template such that `__errorlist<Ts...>` names
  //    `__typelist<SetError<Ts>...>`, and let `Es...` by the concatenation of
  //    the packs that are the template arguments to each `completion_signature`
  //    in `ECs...`.
  //  * Let `Ss...` be an empty pack if `sends_stopped<Sndr, Env>` is
  //    `false`; otherwise, a pack containing the template arguments of the
  //    `completion_signatures` instantiation named by `SetStopped`.
  //  * Let `MoreSigs...` be a pack of the template arguments of the
  //    `completion_signatures` instantiation named by `AddlSigs`.
  //
  //  Then `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` names the type `completion_signatures< Sigs... >` where
  //  `Sigs...` is the unique set of types in `[Vs..., Es..., Ss...,
  //  MoreSigs...]`.
  //
  //  If any of the above type computations are ill-formed,
  //  `make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetError,
  //  SendsStopped>` is an alias for an empty struct
  template <                                                                 //
    class _Sender,                                                           //
    class _Env = __default_env,                                              //
    __valid_completion_signatures _Sigs = completion_signatures<>,           //
    template <class...> class _SetValue = __compl_sigs::__default_set_value, //
    template <class> class _SetError = __compl_sigs::__default_set_error,    //
    __valid_completion_signatures _SetStopped = completion_signatures<set_stopped_t()>>
    requires sender_in<_Sender, _Env>
  using make_completion_signatures = //
    __msuccess_or_t<                 //
      __try_make_completion_signatures<
        _Sender,
        _Env,
        _Sigs,
        __q<_SetValue>,
        __q<_SetError>,
        _SetStopped>>;

  // Needed fairly often
  using __with_exception_ptr = completion_signatures<set_error_t(std::exception_ptr)>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  namespace __schedule {
    struct schedule_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_t, _Scheduler>
      STDEXEC_ATTRIBUTE((host, device)) auto operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>) {
        static_assert(sender<tag_invoke_result_t<schedule_t, _Scheduler>>);
        return tag_invoke(schedule_t{}, (_Scheduler&&) __sched);
      }

      friend constexpr bool tag_invoke(forwarding_query_t, schedule_t) {
        return false;
      }
    };
  }

  using __schedule::schedule_t;
  inline constexpr schedule_t schedule{};

  // NOT TO SPEC
  template <class _Tag, const auto& _Predicate>
  concept tag_category = //
    requires {
      typename __mbool<bool{_Predicate(_Tag{})}>;
      requires bool{_Predicate(_Tag{})};
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.schedulers]
  template <class _Scheduler>
  concept __has_schedule = //
    requires(_Scheduler&& __sched) {
      { schedule((_Scheduler&&) __sched) } -> sender;
    };

  template <class _Scheduler>
  concept __sender_has_completion_scheduler =
    requires(_Scheduler&& __sched, get_completion_scheduler_t<set_value_t>&& __tag) {
      {
        tag_invoke(std::move(__tag), get_env(schedule((_Scheduler&&) __sched)))
      } -> same_as<__decay_t<_Scheduler>>;
    };

  template <class _Scheduler>
  concept scheduler =                                //
    __has_schedule<_Scheduler> &&                    //
    __sender_has_completion_scheduler<_Scheduler> && //
    equality_comparable<__decay_t<_Scheduler>> &&    //
    copy_constructible<__decay_t<_Scheduler>>;

  template <scheduler _Scheduler>
  using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  template <receiver _Receiver>
  using __current_scheduler_t = __call_result_t<get_scheduler_t, env_of_t<_Receiver>>;

  template <class _SchedulerProvider>
  concept __scheduler_provider = //
    requires(const _SchedulerProvider& __sp) {
      { get_scheduler(__sp) } -> scheduler;
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  namespace __start {
    struct start_t {
      template <class _Op>
        requires tag_invocable<start_t, _Op&>
      STDEXEC_ATTRIBUTE((always_inline)) //
        void
        operator()(_Op& __op) const noexcept {
        static_assert(nothrow_tag_invocable<start_t, _Op&>);
        (void) tag_invoke(start_t{}, __op);
      }
    };
  }

  using __start::start_t;
  inline constexpr start_t start{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template <class _Op>
  concept operation_state =  //
    destructible<_Op> &&     //
    std::is_object_v<_Op> && //
    requires(_Op& __op) {    //
      start(__op);
    };

#if !STDEXEC_STD_NO_COROUTINES_
  /////////////////////////////////////////////////////////////////////////////
  // __connect_awaitable_
  namespace __connect_awaitable_ {
    struct __promise_base {
      __coro::suspend_always initial_suspend() noexcept {
        return {};
      }

      [[noreturn]] __coro::suspend_always final_suspend() noexcept {
        std::terminate();
      }

      [[noreturn]] void unhandled_exception() noexcept {
        std::terminate();
      }

      [[noreturn]] void return_void() noexcept {
        std::terminate();
      }
    };

    struct __operation_base {
      __coro::coroutine_handle<> __coro_;

      explicit __operation_base(__coro::coroutine_handle<> __hcoro) noexcept
        : __coro_(__hcoro) {
      }

      __operation_base(__operation_base&& __other) noexcept
        : __coro_(std::exchange(__other.__coro_, {})) {
      }

      ~__operation_base() {
        if (__coro_) {
#if STDEXEC_MSVC()
          // MSVCBUG https://developercommunity.visualstudio.com/t/Double-destroy-of-a-local-in-coroutine-d/10456428

          // Reassign __coro_ before calling destroy to make the mutation
          // observable and to hopefully ensure that the compiler does not eliminate it.
          auto __coro = __coro_;
          __coro_ = {};
          __coro.destroy();
#else
          __coro_.destroy();
#endif
        }
      }

      friend void tag_invoke(start_t, __operation_base& __self) noexcept {
        __self.__coro_.resume();
      }
    };

    template <class _ReceiverId>
    struct __promise;

    template <class _ReceiverId>
    struct __operation {
      struct __t : __operation_base {
        using promise_type = stdexec::__t<__promise<_ReceiverId>>;
        using __operation_base::__operation_base;
      };
    };

    template <class _ReceiverId>
    struct __promise {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __promise_base {
        using __id = __promise;

        explicit __t(auto&, _Receiver& __rcvr) noexcept
          : __rcvr_(__rcvr) {
        }

        __coro::coroutine_handle<> unhandled_stopped() noexcept {
          set_stopped((_Receiver&&) __rcvr_);
          // Returning noop_coroutine here causes the __connect_awaitable
          // coroutine to never resume past the point where it co_await's
          // the awaitable.
          return __coro::noop_coroutine();
        }

        stdexec::__t<__operation<_ReceiverId>> get_return_object() noexcept {
          return stdexec::__t<__operation<_ReceiverId>>{
            __coro::coroutine_handle<__t>::from_promise(*this)};
        }

        template <class _Awaitable>
        _Awaitable&& await_transform(_Awaitable&& __awaitable) noexcept {
          return (_Awaitable&&) __awaitable;
        }

        template <class _Awaitable>
          requires tag_invocable<as_awaitable_t, _Awaitable, __t&>
        auto await_transform(_Awaitable&& __awaitable) //
          noexcept(nothrow_tag_invocable<as_awaitable_t, _Awaitable, __t&>)
            -> tag_invoke_result_t<as_awaitable_t, _Awaitable, __t&> {
          return tag_invoke(as_awaitable, (_Awaitable&&) __awaitable, *this);
        }

        // Pass through the get_env receiver query
        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__rcvr_);
        }

        _Receiver& __rcvr_;
      };
    };

    template <receiver _Receiver>
    using __promise_t = __t<__promise<__id<_Receiver>>>;

    template <receiver _Receiver>
    using __operation_t = __t<__operation<__id<_Receiver>>>;

    struct __connect_awaitable_t {
     private:
      template <class _Fun, class... _Ts>
      static auto __co_call(_Fun __fun, _Ts&&... __as) noexcept {
        auto __fn = [&, __fun]() noexcept {
          __fun((_Ts&&) __as...);
        };

        struct __awaiter {
          decltype(__fn) __fn_;

          static constexpr bool await_ready() noexcept {
            return false;
          }

          void await_suspend(__coro::coroutine_handle<>) noexcept {
            __fn_();
          }

          [[noreturn]] void await_resume() noexcept {
            std::terminate();
          }
        };

        return __awaiter{__fn};
      }

      template <class _Awaitable, class _Receiver>
#if STDEXEC_GCC() && (__GNUC__ > 11)
      __attribute__((__used__))
#endif
      static __operation_t<_Receiver>
        __co_impl(_Awaitable __awaitable, _Receiver __rcvr) {
        using __result_t = __await_result_t<_Awaitable, __promise_t<_Receiver>>;
        std::exception_ptr __eptr;
        try {
          if constexpr (same_as<__result_t, void>)
            co_await (
              co_await (_Awaitable&&) __awaitable, __co_call(set_value, (_Receiver&&) __rcvr));
          else
            co_await __co_call(
              set_value, (_Receiver&&) __rcvr, co_await (_Awaitable&&) __awaitable);
        } catch (...) {
          __eptr = std::current_exception();
        }
        co_await __co_call(set_error, (_Receiver&&) __rcvr, (std::exception_ptr&&) __eptr);
      }

      template <receiver _Receiver, class _Awaitable>
      using __completions_t = //
        completion_signatures<
          __minvoke< // set_value_t() or set_value_t(T)
            __remove<void, __qf<set_value_t>>,
            __await_result_t<_Awaitable, __promise_t<_Receiver>>>,
          set_error_t(std::exception_ptr),
          set_stopped_t()>;

     public:
      template <class _Receiver, __awaitable<__promise_t<_Receiver>> _Awaitable>
        requires receiver_of<_Receiver, __completions_t<_Receiver, _Awaitable>>
      __operation_t<_Receiver> operator()(_Awaitable&& __awaitable, _Receiver __rcvr) const {
        return __co_impl((_Awaitable&&) __awaitable, (_Receiver&&) __rcvr);
      }
    };
  } // namespace __connect_awaitable_

  using __connect_awaitable_::__connect_awaitable_t;
#else
  struct __connect_awaitable_t { };
#endif
  inline constexpr __connect_awaitable_t __connect_awaitable{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  namespace __connect {
    struct connect_t;

    template <class _Sender, class _Receiver>
    using __tfx_sender = //
      transform_sender_result_t<
        __late_domain_of_t<_Sender, env_of_t<_Receiver&>>,
        _Sender,
        env_of_t<_Receiver&>>;

    template <class _Sender, class _Receiver>
    concept __connectable_with_tag_invoke_ =     //
      receiver<_Receiver> &&                     //
      sender_in<_Sender, env_of_t<_Receiver>> && //
      __receiver_from<_Receiver, _Sender> &&     //
      tag_invocable<connect_t, _Sender, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __connectable_with_tag_invoke = //
      __connectable_with_tag_invoke_<__tfx_sender<_Sender, _Receiver>, _Receiver>;

    template <class _Sender, class _Receiver>
    concept __connectable_with_co_await = //
      __callable<__connect_awaitable_t, __tfx_sender<_Sender, _Receiver>, _Receiver>;

    struct connect_t {

      template <class _Sender, class _Env>
      static constexpr bool __check_signatures() {
        if constexpr (sender_in<_Sender, _Env>) {
          // Instantiate __debug_sender via completion_signatures_of_t
          // to check that the actual completions match the expected
          // completions.
          //
          // Instantiate completion_signatures_of_t only if sender_in
          // is true to workaround Clang not implementing CWG#2369 yet (connect()
          // does have a constraint for _Sender satisfying sender_in).
          using __checked_signatures [[maybe_unused]] = completion_signatures_of_t<_Sender, _Env>;
        }
        return true;
      }

      template <class _Sender, class _Receiver>
      static constexpr auto __select_impl() noexcept {
        using _Domain = __late_domain_of_t<_Sender, env_of_t<_Receiver&>>;
        constexpr bool _NothrowTfxSender =
          __nothrow_callable<get_env_t, _Receiver&>
          && __nothrow_callable<transform_sender_t, _Domain, _Sender, env_of_t<_Receiver&>>;
        using _TfxSender = __tfx_sender<_Sender, _Receiver&>;

#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
        static_assert(__check_signatures<_TfxSender, env_of_t<_Receiver>>());
#endif

        if constexpr (__connectable_with_tag_invoke<_Sender, _Receiver>) {
          using _Result = tag_invoke_result_t<connect_t, _TfxSender, _Receiver>;
          constexpr bool _Nothrow = //
            _NothrowTfxSender && nothrow_tag_invocable<connect_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__connectable_with_co_await<_Sender, _Receiver>) {
          using _Result = __call_result_t<__connect_awaitable_t, _TfxSender, _Receiver>;
          return static_cast<_Result (*)()>(nullptr);
        } else {
          using _Result = __debug::__debug_operation;
          return static_cast<_Result (*)() noexcept(_NothrowTfxSender)>(nullptr);
        }
      }

      template <class _Sender, class _Receiver>
      using __select_impl_t = decltype(__select_impl<_Sender, _Receiver>());

      template <sender _Sender, receiver _Receiver>
        requires __connectable_with_tag_invoke<_Sender, _Receiver>
              || __connectable_with_co_await<_Sender, _Receiver>
              || __is_debug_env<env_of_t<_Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(__nothrow_callable<__select_impl_t<_Sender, _Receiver>>)
          -> __call_result_t<__select_impl_t<_Sender, _Receiver>> {
        using _TfxSender = __tfx_sender<_Sender, _Receiver&>;
        auto&& __env = get_env(__rcvr);
        auto __domain = __get_late_domain(__sndr, __env);

        if constexpr (__connectable_with_tag_invoke<_Sender, _Receiver>) {
          static_assert(
            operation_state<tag_invoke_result_t<connect_t, _TfxSender, _Receiver>>,
            "stdexec::connect(sender, receiver) must return a type that "
            "satisfies the operation_state concept");
          return tag_invoke(
            connect_t{},
            transform_sender(__domain, (_Sender&&) __sndr, __env),
            (_Receiver&&) __rcvr);
        } else if constexpr (__connectable_with_co_await<_Sender, _Receiver>) {
          return __connect_awaitable( //
            transform_sender(__domain, (_Sender&&) __sndr, __env),
            (_Receiver&&) __rcvr);
        } else {
          // This should generate an instantiation backtrace that contains useful
          // debugging information.
          using __tag_invoke::tag_invoke;
          tag_invoke(
            *this, transform_sender(__domain, (_Sender&&) __sndr, __env), (_Receiver&&) __rcvr);
        }
      }

      friend constexpr bool tag_invoke(forwarding_query_t, connect_t) noexcept {
        return false;
      }
    };
  } // namespace __connect

  using __connect::connect_t;
  inline constexpr __connect::connect_t connect{};

  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  template <class _Sender, class _Receiver>
  concept sender_to =
    receiver<_Receiver> &&                     //
    sender_in<_Sender, env_of_t<_Receiver>> && //
    __receiver_from<_Receiver, _Sender> &&     //
    requires(_Sender&& __sndr, _Receiver&& __rcvr) {
      connect((_Sender&&) __sndr, (_Receiver&&) __rcvr);
    };

  template <class _Tag, class... _Args>
  _Tag __tag_of_sig_(_Tag (*)(_Args...));
  template <class _Sig>
  using __tag_of_sig_t = decltype(stdexec::__tag_of_sig_((_Sig*) nullptr));

  template <class _Sender, class _SetSig, class _Env = __default_env>
  concept sender_of =
    sender_in<_Sender, _Env>
    && same_as<
      __types<_SetSig>,
      __gather_completions_for<
        __tag_of_sig_t<_SetSig>,
        _Sender,
        _Env,
        __qf<__tag_of_sig_t<_SetSig>>,
        __q<__types>>>;

#if !STDEXEC_STD_NO_COROUTINES_
  /////////////////////////////////////////////////////////////////////////////
  // stdexec::as_awaitable [execution.coro_utils.as_awaitable]
  namespace __as_awaitable {
    struct __void { };
    template <class _Value>
    using __value_or_void_t = __if<std::is_same<_Value, void>, __void, _Value>;
    template <class _Value>
    using __expected_t =
      std::variant<std::monostate, __value_or_void_t<_Value>, std::exception_ptr>;

    template <class _Value>
    struct __receiver_base {
      using receiver_concept = receiver_t;

      template <same_as<set_value_t> _Tag, class... _Us>
        requires constructible_from<__value_or_void_t<_Value>, _Us...>
      friend void tag_invoke(_Tag, __receiver_base&& __self, _Us&&... __us) noexcept {
        try {
          __self.__result_->template emplace<1>((_Us&&) __us...);
          __self.__continuation_.resume();
        } catch (...) {
          set_error((__receiver_base&&) __self, std::current_exception());
        }
      }

      template <same_as<set_error_t> _Tag, class _Error>
      friend void tag_invoke(_Tag, __receiver_base&& __self, _Error&& __err) noexcept {
        if constexpr (__decays_to<_Error, std::exception_ptr>)
          __self.__result_->template emplace<2>((_Error&&) __err);
        else if constexpr (__decays_to<_Error, std::error_code>)
          __self.__result_->template emplace<2>(std::make_exception_ptr(std::system_error(__err)));
        else
          __self.__result_->template emplace<2>(std::make_exception_ptr((_Error&&) __err));
        __self.__continuation_.resume();
      }

      __expected_t<_Value>* __result_;
      __coro::coroutine_handle<> __continuation_;
    };

    template <class _PromiseId, class _Value>
    struct __receiver {
      using _Promise = stdexec::__t<_PromiseId>;

      struct __t : __receiver_base<_Value> {
        using __id = __receiver;

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
            __self.__continuation_.address());
          __coro::coroutine_handle<> __stopped_continuation =
            __continuation.promise().unhandled_stopped();
          __stopped_continuation.resume();
        }

        // Forward get_env query to the coroutine promise
        friend env_of_t<_Promise&> tag_invoke(get_env_t, const __t& __self) noexcept {
          auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
            __self.__continuation_.address());
          return get_env(__continuation.promise());
        }
      };
    };

    template <class _Sender, class _Promise>
    using __value_t = __decay_t<
      __value_types_of_t< _Sender, env_of_t<_Promise&>, __msingle_or<void>, __msingle_or<void>>>;

    template <class _Sender, class _Promise>
    using __receiver_t = __t<__receiver<__id<_Promise>, __value_t<_Sender, _Promise>>>;

    template <class _Value>
    struct __sender_awaitable_base {
      bool await_ready() const noexcept {
        return false;
      }

      _Value await_resume() {
        switch (__result_.index()) {
        case 0: // receiver contract not satisfied
          STDEXEC_ASSERT(!"_Should never get here");
          break;
        case 1: // set_value
          if constexpr (!std::is_void_v<_Value>)
            return (_Value&&) std::get<1>(__result_);
          else
            return;
        case 2: // set_error
          std::rethrow_exception(std::get<2>(__result_));
        }
        std::terminate();
      }

     protected:
      __expected_t<_Value> __result_;
    };

    template <class _PromiseId, class _SenderId>
    struct __sender_awaitable {
      using _Promise = stdexec::__t<_PromiseId>;
      using _Sender = stdexec::__t<_SenderId>;
      using __value = __value_t<_Sender, _Promise>;

      struct __t : __sender_awaitable_base<__value> {
        __t(_Sender&& sndr, __coro::coroutine_handle<_Promise> __hcoro) //
          noexcept(__nothrow_connectable<_Sender, __receiver>)
          : __op_state_(connect(
            (_Sender&&) sndr,
            __receiver{
              {&this->__result_, __hcoro}
        })) {
        }

        void await_suspend(__coro::coroutine_handle<_Promise>) noexcept {
          start(__op_state_);
        }
       private:
        using __receiver = __receiver_t<_Sender, _Promise>;
        connect_result_t<_Sender, __receiver> __op_state_;
      };
    };

    template <class _Promise, class _Sender>
    using __sender_awaitable_t = __t<__sender_awaitable<__id<_Promise>, __id<_Sender>>>;

    template <class _Sender, class _Promise>
    concept __awaitable_sender =
      sender_in<_Sender, env_of_t<_Promise&>> &&             //
      __mvalid<__value_t, _Sender, _Promise> &&              //
      sender_to<_Sender, __receiver_t<_Sender, _Promise>> && //
      requires(_Promise& __promise) {
        { __promise.unhandled_stopped() } -> convertible_to<__coro::coroutine_handle<>>;
      };

    struct __unspecified {
      __unspecified get_return_object() noexcept;
      __unspecified initial_suspend() noexcept;
      __unspecified final_suspend() noexcept;
      void unhandled_exception() noexcept;
      void return_void() noexcept;
      __coro::coroutine_handle<> unhandled_stopped() noexcept;
    };

    struct as_awaitable_t {
      template <class _Tp, class _Promise>
      static constexpr auto __select_impl_() noexcept {
        if constexpr (tag_invocable<as_awaitable_t, _Tp, _Promise&>) {
          using _Result = tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
          constexpr bool _Nothrow = nothrow_tag_invocable<as_awaitable_t, _Tp, _Promise&>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else if constexpr (__awaitable<_Tp, __unspecified>) { // NOT __awaitable<_Tp, _Promise> !!
          return static_cast < _Tp && (*) () noexcept > (nullptr);
        } else if constexpr (__awaitable_sender<_Tp, _Promise>) {
          using _Result = __sender_awaitable_t<_Promise, _Tp>;
          constexpr bool _Nothrow =
            __nothrow_constructible_from<_Result, _Tp, __coro::coroutine_handle<_Promise>>;
          return static_cast<_Result (*)() noexcept(_Nothrow)>(nullptr);
        } else {
          return static_cast < _Tp && (*) () noexcept > (nullptr);
        }
      }
      template <class _Tp, class _Promise>
      using __select_impl_t = decltype(__select_impl_<_Tp, _Promise>());

      template <class _Tp, class _Promise>
      auto operator()(_Tp&& __t, _Promise& __promise) const
        noexcept(__nothrow_callable<__select_impl_t<_Tp, _Promise>>)
          -> __call_result_t<__select_impl_t<_Tp, _Promise>> {
        if constexpr (tag_invocable<as_awaitable_t, _Tp, _Promise&>) {
          using _Result = tag_invoke_result_t<as_awaitable_t, _Tp, _Promise&>;
          static_assert(__awaitable<_Result, _Promise>);
          return tag_invoke(*this, (_Tp&&) __t, __promise);
        } else if constexpr (__awaitable<_Tp, __unspecified>) { // NOT __awaitable<_Tp, _Promise> !!
          return (_Tp&&) __t;
        } else if constexpr (__awaitable_sender<_Tp, _Promise>) {
          auto __hcoro = __coro::coroutine_handle<_Promise>::from_promise(__promise);
          return __sender_awaitable_t<_Promise, _Tp>{(_Tp&&) __t, __hcoro};
        } else {
          return (_Tp&&) __t;
        }
      }
    };
  } // namespace __as_awaitable

  using __as_awaitable::as_awaitable_t;
  inline constexpr as_awaitable_t as_awaitable{};

  namespace __with_awaitable_senders {

    template <class _Promise = void>
    class __continuation_handle;

    template <>
    class __continuation_handle<void> {
     public:
      __continuation_handle() = default;

      template <class _Promise>
      __continuation_handle(__coro::coroutine_handle<_Promise> __coro) noexcept
        : __coro_(__coro) {
        if constexpr (requires(_Promise& __promise) { __promise.unhandled_stopped(); }) {
          __stopped_callback_ = [](void* __address) noexcept -> __coro::coroutine_handle<> {
            // This causes the rest of the coroutine (the part after the co_await
            // of the sender) to be skipped and invokes the calling coroutine's
            // stopped handler.
            return __coro::coroutine_handle<_Promise>::from_address(__address)
              .promise()
              .unhandled_stopped();
          };
        }
        // If _Promise doesn't implement unhandled_stopped(), then if a "stopped" unwind
        // reaches this point, it's considered an unhandled exception and terminate()
        // is called.
      }

      __coro::coroutine_handle<> handle() const noexcept {
        return __coro_;
      }

      __coro::coroutine_handle<> unhandled_stopped() const noexcept {
        return __stopped_callback_(__coro_.address());
      }

     private:
      __coro::coroutine_handle<> __coro_{};
      using __stopped_callback_t = __coro::coroutine_handle<> (*)(void*) noexcept;
      __stopped_callback_t __stopped_callback_ = [](void*) noexcept -> __coro::coroutine_handle<> {
        std::terminate();
      };
    };

    template <class _Promise>
    class __continuation_handle {
     public:
      __continuation_handle() = default;

      __continuation_handle(__coro::coroutine_handle<_Promise> __coro) noexcept
        : __continuation_{__coro} {
      }

      __coro::coroutine_handle<_Promise> handle() const noexcept {
        return __coro::coroutine_handle<_Promise>::from_address(__continuation_.handle().address());
      }

      __coro::coroutine_handle<> unhandled_stopped() const noexcept {
        return __continuation_.unhandled_stopped();
      }

     private:
      __continuation_handle<> __continuation_{};
    };

    struct __with_awaitable_senders_base {
      template <class _OtherPromise>
      void set_continuation(__coro::coroutine_handle<_OtherPromise> __hcoro) noexcept {
        static_assert(!std::is_void_v<_OtherPromise>);
        __continuation_ = __hcoro;
      }

      void set_continuation(__continuation_handle<> __continuation) noexcept {
        __continuation_ = __continuation;
      }

      __continuation_handle<> continuation() const noexcept {
        return __continuation_;
      }

      __coro::coroutine_handle<> unhandled_stopped() noexcept {
        return __continuation_.unhandled_stopped();
      }

     private:
      __continuation_handle<> __continuation_{};
    };

    template <class _Promise>
    struct with_awaitable_senders : __with_awaitable_senders_base {
      template <class _Value>
      auto await_transform(_Value&& __val) -> __call_result_t<as_awaitable_t, _Value, _Promise&> {
        static_assert(derived_from<_Promise, with_awaitable_senders>);
        return as_awaitable((_Value&&) __val, static_cast<_Promise&>(*this));
      }
    };
  } // namespace __with_awaitable_senders;

  using __with_awaitable_senders::with_awaitable_senders;
  using __with_awaitable_senders::__continuation_handle;
#endif

  namespace {
    inline constexpr auto __ref = []<class _Ty>(_Ty& __ty) noexcept {
      return [__ty = &__ty]() noexcept -> decltype(auto) {
        return (*__ty);
      };
    };
  }

  template <class _Ty>
  using __ref_t = decltype(__ref(__declval<_Ty&>()));

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __submit
  namespace __submit_ {
    template <class _OpRef>
    struct __receiver {
      using receiver_concept = receiver_t;
      using __t = __receiver;
      using __id = __receiver;

      using _Operation = __decay_t<__call_result_t<_OpRef>>;
      using _Receiver = stdexec::__t<__mapply<__q<__msecond>, _Operation>>;

      _OpRef __opref_;

      // Forward all the receiver ops, and delete the operation state.
      template <__completion_tag _Tag, class... _As>
        requires __callable<_Tag, _Receiver, _As...>
      friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept {
        __tag((_Receiver&&) __self.__opref_().__rcvr_, (_As&&) __as...);
        __self.__delete_op();
      }

      void __delete_op() noexcept {
        _Operation* __op = &__opref_();
        if constexpr (__callable<get_allocator_t, env_of_t<_Receiver>>) {
          auto&& __env = get_env(__op->__rcvr_);
          auto __alloc = get_allocator(__env);
          using _Alloc = decltype(__alloc);
          using _OpAlloc =
            typename std::allocator_traits<_Alloc>::template rebind_alloc<_Operation>;
          _OpAlloc __op_alloc{__alloc};
          std::allocator_traits<_OpAlloc>::destroy(__op_alloc, __op);
          std::allocator_traits<_OpAlloc>::deallocate(__op_alloc, __op, 1);
        } else {
          delete __op;
        }
      }

      // Forward all receiever queries.
      friend auto tag_invoke(get_env_t, const __receiver& __self) noexcept -> env_of_t<_Receiver&> {
        return get_env(__self.__opref_().__rcvr_);
      }
    };

    template <class _SenderId, class _ReceiverId>
    struct __operation {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = __receiver<__ref_t<__operation>>;

      STDEXEC_ATTRIBUTE((no_unique_address)) _Receiver __rcvr_;
      connect_result_t<_Sender, __receiver_t> __op_state_;

      __operation(_Sender&& __sndr, _Receiver __rcvr)
        : __rcvr_((_Receiver&&) __rcvr)
        , __op_state_(connect((_Sender&&) __sndr, __receiver_t{__ref(*this)})) {
      }
    };

    struct __submit_t {
      template <receiver _Receiver, sender_to<_Receiver> _Sender>
      void operator()(_Sender&& __sndr, _Receiver __rcvr) const noexcept(false) {
        if constexpr (__callable<get_allocator_t, env_of_t<_Receiver>>) {
          auto&& __env = get_env(__rcvr);
          auto __alloc = get_allocator(__env);
          using _Alloc = decltype(__alloc);
          using _Op = __operation<__id<_Sender>, __id<_Receiver>>;
          using _OpAlloc = typename std::allocator_traits<_Alloc>::template rebind_alloc<_Op>;
          _OpAlloc __op_alloc{__alloc};
          auto __op = std::allocator_traits<_OpAlloc>::allocate(__op_alloc, 1);
          try {
            std::allocator_traits<_OpAlloc>::construct(
              __op_alloc, __op, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
            start(__op->__op_state_);
          } catch (...) {
            std::allocator_traits<_OpAlloc>::deallocate(__op_alloc, __op, 1);
            throw;
          }
        } else {
          start((new __operation<__id<_Sender>, __id<_Receiver>>{
                   (_Sender&&) __sndr, (_Receiver&&) __rcvr})
                  ->__op_state_);
        }
      }
    };
  } // namespace __submit_

  using __submit_::__submit_t;
  inline constexpr __submit_t __submit{};

  namespace __inln {
    struct __schedule_t { };

    struct __scheduler {
      using __t = __scheduler;
      using __id = __scheduler;

      template <class _Tag = __schedule_t>
      STDEXEC_ATTRIBUTE((host, device))
      friend auto tag_invoke(schedule_t, __scheduler) {
        return __make_sexpr<_Tag>();
      }

      friend forward_progress_guarantee
        tag_invoke(get_forward_progress_guarantee_t, __scheduler) noexcept {
        return forward_progress_guarantee::weakly_parallel;
      }

      bool operator==(const __scheduler&) const noexcept = default;
    };
  } // namespace __inln

  template <>
  struct __sexpr_impl<__inln::__schedule_t> : __sexpr_defaults {
    static constexpr auto get_attrs = //
      [](__ignore) noexcept
      -> __env::__prop<__inln::__scheduler(get_completion_scheduler_t<set_value_t>)> {
      return __mkprop(__inln::__scheduler{}, get_completion_scheduler<set_value_t>);
    };

    static constexpr auto get_completion_signatures = //
      [](__ignore, __ignore) noexcept -> completion_signatures<set_value_t()> {
      return {};
    };

    static constexpr auto start = //
      []<class _Receiver>(__ignore, _Receiver& __rcvr) noexcept -> void {
      set_value((_Receiver&&) __rcvr);
    };
  };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  namespace __start_detached {
    template <class _EnvId>
    struct __detached_receiver {
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        using receiver_concept = receiver_t;
        using __id = __detached_receiver;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Env __env_;

        template <same_as<set_value_t> _Tag, class... _As>
        friend void tag_invoke(_Tag, __t&&, _As&&...) noexcept {
        }

        template <same_as<set_error_t> _Tag, class _Error>
        [[noreturn]] friend void tag_invoke(_Tag, __t&&, _Error&&) noexcept {
          std::terminate();
        }

        template <same_as<set_stopped_t> _Tag>
        friend void tag_invoke(_Tag, __t&&) noexcept {
        }

        friend const _Env& tag_invoke(get_env_t, const __t& __self) noexcept {
          // BUGBUG NOT TO SPEC
          return __self.__env_;
        }
      };
    };
    template <class _Env = empty_env>
    using __detached_receiver_t = __t<__detached_receiver<__id<__decay_t<_Env>>>>;

    struct start_detached_t {
      template <sender_in<empty_env> _Sender>
        requires __callable<apply_sender_t, __early_domain_of_t<_Sender>, start_detached_t, _Sender>
      void operator()(_Sender&& __sndr) const {
        auto __domain = __get_early_domain(__sndr);
        stdexec::apply_sender(__domain, *this, (_Sender&&) __sndr);
      }

      template <class _Env, sender_in<_Env> _Sender>
        requires __callable<
          apply_sender_t,
          __late_domain_of_t<_Sender, _Env>,
          start_detached_t,
          _Sender,
          _Env>
      void operator()(_Sender&& __sndr, _Env&& __env) const {
        auto __domain = __get_late_domain(__sndr, __env);
        stdexec::apply_sender(__domain, *this, (_Sender&&) __sndr, (_Env&&) __env);
      }

      using _Sender = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          start_detached_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(start_detached_t, _Sender)>;

      template <class _Sender, class _Env = empty_env>
        requires sender_to<_Sender, __detached_receiver_t<_Env>>
      void apply_sender(_Sender&& __sndr, _Env&& __env = {}) const {
        __submit((_Sender&&) __sndr, __detached_receiver_t<_Env>{(_Env&&) __env});
      }
    };
  } // namespace __start_detached

  using __start_detached::start_detached_t;
  inline constexpr start_detached_t start_detached{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  namespace __just {
    template <class _JustTag>
    struct __impl : __sexpr_defaults {
      using __tag_t = typename _JustTag::__tag_t;

      static constexpr auto get_completion_signatures =
        []<class _Sender>(_Sender&&, __ignore) noexcept {
          static_assert(sender_expr_for<_Sender, _JustTag>);
          return completion_signatures<__mapply<__qf<__tag_t>, __decay_t<__data_of<_Sender>>>>{};
        };


      static constexpr auto start =
        []<class _State, class _Receiver>(_State& __state, _Receiver& __rcvr) noexcept -> void {
        __tup::__apply(
          [&]<class... _Ts>(_Ts&... __ts) noexcept {
            __tag_t()(std::move(__rcvr), std::move(__ts)...);
          },
          __state);
      };
    };

    struct just_t {
      using __tag_t = set_value_t;

      template <__movable_value... _Ts>
      STDEXEC_ATTRIBUTE((host, device))
      auto operator()(_Ts&&... __ts) const noexcept((__nothrow_decay_copyable<_Ts> && ...)) {
        return __make_sexpr<just_t>(__tuple{(_Ts&&) __ts...});
      }
    };

    struct just_error_t {
      using __tag_t = set_error_t;

      template <__movable_value _Error>
      STDEXEC_ATTRIBUTE((host, device))
      auto operator()(_Error&& __err) const noexcept(__nothrow_decay_copyable<_Error>) {
        return __make_sexpr<just_error_t>(__tuple{(_Error&&) __err});
      }
    };

    struct just_stopped_t {
      using __tag_t = set_stopped_t;

      template <class _Tag = just_stopped_t>
      STDEXEC_ATTRIBUTE((host, device))
      auto operator()() const noexcept {
        return __make_sexpr<_Tag>(__tuple{});
      }
    };
  }

  using __just::just_t;
  using __just::just_error_t;
  using __just::just_stopped_t;

  template <>
  struct __sexpr_impl<just_t> : __just::__impl<just_t> { };

  template <>
  struct __sexpr_impl<just_error_t> : __just::__impl<just_error_t> { };

  template <>
  struct __sexpr_impl<just_stopped_t> : __just::__impl<just_stopped_t> { };

  inline constexpr just_t just{};
  inline constexpr just_error_t just_error{};
  inline constexpr just_stopped_t just_stopped{};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  namespace __execute_ {
    template <class _Fun>
    struct __as_receiver {
      using receiver_concept = receiver_t;
      _Fun __fun_;

      template <same_as<set_value_t> _Tag>
      friend void tag_invoke(_Tag, __as_receiver&& __rcvr) noexcept {
        try {
          __rcvr.__fun_();
        } catch (...) {
          set_error((__as_receiver&&) __rcvr, std::exception_ptr());
        }
      }

      template <same_as<set_error_t> _Tag>
      [[noreturn]] friend void tag_invoke(_Tag, __as_receiver&&, std::exception_ptr) noexcept {
        std::terminate();
      }

      template <same_as<set_stopped_t> _Tag>
      friend void tag_invoke(_Tag, __as_receiver&&) noexcept {
      }

      friend empty_env tag_invoke(get_env_t, const __as_receiver&) noexcept {
        return {};
      }
    };

    struct execute_t {
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const noexcept(false) {
        // Look for a legacy customization
        if constexpr (tag_invocable<execute_t, _Scheduler, _Fun>) {
          tag_invoke(execute_t{}, (_Scheduler&&) __sched, (_Fun&&) __fun);
        } else {
          auto __domain = query_or(get_domain, __sched, default_domain());
          stdexec::apply_sender(__domain, *this, schedule((_Scheduler&&) __sched), (_Fun&&) __fun);
        }
      }

      template <sender_of<set_value_t()> _Sender, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void apply_sender(_Sender&& __sndr, _Fun __fun) const noexcept(false) {
        __submit((_Sender&&) __sndr, __as_receiver<_Fun>{(_Fun&&) __fun});
      }
    };
  }

  using __execute_::execute_t;
  inline constexpr execute_t execute{};

  // NOT TO SPEC:
  namespace __closure {
    template <__class _Dp>
    struct sender_adaptor_closure;
  }

  using __closure::sender_adaptor_closure;

  template <class _Tp>
  concept __sender_adaptor_closure =
    derived_from<__decay_t<_Tp>, sender_adaptor_closure<__decay_t<_Tp>>>
    && move_constructible<__decay_t<_Tp>> && constructible_from<__decay_t<_Tp>, _Tp>;

  template <class _Tp, class _Sender>
  concept __sender_adaptor_closure_for =
    __sender_adaptor_closure<_Tp> && sender<__decay_t<_Sender>>
    && __callable<_Tp, __decay_t<_Sender>> && sender<__call_result_t<_Tp, __decay_t<_Sender>>>;

  namespace __closure {
    template <class _T0, class _T1>
    struct __compose : sender_adaptor_closure<__compose<_T0, _T1>> {
      STDEXEC_ATTRIBUTE((no_unique_address)) _T0 __t0_;
      STDEXEC_ATTRIBUTE((no_unique_address)) _T1 __t1_;

      template <sender _Sender>
        requires __callable<_T0, _Sender> && __callable<_T1, __call_result_t<_T0, _Sender>>
      STDEXEC_ATTRIBUTE((always_inline)) //
        __call_result_t<_T1, __call_result_t<_T0, _Sender>>
        operator()(_Sender&& __sndr) && {
        return ((_T1&&) __t1_)(((_T0&&) __t0_)((_Sender&&) __sndr));
      }

      template <sender _Sender>
        requires __callable<const _T0&, _Sender>
              && __callable<const _T1&, __call_result_t<const _T0&, _Sender>>
      STDEXEC_ATTRIBUTE((always_inline)) //
        __call_result_t<_T1, __call_result_t<_T0, _Sender>>
        operator()(_Sender&& __sndr) const & {
        return __t1_(__t0_((_Sender&&) __sndr));
      }
    };

    template <__class _Dp>
    struct sender_adaptor_closure { };

    template <sender _Sender, __sender_adaptor_closure_for<_Sender> _Closure>
    STDEXEC_ATTRIBUTE((always_inline)) //
    __call_result_t<_Closure, _Sender> operator|(_Sender&& __sndr, _Closure&& __clsur) {
      return ((_Closure&&) __clsur)((_Sender&&) __sndr);
    }

    template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
    STDEXEC_ATTRIBUTE((always_inline)) //
    __compose<__decay_t<_T0>, __decay_t<_T1>> operator|(_T0&& __t0, _T1&& __t1) {
      return {{}, (_T0&&) __t0, (_T1&&) __t1};
    }

    template <class _Fun, class... _As>
    struct __binder_back : sender_adaptor_closure<__binder_back<_Fun, _As...>> {
      STDEXEC_ATTRIBUTE((no_unique_address)) _Fun __fun_;
      std::tuple<_As...> __as_;

      template <sender _Sender>
        requires __callable<_Fun, _Sender, _As...>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        __call_result_t<_Fun, _Sender, _As...>
        operator()(_Sender&& __sndr) && noexcept(__nothrow_callable<_Fun, _Sender, _As...>) {
        return __apply(
          [&__sndr, this](_As&... __as) -> __call_result_t<_Fun, _Sender, _As...> {
            return ((_Fun&&) __fun_)((_Sender&&) __sndr, (_As&&) __as...);
          },
          __as_);
      }

      template <sender _Sender>
        requires __callable<const _Fun&, _Sender, const _As&...>
      STDEXEC_ATTRIBUTE((host, device)) __call_result_t<const _Fun&, _Sender, const _As&...>
        operator()(_Sender&& __sndr) const & //
        noexcept(__nothrow_callable<const _Fun&, _Sender, const _As&...>) {
        return __apply(
          [&__sndr,
           this](const _As&... __as) -> __call_result_t<const _Fun&, _Sender, const _As&...> {
            return __fun_((_Sender&&) __sndr, __as...);
          },
          __as_);
      }
    };
  } // namespace __closure

  using __closure::__binder_back;

  namespace __adaptors {
    // A derived-to-base cast that works even when the base is not
    // accessible from derived.
    template <class _Tp, class _Up>
    STDEXEC_ATTRIBUTE((host, device))
    __copy_cvref_t<_Up&&, _Tp> __c_cast(_Up&& u) noexcept
      requires __decays_to<_Tp, _Tp>
    {
      static_assert(std::is_reference_v<__copy_cvref_t<_Up&&, _Tp>>);
      static_assert(STDEXEC_IS_BASE_OF(_Tp, __decay_t<_Up>));
      return (__copy_cvref_t<_Up&&, _Tp>) (_Up&&) u;
    }

    namespace __no {
      struct __nope { };

      struct __receiver : __nope {
        using receiver_concept = receiver_t;
      };

      template <same_as<set_error_t> _Tag>
      void tag_invoke(_Tag, __receiver, std::exception_ptr) noexcept;
      template <same_as<set_stopped_t> _Tag>
      void tag_invoke(_Tag, __receiver) noexcept;
      empty_env tag_invoke(get_env_t, __receiver) noexcept;
    }

    using __not_a_receiver = __no::__receiver;

    template <class _Base>
    struct __adaptor_base {
      template <class _T1>
        requires constructible_from<_Base, _T1>
      explicit __adaptor_base(_T1&& __base)
        : __base_((_T1&&) __base) {
      }

     private:
      STDEXEC_ATTRIBUTE((no_unique_address)) _Base __base_;

     protected:
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
      _Base& base() & noexcept {
        return __base_;
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
      const _Base& base() const & noexcept {
        return __base_;
      }

      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
      _Base&& base() && noexcept {
        return (_Base&&) __base_;
      }
    };

    template <derived_from<__no::__nope> _Base>
    struct __adaptor_base<_Base> { };

// BUGBUG Not to spec: on gcc and nvc++, member functions in derived classes
// don't shadow type aliases of the same name in base classes. :-O
// On mingw gcc, 'bool(type::existing_member_function)' evaluates to true,
// but 'int(type::existing_member_function)' is an error (as desired).
#define _DISPATCH_MEMBER(_TAG) \
  template <class _Self, class... _Ts> \
  STDEXEC_ATTRIBUTE((host, device, always_inline)) \
  static auto __call_##_TAG(_Self&& __self, _Ts&&... __ts) noexcept \
    -> decltype(((_Self&&) __self)._TAG((_Ts&&) __ts...)) { \
    static_assert(noexcept(((_Self&&) __self)._TAG((_Ts&&) __ts...))); \
    return ((_Self&&) __self)._TAG((_Ts&&) __ts...); \
  } /**/
#define _CALL_MEMBER(_TAG, ...) __call_##_TAG(__VA_ARGS__)

#if STDEXEC_CLANG()
// Only clang gets this right.
#define _MISSING_MEMBER(_Dp, _TAG) requires { typename _Dp::_TAG; }
#define _DEFINE_MEMBER(_TAG) _DISPATCH_MEMBER(_TAG) using _TAG = void
#else
#define _MISSING_MEMBER(_Dp, _TAG) (__missing_##_TAG<_Dp>())
#define _DEFINE_MEMBER(_TAG) \
  template <class _Dp> \
  static constexpr bool __missing_##_TAG() noexcept { \
    return requires { requires bool(int(_Dp::_TAG)); }; \
  } \
  _DISPATCH_MEMBER(_TAG) \
  static constexpr int _TAG = 1 /**/
#endif

    template <__class _Derived, class _Base = __not_a_receiver>
    struct receiver_adaptor
      : __adaptor_base<_Base>
      , receiver_t {
      friend _Derived;
      _DEFINE_MEMBER(set_value);
      _DEFINE_MEMBER(set_error);
      _DEFINE_MEMBER(set_stopped);
      _DEFINE_MEMBER(get_env);

      static constexpr bool __has_base = !derived_from<_Base, __no::__nope>;

      template <class _Dp>
      using __base_from_derived_t = decltype(__declval<_Dp>().base());

      using __get_base_t =
        __if_c< __has_base, __mbind_back_q<__copy_cvref_t, _Base>, __q<__base_from_derived_t>>;

      template <class _Dp>
      using __base_t = __minvoke<__get_base_t, _Dp&&>;

      template <class _Dp>
      STDEXEC_ATTRIBUTE((host, device))
      static __base_t<_Dp> __get_base(_Dp&& __self) noexcept {
        if constexpr (__has_base) {
          return __c_cast<receiver_adaptor>((_Dp&&) __self).base();
        } else {
          return ((_Dp&&) __self).base();
        }
      }

      template <same_as<set_value_t> _SetValue, class... _As>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      friend auto tag_invoke(_SetValue, _Derived&& __self, _As&&... __as) noexcept //
        -> __msecond<                                                              //
          __if_c<same_as<set_value_t, _SetValue>>,
          decltype(_CALL_MEMBER(set_value, (_Derived&&) __self, (_As&&) __as...))> {
        static_assert(noexcept(_CALL_MEMBER(set_value, (_Derived&&) __self, (_As&&) __as...)));
        _CALL_MEMBER(set_value, (_Derived&&) __self, (_As&&) __as...);
      }

      template <same_as<set_value_t> _SetValue, class _Dp = _Derived, class... _As>
        requires _MISSING_MEMBER(_Dp, set_value) && tag_invocable<_SetValue, __base_t<_Dp>, _As...>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        friend void tag_invoke(_SetValue, _Derived&& __self, _As&&... __as) noexcept {
        stdexec::set_value(__get_base((_Dp&&) __self), (_As&&) __as...);
      }

      template <same_as<set_error_t> _SetError, class _Error>
      STDEXEC_ATTRIBUTE((host, device, always_inline))                              //
      friend auto tag_invoke(_SetError, _Derived&& __self, _Error&& __err) noexcept //
        -> __msecond<                                                               //
          __if_c<same_as<set_error_t, _SetError>>,
          decltype(_CALL_MEMBER(set_error, (_Derived&&) __self, (_Error&&) __err))> {
        static_assert(noexcept(_CALL_MEMBER(set_error, (_Derived&&) __self, (_Error&&) __err)));
        _CALL_MEMBER(set_error, (_Derived&&) __self, (_Error&&) __err);
      }

      template <same_as<set_error_t> _SetError, class _Error, class _Dp = _Derived>
        requires _MISSING_MEMBER(_Dp, set_error) && tag_invocable<_SetError, __base_t<_Dp>, _Error>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        friend void tag_invoke(_SetError, _Derived&& __self, _Error&& __err) noexcept {
        stdexec::set_error(__get_base((_Derived&&) __self), (_Error&&) __err);
      }

      template <same_as<set_stopped_t> _SetStopped, class _Dp = _Derived>
      STDEXEC_ATTRIBUTE((host, device, always_inline))                //
      friend auto tag_invoke(_SetStopped, _Derived&& __self) noexcept //
        -> __msecond<                                                 //
          __if_c<same_as<set_stopped_t, _SetStopped>>,
          decltype(_CALL_MEMBER(set_stopped, (_Dp&&) __self))> {
        static_assert(noexcept(_CALL_MEMBER(set_stopped, (_Derived&&) __self)));
        _CALL_MEMBER(set_stopped, (_Derived&&) __self);
      }

      template <same_as<set_stopped_t> _SetStopped, class _Dp = _Derived>
        requires _MISSING_MEMBER(_Dp, set_stopped) && tag_invocable<_SetStopped, __base_t<_Dp>>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        friend void tag_invoke(_SetStopped, _Derived&& __self) noexcept {
        stdexec::set_stopped(__get_base((_Derived&&) __self));
      }

      // Pass through the get_env receiver query
      template <same_as<get_env_t> _GetEnv, class _Dp = _Derived>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
      friend auto tag_invoke(_GetEnv, const _Derived& __self) noexcept
        -> decltype(_CALL_MEMBER(get_env, (const _Dp&) __self)) {
        static_assert(noexcept(_CALL_MEMBER(get_env, __self)));
        return _CALL_MEMBER(get_env, __self);
      }

      template <same_as<get_env_t> _GetEnv, class _Dp = _Derived>
        requires _MISSING_MEMBER(_Dp, get_env)
      STDEXEC_ATTRIBUTE((host, device, always_inline)) //
        friend auto tag_invoke(_GetEnv, const _Derived& __self) noexcept
        -> env_of_t<__base_t<const _Dp&>> {
        return stdexec::get_env(__get_base(__self));
      }

     public:
      receiver_adaptor() = default;
      using __adaptor_base<_Base>::__adaptor_base;

      using receiver_concept = receiver_t;
    };
  } // namespace __adaptors

  template <__class _Derived, receiver _Base = __adaptors::__not_a_receiver>
  using receiver_adaptor = __adaptors::receiver_adaptor<_Derived, _Base>;

  template <class _Receiver, class _Fun, class... _As>
  concept __receiver_of_invoke_result = //
    receiver_of<
      _Receiver,
      completion_signatures<
        __minvoke<__remove<void, __qf<set_value_t>>, __invoke_result_t<_Fun, _As...>>>>;

  template <bool _CanThrow = false, class _Receiver, class _Fun, class... _As>
  void __set_value_invoke(_Receiver&& __rcvr, _Fun&& __fun, _As&&... __as) noexcept(!_CanThrow) {
    if constexpr (_CanThrow || __nothrow_invocable<_Fun, _As...>) {
      if constexpr (same_as<void, __invoke_result_t<_Fun, _As...>>) {
        __invoke((_Fun&&) __fun, (_As&&) __as...);
        set_value((_Receiver&&) __rcvr);
      } else {
        set_value((_Receiver&&) __rcvr, __invoke((_Fun&&) __fun, (_As&&) __as...));
      }
    } else {
      try {
        stdexec::__set_value_invoke<true>((_Receiver&&) __rcvr, (_Fun&&) __fun, (_As&&) __as...);
      } catch (...) {
        set_error((_Receiver&&) __rcvr, std::current_exception());
      }
    }
  }

  template <class _Fun>
  struct _WITH_FUNCTION_ { };

  template <class... _Args>
  struct _WITH_ARGUMENTS_ { };

  inline constexpr __mstring __not_callable_diag =
    "The specified function is not callable with the arguments provided."__csz;

  template <__mstring _Context, __mstring _Diagnostic = __not_callable_diag>
  struct _NOT_CALLABLE_ { };

  template <__mstring _Context>
  struct __callable_error {
    template <class _Fun, class... _Args>
    using __f =     //
      __mexception< //
        _NOT_CALLABLE_<_Context>,
        _WITH_FUNCTION_<_Fun>,
        _WITH_ARGUMENTS_<_Args...>>;
  };

  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __non_throwing_ = __mbool<__nothrow_invocable<_Fun, _Args...>>;

  template <class _Tag, class _Fun, class _Sender, class _Env, class _Catch>
  using __with_error_invoke_t = //
    __if<
      __gather_completions_for<
        _Tag,
        _Sender,
        _Env,
        __mbind_front<__mtry_catch_q<__non_throwing_, _Catch>, _Fun>,
        __q<__mand>>,
      completion_signatures<>,
      __with_exception_ptr>;

  template <class _Fun, class... _Args>
    requires __invocable<_Fun, _Args...>
  using __set_value_invoke_t = //
    completion_signatures<
      __minvoke< __remove<void, __qf<set_value_t>>, __invoke_result_t<_Fun, _Args...>>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  namespace __then {
    inline constexpr __mstring __then_context = "In stdexec::then(Sender, Function)..."__csz;
    using __on_not_callable = __callable_error<__then_context>;

    template <class _Fun, class _CvrefSender, class _Env>
    using __completion_signatures_t = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __with_error_invoke_t< set_value_t, _Fun, _CvrefSender, _Env, __on_not_callable>,
        __mbind_front<__mtry_catch_q<__set_value_invoke_t, __on_not_callable>, _Fun>>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct then_t {
      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<then_t>((_Fun&&) __fun, (_Sender&&) __sndr));
      }

      template <__movable_value _Fun>
      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<then_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }

      using _Sender = __1;
      using _Fun = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          then_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Fun),
        tag_invoke_t(then_t, _Sender, _Fun)>;
    };

    struct __then_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&&, _Env&&) noexcept
        -> __completion_signatures_t<__decay_t<__data_of<_Sender>>, __child_of<_Sender>, _Env> {
        static_assert(sender_expr_for<_Sender, then_t>);
        return {};
      };

      static constexpr auto complete = //
        []<class _Tag, class... _Args>(
          __ignore,
          auto& __state,
          auto& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_value_t>) {
          stdexec::__set_value_invoke(std::move(__rcvr), std::move(__state), (_Args&&) __args...);
        } else {
          _Tag()(std::move(__rcvr), (_Args&&) __args...);
        }
      };
    };
  } // namespace __then

  using __then::then_t;
  inline constexpr then_t then{};

  template <>
  struct __sexpr_impl<then_t> : __then::__then_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_error]
  namespace __upon_error {
    inline constexpr __mstring __upon_error_context =
      "In stdexec::upon_error(Sender, Function)..."__csz;
    using __on_not_callable = __callable_error<__upon_error_context>;

    template <class _Fun, class _CvrefSender, class _Env>
    using __completion_signatures_t = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __with_error_invoke_t< set_error_t, _Fun, _CvrefSender, _Env, __on_not_callable>,
        __q<__compl_sigs::__default_set_value>,
        __mbind_front<__mtry_catch_q<__set_value_invoke_t, __on_not_callable>, _Fun>>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct upon_error_t {
      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<upon_error_t>((_Fun&&) __fun, (_Sender&&) __sndr));
      }

      template <__movable_value _Fun>
      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<upon_error_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }

      using _Sender = __1;
      using _Fun = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          upon_error_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Fun),
        tag_invoke_t(upon_error_t, _Sender, _Fun)>;
    };

    struct __upon_error_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&&, _Env&&) noexcept
        -> __completion_signatures_t<__decay_t<__data_of<_Sender>>, __child_of<_Sender>, _Env> {
            static_assert(sender_expr_for<_Sender, upon_error_t>);
            return {};
          };

      static constexpr auto complete = //
        []<class _Tag, class... _Args>(
          __ignore,
          auto& __state,
          auto& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_error_t>) {
          stdexec::__set_value_invoke(std::move(__rcvr), std::move(__state), (_Args&&) __args...);
        } else {
          _Tag()(std::move(__rcvr), (_Args&&) __args...);
        }
      };
    };
  } // namespace __upon_error

  using __upon_error::upon_error_t;
  inline constexpr upon_error_t upon_error{};

  template <>
  struct __sexpr_impl<upon_error_t> : __upon_error::__upon_error_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.upon_stopped]
  namespace __upon_stopped {
    inline constexpr __mstring __upon_stopped_context =
      "In stdexec::upon_stopped(Sender, Function)..."__csz;
    using __on_not_callable = __callable_error<__upon_stopped_context>;

    template <class _Fun, class _CvrefSender, class _Env>
    using __completion_signatures_t = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __with_error_invoke_t< set_stopped_t, _Fun, _CvrefSender, _Env, __on_not_callable>,
        __q<__compl_sigs::__default_set_value>,
        __q<__compl_sigs::__default_set_error>,
        __set_value_invoke_t<_Fun>>;

    ////////////////////////////////////////////////////////////////////////////////////////////////
    struct upon_stopped_t {
      template <sender _Sender, __movable_value _Fun>
        requires __callable<_Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<upon_stopped_t>((_Fun&&) __fun, (_Sender&&) __sndr));
      }

      template <__movable_value _Fun>
        requires __callable<_Fun>
      STDEXEC_ATTRIBUTE((always_inline)) //
        __binder_back<upon_stopped_t, _Fun>
        operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }

      using _Sender = __1;
      using _Fun = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          upon_stopped_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Fun),
        tag_invoke_t(upon_stopped_t, _Sender, _Fun)>;
    };

    struct __upon_stopped_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&&, _Env&&) noexcept
        -> __completion_signatures_t<__decay_t<__data_of<_Sender>>, __child_of<_Sender>, _Env> {
            static_assert(sender_expr_for<_Sender, upon_stopped_t>);
            return {};
          };

      static constexpr auto complete = //
        []<class _Tag, class... _Args>(
          __ignore,
          auto& __state,
          auto& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_stopped_t>) {
          stdexec::__set_value_invoke(std::move(__rcvr), std::move(__state), (_Args&&) __args...);
        } else {
          _Tag()(std::move(__rcvr), (_Args&&) __args...);
        }
      };
    };
  } // namespace __upon_stopped

  using __upon_stopped::upon_stopped_t;
  inline constexpr upon_stopped_t upon_stopped{};

  template <>
  struct __sexpr_impl<upon_stopped_t> : __upon_stopped::__upon_stopped_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.bulk]
  namespace __bulk {
    inline constexpr __mstring __bulk_context = "In stdexec::bulk(Sender, Shape, Function)..."__csz;
    using __on_not_callable = __callable_error<__bulk_context>;

    template <class _Shape, class _Fun>
    struct __data {
      _Shape __shape_;
      STDEXEC_ATTRIBUTE((no_unique_address)) _Fun __fun_;
      static constexpr auto __mbrs_ = __mliterals<&__data::__shape_, &__data::__fun_>();
    };
    template <class _Shape, class _Fun>
    __data(_Shape, _Fun) -> __data<_Shape, _Fun>;

    template <class _Ty>
    using __decay_ref = __decay_t<_Ty>&;

    template <class _CvrefSender, class _Env, class _Shape, class _Fun, class _Catch>
    using __with_error_invoke_t = //
      __if<
        __try_value_types_of_t<
          _CvrefSender,
          _Env,
          __transform<
            __q<__decay_ref>,
            __mbind_front<__mtry_catch_q<__non_throwing_, _Catch>, _Fun, _Shape>>,
          __q<__mand>>,
        completion_signatures<>,
        __with_exception_ptr>;

    template <class _CvrefSender, class _Env, class _Shape, class _Fun>
    using __completion_signatures = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __with_error_invoke_t<_CvrefSender, _Env, _Shape, _Fun, __on_not_callable>>;

    struct bulk_t {
      template <sender _Sender, integral _Shape, __movable_value _Fun>
      STDEXEC_ATTRIBUTE((host, device))
      auto operator()(_Sender&& __sndr, _Shape __shape, _Fun __fun) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<bulk_t>(__data{__shape, (_Fun&&) __fun}, (_Sender&&) __sndr));
      }

      template <integral _Shape, class _Fun>
      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<bulk_t, _Shape, _Fun> operator()(_Shape __shape, _Fun __fun) const {
        return {
          {},
          {},
          {(_Shape&&) __shape, (_Fun&&) __fun}
        };
      }

      // This describes how to use the pieces of a bulk sender to find
      // legacy customizations of the bulk algorithm.
      using _Sender = __1;
      using _Shape = __nth_member<0>(__0);
      using _Fun = __nth_member<1>(__0);
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          bulk_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(_Sender&)),
          _Sender,
          _Shape,
          _Fun),
        tag_invoke_t(bulk_t, _Sender, _Shape, _Fun)>;
    };

    struct __bulk_impl : __sexpr_defaults {
      template <class _Sender>
      using __fun_t = decltype(__decay_t<__data_of<_Sender>>::__fun_);

      template <class _Sender>
      using __shape_t = decltype(__decay_t<__data_of<_Sender>>::__shape_);

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&&, _Env&&) noexcept
        -> __completion_signatures<__child_of<_Sender>, _Env, __shape_t<_Sender>, __fun_t<_Sender>> {
            static_assert(sender_expr_for<_Sender, bulk_t>);
            return {};
          };

      static constexpr auto complete = //
        []<class _Tag, class... _Args>(
          __ignore,
          auto& __state,
          auto& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (std::same_as<_Tag, set_value_t>) {
          using __shape_t = decltype(__state.__shape_);
          if constexpr (noexcept(__state.__fun_(__shape_t{}, __args...))) {
            for (__shape_t __i{}; __i != __state.__shape_; ++__i) {
              __state.__fun_(__i, __args...);
            }
            _Tag()(std::move(__rcvr), (_Args&&) __args...);
          } else {
            try {
              for (__shape_t __i{}; __i != __state.__shape_; ++__i) {
                __state.__fun_(__i, __args...);
              }
              _Tag()(std::move(__rcvr), (_Args&&) __args...);
            } catch (...) {
              set_error(std::move(__rcvr), std::current_exception());
            }
          }
        } else {
          _Tag()(std::move(__rcvr), (_Args&&) __args...);
        }
      };
    };
  }

  using __bulk::bulk_t;
  inline constexpr bulk_t bulk{};

  template <>
  struct __sexpr_impl<bulk_t> : __bulk::__bulk_impl { };

  ////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.split]
  namespace __split {
    struct __on_stop_request {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _Receiver>
    auto __notify_visitor(_Receiver&& __rcvr) noexcept {
      return [&](const auto& __tupl) noexcept -> void {
        __apply(
          [&](auto __tag, const auto&... __args) noexcept -> void {
            __tag(std::move(__rcvr), __args...);
          },
          __tupl);
      };
    }

    struct __split_state_base : __immovable {
      using __notify_fn = void(__split_state_base*) noexcept;

      __split_state_base* __next_{};
      __notify_fn* __notify_{};
    };

    template <class _Sender, class _Receiver>
    struct __split_state
      : __split_state_base
      , __enable_receiver_from_this<_Sender, _Receiver> {
      using __shared_state_ptr = __decay_t<__data_of<_Sender>>;
      using __on_stop_cb = //
        typename stop_token_of_t<env_of_t<_Receiver>&>::template callback_type<__on_stop_request>;

      explicit __split_state(_Sender&& __sndr) noexcept
        : __split_state_base{{}, nullptr, __notify}
        , __on_stop_()
        , __shared_state_(__sndr.apply((_Sender&&) __sndr, __detail::__get_data())) {
      }

      static void __notify(__split_state_base* __self) noexcept {
        __split_state* __op = static_cast<__split_state*>(__self);
        __op->__on_stop_.reset();
        std::visit(__split::__notify_visitor(__op->__receiver()), __op->__shared_state_->__data_);
      }

      std::optional<__on_stop_cb> __on_stop_;
      __shared_state_ptr __shared_state_;
    };

    template <class _CvrefSender, class _Env>
    struct __sh_state;

    template <class _BaseEnv>
    using __env_t = //
      __make_env_t<
        _BaseEnv, // BUGBUG NOT TO SPEC
        __with<get_stop_token_t, in_place_stop_token>>;

    template <class _CvrefSenderId, class _EnvId>
    struct __receiver {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        __sh_state<_CvrefSender, _Env>& __sh_state_;

        using receiver_concept = receiver_t;
        using __id = __receiver;

        template <__completion_tag _Tag, class... _As>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          __sh_state<_CvrefSender, _Env>& __state = __self.__sh_state_;

          try {
            using __tuple_t = __decayed_tuple<_Tag, _As...>;
            __state.__data_.template emplace<__tuple_t>(__tag, (_As&&) __as...);
          } catch (...) {
            using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
            __state.__data_.template emplace<__tuple_t>(set_error, std::current_exception());
          }
          __state.__notify();
        }

        friend const __env_t<_Env>& tag_invoke(get_env_t, const __t& __self) noexcept {
          return __self.__sh_state_.__env_;
        }
      };
    };

    template <class _CvrefSender, class _Env = empty_env>
    struct __sh_state {
      using __t = __sh_state;
      using __id = __sh_state;

      template <class... _Ts>
      using __bind_tuples = //
        __mbind_front_q<
          __variant,
          std::tuple<set_stopped_t>, // Initial state of the variant is set_stopped
          std::tuple<set_error_t, std::exception_ptr>,
          _Ts...>;

      using __bound_values_t = //
        __value_types_of_t<
          _CvrefSender,
          __env_t<_Env>,
          __mbind_front_q<__decayed_tuple, set_value_t>,
          __q<__bind_tuples>>;

      using __variant_t = //
        __error_types_of_t<
          _CvrefSender,
          __env_t<_Env>,
          __transform< __mbind_front_q<__decayed_tuple, set_error_t>, __bound_values_t>>;

      using __receiver_t = stdexec::__t<__receiver<__cvref_id<_CvrefSender>, stdexec::__id<_Env>>>;

      in_place_stop_source __stop_source_{};
      __variant_t __data_;
      std::atomic<void*> __head_{nullptr};
      __env_t<_Env> __env_;
      connect_result_t<_CvrefSender, __receiver_t> __op_state2_;

      explicit __sh_state(_CvrefSender&& __sndr, _Env __env = {})
        : __env_(__make_env((_Env&&) __env, __mkprop(__stop_source_.get_token(), get_stop_token)))
        , __op_state2_(connect((_CvrefSender&&) __sndr, __receiver_t{*this})) {
      }

      void __notify() noexcept {
        void* const __completion_state = static_cast<void*>(this);
        void* __old = __head_.exchange(__completion_state, std::memory_order_acq_rel);
        __split_state_base* __state = static_cast<__split_state_base*>(__old);

        while (__state != nullptr) {
          __split_state_base* __next = __state->__next_;
          __state->__notify_(__state);
          __state = __next;
        }
      }
    };

    template <class... _Tys>
    using __set_value_t = completion_signatures<set_value_t(const __decay_t<_Tys>&...)>;

    template <class _Ty>
    using __set_error_t = completion_signatures<set_error_t(const __decay_t<_Ty>&)>;

    template <class _CvrefSenderId, class _EnvId>
    using __completions_t = //
      __try_make_completion_signatures<
        // NOT TO SPEC:
        // See https://github.com/brycelelbach/wg21_p2300_execution/issues/26
        __cvref_t<_CvrefSenderId>,
        __env_t<__t<_EnvId>>,
        completion_signatures<
          set_error_t(const std::exception_ptr&),
          set_stopped_t()>, // NOT TO SPEC
        __q<__set_value_t>,
        __q<__set_error_t>>;

    struct __split_t { };

    struct split_t : __split_t {
      template <sender _Sender>
        requires sender_in<_Sender, empty_env> && __decay_copyable<env_of_t<_Sender>>
      auto operator()(_Sender&& __sndr) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(__domain, __make_sexpr<split_t>(__(), (_Sender&&) __sndr));
      }

      template <sender _Sender, class _Env>
        requires sender_in<_Sender, _Env> && __decay_copyable<env_of_t<_Sender>>
      auto operator()(_Sender&& __sndr, _Env&& __env) const {
        auto __domain = __get_late_domain(__sndr, __env);
        return stdexec::transform_sender(
          __domain, __make_sexpr<split_t>(__(), (_Sender&&) __sndr), (_Env&&) __env);
      }

      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<split_t> operator()() const {
        return {{}, {}, {}};
      }

      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<
          tag_invoke_t(
            split_t,
            get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
            _Sender),
          tag_invoke_t(split_t, _Sender)>;

      template <class _Sender, class... _Env>
      static auto transform_sender(_Sender&& __sndr, _Env... __env) {
        return __sexpr_apply(
          (_Sender&&) __sndr, [&]<class _Child>(__ignore, __ignore, _Child&& __child) {
            using __sh_state_t = __sh_state<_Child, _Env...>;
            auto __sh_state = std::make_shared<__sh_state_t>(
              (_Child&&) __child, std::move(__env)...);
            return __make_sexpr<__split_t>(std::move(__sh_state));
          });
      }
    };

    struct __split_impl : __sexpr_defaults {
      static constexpr auto get_state = //
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver&) noexcept
        -> __split_state<_Sender, _Receiver> {
        static_assert(sender_expr_for<_Sender, __split_t>);
        return __split_state<_Sender, _Receiver>{(_Sender&&) __sndr};
      };

      static constexpr auto start = //
        []<class _Sender, class _Receiver>(
          __split_state<_Sender, _Receiver>& __state,
          _Receiver& __rcvr) noexcept -> void {
        auto* __shared_state = __state.__shared_state_.get();
        std::atomic<void*>& __head = __shared_state->__head_;
        void* const __completion_state = static_cast<void*>(__shared_state);
        void* __old = __head.load(std::memory_order_acquire);

        if (__old != __completion_state) {
          __state.__on_stop_.emplace(
            get_stop_token(stdexec::get_env(__rcvr)),
            __on_stop_request{__shared_state->__stop_source_});
        }

        do {
          if (__old == __completion_state) {
            __state.__notify(&__state);
            return;
          }
          __state.__next_ = static_cast<__split_state_base*>(__old);
        } while (!__head.compare_exchange_weak(
          __old,
          static_cast<void*>(&__state),
          std::memory_order_release,
          std::memory_order_acquire));

        if (__old == nullptr) {
          // the inner sender isn't running
          if (__shared_state->__stop_source_.stop_requested()) {
            // 1. resets __head to completion state
            // 2. notifies waiting threads
            // 3. propagates "stopped" signal to `out_r'`
            __shared_state->__notify();
          } else {
            stdexec::start(__shared_state->__op_state2_);
          }
        }
      };

      static constexpr auto __get_completion_signatures_fn =       //
        []<class _ShState>(auto, const std::shared_ptr<_ShState>&) //
        -> __mapply<__q<__completions_t>, __id<_ShState>> {
        return {};
      };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _OtherEnv>(_Self&&, _OtherEnv&&) noexcept
        -> __call_result_t<__sexpr_apply_t, _Self, __mtypeof<__get_completion_signatures_fn>> {
        static_assert(sender_expr_for<_Self, __split_t>);
        return {};
      };
    };
  } // namespace __split

  using __split::split_t;
  inline constexpr split_t split{};

  template <>
  struct __sexpr_impl<__split::__split_t> : __split::__split_impl { };

  template <>
  struct __sexpr_impl<split_t> : __split::__split_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.ensure_started]
  namespace __ensure_started {
    template <class _BaseEnv>
    using __env_t = //
      __make_env_t<
        _BaseEnv, // NOT TO SPEC
        __with<get_stop_token_t, in_place_stop_token>>;

    template <class _CvrefSenderId, class _EnvId>
    struct __sh_state;

    template <class _CvrefSenderId, class _EnvId = __id<empty_env>>
    struct __receiver {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      class __t {
        __intrusive_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state_;

       public:
        using receiver_concept = receiver_t;
        using __id = __receiver;

        explicit __t(stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __shared_state) noexcept
          : __shared_state_(__shared_state.__intrusive_from_this()) {
        }

        template <__completion_tag _Tag, class... _As>
        friend void tag_invoke(_Tag __tag, __t&& __self, _As&&... __as) noexcept {
          stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>& __state = *__self.__shared_state_;

          try {
            using __tuple_t = __decayed_tuple<_Tag, _As...>;
            __state.__data_.template emplace<__tuple_t>(__tag, (_As&&) __as...);
          } catch (...) {
            using __tuple_t = __decayed_tuple<set_error_t, std::exception_ptr>;
            __state.__data_.template emplace<__tuple_t>(set_error, std::current_exception());
          }

          __state.__notify();
          __self.__shared_state_.reset();
        }

        friend const __env_t<_Env>& tag_invoke(get_env_t, const __t& __self) noexcept {
          return __self.__shared_state_->__env_;
        }
      };
    };

    struct __operation_base {
      using __notify_fn = void(__operation_base*) noexcept;
      __notify_fn* __notify_{};
    };

    template <class _CvrefSenderId, class _EnvId = __id<empty_env>>
    struct __sh_state {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;

      struct __t : __enable_intrusive_from_this<__t> {
        using __id = __sh_state;

        template <class... _Ts>
        using __bind_tuples = //
          __mbind_front_q<
            __variant,
            std::tuple<set_stopped_t>, // Initial state of the variant is set_stopped
            std::tuple<set_error_t, std::exception_ptr>,
            _Ts...>;

        using __bound_values_t = //
          __value_types_of_t<
            _CvrefSender,
            __env_t<_Env>,
            __mbind_front_q<__decayed_tuple, set_value_t>,
            __q<__bind_tuples>>;

        using __variant_t = //
          __error_types_of_t<
            _CvrefSender,
            __env_t<_Env>,
            __transform< __mbind_front_q<__decayed_tuple, set_error_t>, __bound_values_t>>;

        using __receiver_t = stdexec::__t<__receiver<_CvrefSenderId, _EnvId>>;

        __variant_t __data_;
        in_place_stop_source __stop_source_{};

        std::atomic<void*> __op_state1_{nullptr};
        __env_t<_Env> __env_;
        connect_result_t<_CvrefSender, __receiver_t> __op_state2_;

        explicit __t(_CvrefSender&& __sndr, _Env __env = {})
          : __env_(__make_env((_Env&&) __env, __mkprop(__stop_source_.get_token(), get_stop_token)))
          , __op_state2_(connect((_CvrefSender&&) __sndr, __receiver_t{*this})) {
          start(__op_state2_);
        }

        void __notify() noexcept {
          void* const __completion_state = static_cast<void*>(this);
          void* const __old = __op_state1_.exchange(__completion_state, std::memory_order_acq_rel);
          if (__old != nullptr) {
            auto* __op = static_cast<__operation_base*>(__old);
            __op->__notify_(__op);
          }
        }

        void __detach() noexcept {
          __stop_source_.request_stop();
        }
      };
    };

    template <class _CvrefSenderId, class _EnvId, class _ReceiverId>
    struct __operation {
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Env = stdexec::__t<_EnvId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      class __t : public __operation_base {
        struct __on_stop_request {
          in_place_stop_source& __stop_source_;

          void operator()() noexcept {
            __stop_source_.request_stop();
          }
        };

        using __on_stop = //
          std::optional< typename stop_token_of_t< env_of_t<_Receiver>&>::template callback_type<
            __on_stop_request>>;

        _Receiver __rcvr_;
        __on_stop __on_stop_{};
        __intrusive_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state_;

       public:
        using __id = __operation;

        __t(                                                                                //
          _Receiver __rcvr,                                                                 //
          __intrusive_ptr<stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>> __shared_state) //
          noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          : __operation_base{__notify}
          , __rcvr_((_Receiver&&) __rcvr)
          , __shared_state_(std::move(__shared_state)) {
        }

        ~__t() {
          // Check to see if this operation was ever started. If not,
          // detach the (potentially still running) operation:
          if (nullptr == __shared_state_->__op_state1_.load(std::memory_order_acquire)) {
            __shared_state_->__detach();
          }
        }

        STDEXEC_IMMOVABLE(__t);

        static void __notify(__operation_base* __self) noexcept {
          __t* __op = static_cast<__t*>(__self);
          __op->__on_stop_.reset();

          std::visit(
            [&](auto& __tupl) noexcept -> void {
              __apply(
                [&](auto __tag, auto&... __args) noexcept -> void {
                  __tag((_Receiver&&) __op->__rcvr_, std::move(__args)...);
                },
                __tupl);
            },
            __op->__shared_state_->__data_);
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::__t<__sh_state<_CvrefSenderId, _EnvId>>* __shared_state =
            __self.__shared_state_.get();
          std::atomic<void*>& __op_state1 = __shared_state->__op_state1_;
          void* const __completion_state = static_cast<void*>(__shared_state);
          void* const __old = __op_state1.load(std::memory_order_acquire);
          if (__old == __completion_state) {
            __self.__notify(&__self);
          } else {
            // register stop callback:
            __self.__on_stop_.emplace(
              get_stop_token(get_env(__self.__rcvr_)),
              __on_stop_request{__shared_state->__stop_source_});
            // Check if the stop_source has requested cancellation
            if (__shared_state->__stop_source_.stop_requested()) {
              // Stop has already been requested. Don't bother starting
              // the child operations.
              stdexec::set_stopped((_Receiver&&) __self.__rcvr_);
            } else {
              // Otherwise, the inner source hasn't notified completion.
              // Set this operation as the __op_state1 so it's notified.
              void* __old = nullptr;
              if (!__op_state1.compare_exchange_weak(
                    __old, &__self, std::memory_order_release, std::memory_order_acquire)) {
                // We get here when the task completed during the execution
                // of this function. Complete the operation synchronously.
                STDEXEC_ASSERT(__old == __completion_state);
                __self.__notify(&__self);
              }
            }
          }
        }
      };
    };

    template <class _ShState>
    struct __data {
      __data(__intrusive_ptr<_ShState> __sh_state) noexcept
        : __sh_state_(std::move(__sh_state)) {
      }

      __data(__data&&) = default;
      __data& operator=(__data&&) = default;

      ~__data() {
        if (__sh_state_ != nullptr) {
          // detach from the still-running operation.
          // NOT TO SPEC: This also requests cancellation.
          __sh_state_->__detach();
        }
      }

      __intrusive_ptr<_ShState> __sh_state_;
    };

    template <class... _Tys>
    using __set_value_t = completion_signatures<set_value_t(__decay_t<_Tys>&&...)>;

    template <class _Ty>
    using __set_error_t = completion_signatures<set_error_t(__decay_t<_Ty>&&)>;

    template <class _CvrefSenderId, class _EnvId>
    using __completions_t = //
      __try_make_completion_signatures<
        // NOT TO SPEC:
        // See https://github.com/brycelelbach/wg21_p2300_execution/issues/26
        __cvref_t<_CvrefSenderId>,
        __env_t<__t<_EnvId>>,
        completion_signatures<
          set_error_t(std::exception_ptr&&),
          set_stopped_t()>, // NOT TO SPEC
        __q<__set_value_t>,
        __q<__set_error_t>>;

    template <class _Receiver>
    auto __connect_fn_(_Receiver& __rcvr) noexcept {
      return [&]<class _ShState>(auto, __data<_ShState> __dat) //
             noexcept(__nothrow_decay_copyable<_Receiver>)
               -> __t<__mapply<__mbind_back_q<__operation, __id<_Receiver>>, __id<_ShState>>> {
               return {(_Receiver&&) __rcvr, std::move(__dat.__sh_state_)};
             };
    }

    template <class _Receiver>
    auto __connect_fn(_Receiver& __rcvr) noexcept -> decltype(__connect_fn_(__rcvr)) {
      return __connect_fn_(__rcvr);
    }

    inline auto __get_completion_signatures_fn() noexcept {
      return []<class _ShState>(auto, const __data<_ShState>&) //
             -> __mapply<__q<__completions_t>, __id<_ShState>> {
        return {};
      };
    }

    struct __ensure_started_t { };

    struct __ensure_started_impl : __sexpr_defaults {
      static constexpr auto connect = //
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver __rcvr) noexcept(
          __nothrow_callable<
            __sexpr_apply_t,
            _Self,
            decltype(__connect_fn(__declval<_Receiver&>()))>)
        -> __call_result_t< __sexpr_apply_t, _Self, decltype(__connect_fn(__declval<_Receiver&>()))> {
        static_assert(sender_expr_for<_Self, __ensure_started_t>);
        return __sexpr_apply((_Self&&) __self, __connect_fn(__rcvr));
      };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _OtherEnv>(_Self&&, _OtherEnv&&) noexcept
        -> __call_result_t<__sexpr_apply_t, _Self, __result_of<__get_completion_signatures_fn>> {
        static_assert(sender_expr_for<_Self, __ensure_started_t>);
        return {};
      };
    };

    struct ensure_started_t {
      template <sender _Sender>
        requires sender_in<_Sender, empty_env> && __decay_copyable<env_of_t<_Sender>>
      auto operator()(_Sender&& __sndr) const {
        if constexpr (sender_expr_for<_Sender, __ensure_started_t>) {
          return (_Sender&&) __sndr;
        } else {
          auto __domain = __get_early_domain(__sndr);
          return stdexec::transform_sender(
            __domain, __make_sexpr<ensure_started_t>(__(), (_Sender&&) __sndr));
        }
        STDEXEC_UNREACHABLE();
      }

      template <sender _Sender, class _Env>
        requires sender_in<_Sender, _Env> && __decay_copyable<env_of_t<_Sender>>
      auto operator()(_Sender&& __sndr, _Env&& __env) const {
        if constexpr (sender_expr_for<_Sender, __ensure_started_t>) {
          return (_Sender&&) __sndr;
        } else {
          auto __domain = __get_late_domain(__sndr, __env);
          return stdexec::transform_sender(
            __domain, __make_sexpr<ensure_started_t>(__(), (_Sender&&) __sndr), (_Env&&) __env);
        }
        STDEXEC_UNREACHABLE();
      }

      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<ensure_started_t> operator()() const {
        return {{}, {}, {}};
      }

      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<
          tag_invoke_t(
            ensure_started_t,
            get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
            _Sender),
          tag_invoke_t(ensure_started_t, _Sender)>;

      template <class _CvrefSender, class... _Env>
      using __receiver_t =
        stdexec::__t<__meval<__receiver, __cvref_id<_CvrefSender>, __id<_Env>...>>;

      template <class _Sender, class... _Env>
        requires sender_to<__child_of<_Sender>, __receiver_t<__child_of<_Sender>, _Env...>>
      static auto transform_sender(_Sender&& __sndr, _Env... __env) {
        return __sexpr_apply(
          (_Sender&&) __sndr, [&]<class _Child>(__ignore, __ignore, _Child&& __child) {
            using __sh_state_t = __t<__sh_state<__cvref_id<_Child>, __id<_Env>...>>;
            auto __sh_state = __make_intrusive<__sh_state_t>(
              (_Child&&) __child, std::move(__env)...);
            return __make_sexpr<__ensure_started_t>(__data{std::move(__sh_state)});
          });
      }
    };
  } // namespace __ensure_started

  using __ensure_started::ensure_started_t;
  inline constexpr ensure_started_t ensure_started{};

  template <>
  struct __sexpr_impl<__ensure_started::__ensure_started_t>
    : __ensure_started::__ensure_started_impl { };

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_EDG(not_used_in_partial_spec_arg_list)

  /////////////////////////////////////////////////////////////////////////////
  // a receiver adaptor that augments its environment
  namespace __detail {
    template <auto _ReceiverPtr, auto... _EnvFns>
    struct __receiver_with;

    template <class _Operation, class _Receiver, _Receiver _Operation::*_ReceiverPtr, auto... _EnvFns>
    struct __receiver_with<_ReceiverPtr, _EnvFns...> {
      struct __t : receiver_adaptor<__t> {
        using __id = __receiver_with;
        using __env_t =
          __env::__env_join_t<__result_of<_EnvFns, _Operation*>..., env_of_t<_Receiver>>;

        _Operation* __op_state_;

        _Receiver&& base() && noexcept {
          return static_cast<_Receiver&&>(__op_state_->*_ReceiverPtr);
        }

        __env_t get_env() const noexcept {
          return __join_env(_EnvFns(__op_state_)..., stdexec::get_env(__op_state_->*_ReceiverPtr));
        }
      };
    };
  } // namespace __detail

  STDEXEC_PRAGMA_POP()

  //////////////////////////////////////////////////////////////////////////////
  // [exec.let]
  namespace __let {
    template <class _SetTag, class _Domain = dependent_domain>
    struct __let_t;

    template <class _Set>
    struct __on_not_callable_ {
      using __t = __callable_error<"In stdexec::let_value(Sender, Function)..."__csz>;
    };

    template <>
    struct __on_not_callable_<set_error_t> {
      using __t = __callable_error<"In stdexec::let_error(Sender, Function)..."__csz>;
    };

    template <>
    struct __on_not_callable_<set_stopped_t> {
      using __t = __callable_error<"In stdexec::let_stopped(Sender, Function)..."__csz>;
    };

    template <class _Set>
    using __on_not_callable = __t<__on_not_callable_<_Set>>;

    template <class _Tp>
    using __decay_ref = __decay_t<_Tp>&;

    // A metafunction that computes the result sender type for a given set of argument types
    template <class _Fun, class _Set>
    using __result_sender_t = //
      __transform<
        __q<__decay_ref>,
        __mbind_front<__mtry_catch_q<__call_result_t, __on_not_callable<_Set>>, _Fun>>;

    // FUTURE: when we have a scheduler query for "always completes inline",
    // then we can use that instead of hard-coding `__inln::__scheduler` here.
    template <class _Scheduler>
    concept __unknown_context = __one_of<_Scheduler, __none_such, __inln::__scheduler>;

    template <class _Receiver, class _Scheduler>
    struct __receiver_with_sched {
      using receiver_concept = receiver_t;
      _Receiver __rcvr_;
      _Scheduler __sched_;

      template <__completion_tag _Tag, same_as<__receiver_with_sched> _Self, class... _As>
      friend void tag_invoke(_Tag, _Self&& __self, _As&&... __as) noexcept {
        _Tag()((_Receiver&&) __self.__rcvr_, (_As&&) __as...);
      }

      template <same_as<get_env_t> _Tag>
      friend auto tag_invoke(_Tag, const __receiver_with_sched& __self) noexcept {
        return __join_env(
          __mkprop(__self.__sched_, get_scheduler), __mkprop(get_domain), get_env(__self.__rcvr_));
      }
    };

    template <class _Receiver, class _Scheduler>
    __receiver_with_sched(_Receiver, _Scheduler) -> __receiver_with_sched<_Receiver, _Scheduler>;

    // If the input sender knows its completion scheduler, make it the current scheduler
    // in the environment seen by the result sender.
    template <class _Env, class _Scheduler>
    using __result_env_t = __if_c<
      __unknown_context<_Scheduler>,
      _Env,
      __env::__env_join_t<
        __env::__prop<_Scheduler(get_scheduler_t)>,
        __env::__prop<void(get_domain_t)>,
        _Env>>;

    // The receiver that gets connected to the result sender is the input receiver,
    // possibly augmented with the input sender's completion scheduler (which is
    // where the result sender will be started).
    template <class _Receiver, class _Scheduler>
    using __result_receiver_t =
      __if_c< __unknown_context<_Scheduler>, _Receiver, __receiver_with_sched<_Receiver, _Scheduler>>;

    template <class _Receiver, class _Fun, class _Set, class _Sched>
    using __op_state_for = //
      __mcompose<
        __mbind_back_q<connect_result_t, __result_receiver_t<_Receiver, _Sched>>,
        __result_sender_t<_Fun, _Set>>;

    template <class _Set, class _Sig>
    struct __tfx_signal_fn {
      template <class, class, class>
      using __f = completion_signatures<_Sig>;
    };

    template <class _Set, class... _Args>
    struct __tfx_signal_fn<_Set, _Set(_Args...)> {
      template <class _Env, class _Fun, class _Sched>
      using __f = //
        __try_make_completion_signatures<
          __minvoke<__result_sender_t<_Fun, _Set>, _Args...>,
          __result_env_t<_Env, _Sched>,
          // because we don't know if connect-ing the result sender will throw:
          completion_signatures<set_error_t(std::exception_ptr)>>;
    };

    // `_Sched` is the input sender's completion scheduler, or __none_such if it doesn't have one.
    template <class _Env, class _Fun, class _Set, class _Sched, class _Sig>
    using __tfx_signal_t = __minvoke<__tfx_signal_fn<_Set, _Sig>, _Env, _Fun, _Sched>;

    template <class _Sender, class _Set>
    using __completion_sched =
      __query_result_or_t<get_completion_scheduler_t<_Set>, env_of_t<_Sender>, __none_such>;

    template <class _CvrefSender, class _Env, class _LetTag, class _Fun>
    using __completions = //
      __mapply<
        __transform<
          __mbind_front_q<
            __tfx_signal_t,
            _Env,
            _Fun,
            __t<_LetTag>,
            __completion_sched<_CvrefSender, __t<_LetTag>>>,
          __q<__concat_completion_signatures_t> >,
        __completion_signatures_of_t<_CvrefSender, _Env>>;

    // Compute all the domains of all the result senders and make sure they're all the same
    template <class _SetTag, class _Child, class _Fun, class _Env>
    using __result_domain_t = __gather_completions_for<
      _SetTag,
      _Child,
      _Env,
      __mtry_catch<
        __transform<__q<__decay_ref>, __mbind_front_q<__call_result_t, _Fun>>,
        __on_not_callable<_SetTag>>,
      __q<__domain::__common_domain_t>>;

    template <class _LetTag, class _Env>
    auto __mk_transform_env_fn(const _Env& __env) noexcept {
      using _SetTag = __t<_LetTag>;
      return [&]<class _Fun, sender_in<_Env> _Child>(
               __ignore, _Fun&&, _Child&& __child) -> decltype(auto) {
        using _Scheduler = __completion_sched<_Child, _SetTag>;
        if constexpr (__unknown_context<_Scheduler>) {
          return (__env);
        } else {
          return __join_env(
            __mkprop(get_completion_scheduler<_SetTag>(stdexec::get_env(__child)), get_scheduler),
            __mkprop(get_domain),
            __env);
        }
        STDEXEC_UNREACHABLE();
      };
    }

    template <class _LetTag, class _Env>
    auto __mk_transform_sender_fn(const _Env&) noexcept {
      using _SetTag = __t<_LetTag>;
      return []<class _Fun, sender_in<_Env> _Child>(__ignore, _Fun&& __fun, _Child&& __child) {
        using _Domain = __result_domain_t<_SetTag, _Child, _Fun, _Env>;
        static_assert(__none_of<_Domain, __none_such, dependent_domain>);
        return __make_sexpr<__let_t<_SetTag, _Domain>>((_Fun&&) __fun, (_Child&&) __child);
      };
    }

    template <class _Receiver, class _Fun, class _Set, class _Sched, class... _Tuples>
    struct __let_state {
      using __fun_t = _Fun;
      using __sched_t = _Sched;

      using __result_variant = std::variant<std::monostate, _Tuples...>;

      using __op_state_variant = //
        __minvoke<
          __transform<
            __uncurry<__op_state_for<_Receiver, _Fun, _Set, _Sched>>,
            __nullable_variant_t>,
          _Tuples...>;

      decltype(auto) __get_result_receiver(_Receiver&& __rcvr) {
        if constexpr (__unknown_context<_Sched>) {
          return (_Receiver&&) __rcvr;
        } else {
          return __receiver_with_sched{(_Receiver&&) __rcvr, this->__sched_};
        }
      }

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS _Fun __fun_;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS _Sched __sched_;
      __result_variant __args_;
      __op_state_variant __op_state3_;
    };

    template <class _SetTag, class _Domain>
    struct __let_t {
      using __domain_t = _Domain;
      using __t = _SetTag;

      template <sender _Sender, __movable_value _Fun>
      auto operator()(_Sender&& __sndr, _Fun __fun) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<__let_t<_SetTag>>((_Fun&&) __fun, (_Sender&&) __sndr));
      }

      template <class _Fun>
      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<__let_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }

      using _Sender = __1;
      using _Function = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          __let_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender,
          _Function),
        tag_invoke_t(__let_t, _Sender, _Function)>;

      template <sender_expr_for<__let_t<_SetTag>> _Sender, class _Env>
      static decltype(auto) transform_env(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply((_Sender&&) __sndr, __mk_transform_env_fn<__let_t<_SetTag>>(__env));
      }

      template <sender_expr_for<__let_t<_SetTag>> _Sender, class _Env>
        requires same_as<__early_domain_of_t<_Sender>, dependent_domain>
      static decltype(auto) transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply((_Sender&&) __sndr, __mk_transform_sender_fn<__let_t<_SetTag>>(__env));
      }
    };

    template <class _SetTag, class _Domain>
    struct __let_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Child>(__ignore, const _Child& __child) noexcept {
          return __join_env(__mkprop(_Domain(), get_domain), stdexec::get_env(__child));
        };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept
        -> __completions<__child_of<_Self>, _Env, __let_t<_SetTag, _Domain>, __data_of<_Self>> {
        static_assert(sender_expr_for<_Self, __let_t<_SetTag, _Domain>>);
        return {};
      };

      static constexpr auto get_state = //
        []<class _Sender, class _Receiver>(_Sender&& __sndr, _Receiver& __rcvr) {
          static_assert(sender_expr_for<_Sender, __let_t<_SetTag, _Domain>>);
          using _Fun = __data_of<_Sender>;
          using _Sched = __completion_sched<_Sender, _SetTag>;
          using __mk_let_state = __mbind_front_q<__let_state, _Receiver, _Fun, _SetTag, _Sched>;

          using __let_state_t = __gather_completions_for<
            _SetTag,
            __child_of<_Sender>,
            env_of_t<_Receiver>,
            __q<__decayed_tuple>,
            __mk_let_state>;

          _Sched __sched = query_or(
            get_completion_scheduler<_SetTag>, stdexec::get_env(__sndr), __none_such());
          return __let_state_t{__sndr.apply((_Sender&&) __sndr, __detail::__get_data()), __sched};
        };

      template <class _State, class _Receiver, class... _As>
      static void __bind(_State&& __state, _Receiver&& __rcvr, _As&&... __as) noexcept {
        try {
          using __fun_t = typename _State::__fun_t;
          using __sched_t = typename _State::__sched_t;
          using __tuple_t = __decayed_tuple<_As...>;
          using __op_state_t =
            __minvoke<__op_state_for<_Receiver, __fun_t, _SetTag, __sched_t>, _As...>;
          auto& __args = __state.__args_.template emplace<__tuple_t>((_As&&) __as...);
          auto& __op = __state.__op_state3_.template emplace<__op_state_t>(__conv{[&] {
            return stdexec::connect(
              __apply(std::move(__state.__fun_), __args),
              __state.__get_result_receiver((_Receiver&&) __rcvr));
          }});
          stdexec::start(__op);
        } catch (...) {
          set_error(std::move(__rcvr), std::current_exception());
        }
      }

      static constexpr auto complete = //
        []<class _State, class _Receiver, class _Tag, class... _As>(
          __ignore,
          _State& __state,
          _Receiver& __rcvr,
          _Tag,
          _As&&... __as) noexcept -> void {
        if constexpr (std::same_as<_Tag, _SetTag>) {
          __bind((_State&&) __state, (_Receiver&&) __rcvr, (_As&&) __as...);
        } else {
          _Tag()((_Receiver&&) __rcvr, (_As&&) __as...);
        }
      };
    };
  } // namespace __let

  using let_value_t = __let::__let_t<set_value_t>;
  inline constexpr let_value_t let_value{};

  using let_error_t = __let::__let_t<set_error_t>;
  inline constexpr let_error_t let_error{};

  using let_stopped_t = __let::__let_t<set_stopped_t>;
  inline constexpr let_stopped_t let_stopped{};

  template <class _SetTag, class _Domain>
  struct __sexpr_impl<__let::__let_t<_SetTag, _Domain>> : __let::__let_impl<_SetTag, _Domain> { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_optional]
  // [execution.senders.adaptors.stopped_as_error]
  namespace __stopped_as_xxx {
    struct stopped_as_optional_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        return __make_sexpr<stopped_as_optional_t>(__(), (_Sender&&) __sndr);
      }

      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<stopped_as_optional_t> operator()() const noexcept {
        return {};
      }
    };

    struct __stopped_as_optional_impl : __sexpr_defaults {
      template <class... _Tys>
        requires(sizeof...(_Tys) == 1)
      using __set_value_t = completion_signatures<set_value_t(std::optional<__decay_t<_Tys>>...)>;

      template <class _Ty>
      using __set_error_t = completion_signatures<set_error_t(_Ty)>;

      static constexpr auto get_completion_signatures =       //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept //
        -> make_completion_signatures<
          __child_of<_Self>,
          _Env,
          completion_signatures<set_error_t(std::exception_ptr)>,
          __set_value_t,
          __set_error_t,
          completion_signatures<>> {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        return {};
      };

      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(_Self&&, _Receiver&) noexcept
        requires __single_typed_sender<__child_of<_Self>, env_of_t<_Receiver>>
      {
        static_assert(sender_expr_for<_Self, stopped_as_optional_t>);
        using _Value = __decay_t<__single_sender_value_t<__child_of<_Self>, env_of_t<_Receiver>>>;
        return __mtype<_Value>();
      };

      static constexpr auto complete = //
        []<class _State, class _Receiver, __completion_tag _Tag, class... _Args>(
          __ignore,
          _State&,
          _Receiver& __rcvr,
          _Tag,
          _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_Tag, set_value_t>) {
          try {
            static_assert(constructible_from<__t<_State>, _Args...>);
            stdexec::set_value(
              (_Receiver&&) __rcvr, std::optional<__t<_State>>{(_Args&&) __args...});
          } catch (...) {
            stdexec::set_error((_Receiver&&) __rcvr, std::current_exception());
          }
        } else if constexpr (same_as<_Tag, set_error_t>) {
          stdexec::set_error((_Receiver&&) __rcvr, (_Args&&) __args...);
        } else {
          stdexec::set_value((_Receiver&&) __rcvr, std::optional<__t<_State>>{std::nullopt});
        }
      };
    };

    struct stopped_as_error_t {
      template <sender _Sender, __movable_value _Error>
      auto operator()(_Sender&& __sndr, _Error __err) const {
        return (_Sender&&) __sndr
             | let_stopped([__err2 = (_Error&&) __err]() mutable //
                           noexcept(std::is_nothrow_move_constructible_v<_Error>) {
                             return just_error((_Error&&) __err2);
                           });
      }

      template <__movable_value _Error>
      STDEXEC_ATTRIBUTE((always_inline)) //
      auto operator()(_Error __err) const -> __binder_back<stopped_as_error_t, _Error> {
        return {{}, {}, {(_Error&&) __err}};
      }
    };
  } // namespace __stopped_as_xxx

  using __stopped_as_xxx::stopped_as_optional_t;
  inline constexpr stopped_as_optional_t stopped_as_optional{};
  using __stopped_as_xxx::stopped_as_error_t;
  inline constexpr stopped_as_error_t stopped_as_error{};

  template <>
  struct __sexpr_impl<stopped_as_optional_t> : __stopped_as_xxx::__stopped_as_optional_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  namespace __loop {
    class run_loop;

    struct __task : __immovable {
      __task* __next_ = this;

      union {
        void (*__execute_)(__task*) noexcept;
        __task* __tail_;
      };

      void __execute() noexcept {
        (*__execute_)(this);
      }
    };

    template <class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __task {
        using __id = __operation;

        run_loop* __loop_;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Receiver __rcvr_;

        static void __execute_impl(__task* __p) noexcept {
          auto& __rcvr = ((__t*) __p)->__rcvr_;
          try {
            if (get_stop_token(get_env(__rcvr)).stop_requested()) {
              set_stopped((_Receiver&&) __rcvr);
            } else {
              set_value((_Receiver&&) __rcvr);
            }
          } catch (...) {
            set_error((_Receiver&&) __rcvr, std::current_exception());
          }
        }

        explicit __t(__task* __tail) noexcept
          : __task{.__tail_ = __tail} {
        }

        __t(__task* __next, run_loop* __loop, _Receiver __rcvr)
          : __task{{}, __next, {&__execute_impl}}
          , __loop_{__loop}
          , __rcvr_{(_Receiver&&) __rcvr} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__start_();
        }

        void __start_() noexcept;
      };
    };

    class run_loop {
      template <class... Ts>
      using __completion_signatures_ = completion_signatures<Ts...>;

      template <class>
      friend struct __operation;
     public:
      struct __scheduler {
        using __t = __scheduler;
        using __id = __scheduler;
        bool operator==(const __scheduler&) const noexcept = default;

       private:
        struct __schedule_task {
          using __t = __schedule_task;
          using __id = __schedule_task;
          using sender_concept = sender_t;
          using completion_signatures = //
            __completion_signatures_<
              set_value_t(),
              set_error_t(std::exception_ptr),
              set_stopped_t()>;

         private:
          friend __scheduler;

          template <class _Receiver>
          using __operation = stdexec::__t<__operation<stdexec::__id<_Receiver>>>;

          template <class _Receiver>
          friend __operation<_Receiver>
            tag_invoke(connect_t, const __schedule_task& __self, _Receiver __rcvr) {
            return __self.__connect_((_Receiver&&) __rcvr);
          }

          template <class _Receiver>
          __operation<_Receiver> __connect_(_Receiver&& __rcvr) const {
            return {&__loop_->__head_, __loop_, (_Receiver&&) __rcvr};
          }

          struct __env {
            run_loop* __loop_;

            template <class _CPO>
            friend __scheduler
              tag_invoke(get_completion_scheduler_t<_CPO>, const __env& __self) noexcept {
              return __self.__loop_->get_scheduler();
            }
          };

          friend __env tag_invoke(get_env_t, const __schedule_task& __self) noexcept {
            return __env{__self.__loop_};
          }

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop) {
          }

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept
          : __loop_(__loop) {
        }

        friend __schedule_task tag_invoke(schedule_t, const __scheduler& __self) noexcept {
          return __self.__schedule();
        }

        friend stdexec::forward_progress_guarantee
          tag_invoke(get_forward_progress_guarantee_t, const __scheduler&) noexcept {
          return stdexec::forward_progress_guarantee::parallel;
        }

        // BUGBUG NOT TO SPEC
        friend bool tag_invoke(execute_may_block_caller_t, const __scheduler&) noexcept {
          return false;
        }

        __schedule_task __schedule() const noexcept {
          return __schedule_task{__loop_};
        }

        run_loop* __loop_;
      };

      __scheduler get_scheduler() noexcept {
        return __scheduler{this};
      }

      void run();

      void finish();

     private:
      void __push_back_(__task* __task);
      __task* __pop_front_();

      std::mutex __mutex_;
      std::condition_variable __cv_;
      __task __head_{.__tail_ = &__head_};
      bool __stop_ = false;
    };

    template <class _ReceiverId>
    inline void __operation<_ReceiverId>::__t::__start_() noexcept {
      try {
        __loop_->__push_back_(this);
      } catch (...) {
        set_error((_Receiver&&) __rcvr_, std::current_exception());
      }
    }

    inline void run_loop::run() {
      for (__task* __task; (__task = __pop_front_()) != &__head_;) {
        __task->__execute();
      }
    }

    inline void run_loop::finish() {
      std::unique_lock __lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::__push_back_(__task* __task) {
      std::unique_lock __lock{__mutex_};
      __task->__next_ = &__head_;
      __head_.__tail_ = __head_.__tail_->__next_ = __task;
      __cv_.notify_one();
    }

    inline __task* run_loop::__pop_front_() {
      std::unique_lock __lock{__mutex_};
      __cv_.wait(__lock, [this] { return __head_.__next_ != &__head_ || __stop_; });
      if (__head_.__tail_ == __head_.__next_)
        __head_.__tail_ = &__head_;
      return std::exchange(__head_.__next_, __head_.__next_->__next_);
    }
  } // namespace __loop

  // NOT TO SPEC
  using run_loop = __loop::run_loop;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.schedule_from]
  namespace __schedule_from {
    // Compute a variant type that is capable of storing the results of the
    // input sender when it completes. The variant has type:
    //   variant<
    //     monostate,
    //     tuple<set_stopped_t>,
    //     tuple<set_value_t, __decay_t<_Values1>...>,
    //     tuple<set_value_t, __decay_t<_Values2>...>,
    //        ...
    //     tuple<set_error_t, __decay_t<_Error1>>,
    //     tuple<set_error_t, __decay_t<_Error2>>,
    //        ...
    //   >
    template <class _State, class... _Tuples>
    using __make_bind_ = __mbind_back<_State, _Tuples...>;

    template <class _State>
    using __make_bind = __mbind_front_q<__make_bind_, _State>;

    template <class _Tag>
    using __tuple_t = __mbind_front_q<__decayed_tuple, _Tag>;

    template <class _Sender, class _Env, class _State, class _Tag>
    using __bind_completions_t =
      __gather_completions_for<_Tag, _Sender, _Env, __tuple_t<_Tag>, __make_bind<_State>>;

    template <class _Sender, class _Env>
    using __variant_for_t = //
      __minvoke< __minvoke<
        __mfold_right< __nullable_variant_t, __mbind_front_q<__bind_completions_t, _Sender, _Env>>,
        set_value_t,
        set_error_t,
        set_stopped_t>>;

    template <class _SchedulerId, class _VariantId, class _ReceiverId>
    struct __operation1_base;

    // This receiver is to be completed on the execution context
    // associated with the scheduler. When the source sender
    // completes, the completion information is saved off in the
    // operation state so that when this receiver completes, it can
    // read the completion out of the operation state and forward it
    // to the output receiver after transitioning to the scheduler's
    // context.
    template <class _SchedulerId, class _VariantId, class _ReceiverId>
    struct __receiver2 {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using receiver_concept = receiver_t;
        using __id = __receiver2;
        __operation1_base<_SchedulerId, _VariantId, _ReceiverId>* __op_state_;

        // If the work is successfully scheduled on the new execution
        // context and is ready to run, forward the completion signal in
        // the operation state
        template <same_as<set_value_t> _Tag>
        friend void tag_invoke(_Tag, __t&& __self) noexcept {
          __self.__op_state_->__complete();
        }

        template <__one_of<set_error_t, set_stopped_t> _Tag, class... _As>
          requires __callable<_Tag, _Receiver, _As...>
        friend void tag_invoke(_Tag, __t&& __self, _As&&... __as) noexcept {
          _Tag{}((_Receiver&&) __self.__op_state_->__rcvr_, (_As&&) __as...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };
    };

    // This receiver is connected to the input sender. When that
    // sender completes (on whatever context it completes on), save
    // the completion information into the operation state. Then,
    // schedule a second operation to __complete on the execution
    // context of the scheduler. That second receiver will read the
    // completion information out of the operation state and propagate
    // it to the output receiver from within the desired context.
    template <class _SchedulerId, class _VariantId, class _ReceiverId>
    struct __receiver1 {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver2_t = stdexec::__t<__receiver2<_SchedulerId, _VariantId, _ReceiverId>>;

      struct __t {
        using __id = __receiver1;
        using receiver_concept = receiver_t;
        __operation1_base<_SchedulerId, _VariantId, _ReceiverId>* __op_state_;

        template <class... _Args>
        static constexpr bool __nothrow_complete_ = (__nothrow_decay_copyable<_Args> && ...);

        template <class _Tag, class... _Args>
        static void __complete_(_Tag, __t&& __self, _Args&&... __args) //
          noexcept(__nothrow_complete_<_Args...>) {
          // Write the tag and the args into the operation state so that
          // we can forward the completion from within the scheduler's
          // execution context.
          __self.__op_state_->__data_.template emplace<__decayed_tuple<_Tag, _Args...>>(
            _Tag{}, (_Args&&) __args...);
          // Enqueue the schedule operation so the completion happens
          // on the scheduler's execution context.
          start(__self.__op_state_->__state2_);
        }

        template <__completion_tag _Tag, class... _Args>
          requires __callable<_Tag, _Receiver, __decay_t<_Args>...>
        friend void tag_invoke(_Tag __tag, __t&& __self, _Args&&... __args) noexcept {
          __try_call(
            (_Receiver&&) __self.__op_state_->__rcvr_,
            __function_constant_v<__complete_<_Tag, _Args...>>,
            (_Tag&&) __tag,
            (__t&&) __self,
            (_Args&&) __args...);
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<_Receiver> {
          return get_env(__self.__op_state_->__rcvr_);
        }
      };
    };

    template <class _SchedulerId, class _VariantId, class _ReceiverId>
    struct __operation1_base : __immovable {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Variant = stdexec::__t<_VariantId>;
      using __receiver2_t = stdexec::__t<__receiver2<_SchedulerId, _VariantId, _ReceiverId>>;

      _Scheduler __sched_;
      _Receiver __rcvr_;
      _Variant __data_;
      connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t> __state2_;

      __operation1_base(_Scheduler __sched, _Receiver&& __rcvr)
        : __sched_((_Scheduler&&) __sched)
        , __rcvr_((_Receiver&&) __rcvr)
        , __state2_(connect(schedule(__sched_), __receiver2_t{this})) {
      }

      void __complete() noexcept {
        STDEXEC_ASSERT(!__data_.valueless_by_exception());
        std::visit(
          [this]<class _Tup>(_Tup& __tupl) -> void {
            if constexpr (same_as<_Tup, std::monostate>) {
              std::terminate(); // reaching this indicates a bug in schedule_from
            } else {
              __apply(
                [&]<class... _Args>(auto __tag, _Args&... __args) -> void {
                  __tag((_Receiver&&) __rcvr_, (_Args&&) __args...);
                },
                __tupl);
            }
          },
          __data_);
      }
    };

    template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
    struct __operation1 {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _CvrefSender = stdexec::__cvref_t<_CvrefSenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __variant_t = __variant_for_t<_CvrefSender, env_of_t<_Receiver>>;
      using __receiver1_t =
        stdexec::__t<__receiver1<_SchedulerId, stdexec::__id<__variant_t>, _ReceiverId>>;
      using __base_t = __operation1_base<_SchedulerId, stdexec::__id<__variant_t>, _ReceiverId>;

      struct __t : __base_t {
        using __id = __operation1;
        connect_result_t<_CvrefSender, __receiver1_t> __state1_;

        __t(_Scheduler __sched, _CvrefSender&& __sndr, _Receiver&& __rcvr)
          : __base_t{(_Scheduler&&) __sched, (_Receiver&&) __rcvr}
          , __state1_(connect((_CvrefSender&&) __sndr, __receiver1_t{this})) {
        }

        friend void tag_invoke(start_t, __t& __op_state) noexcept {
          start(__op_state.__state1_);
        }
      };
    };

    template <class _Tp>
    using __decay_rvalue_ref = __decay_t<_Tp>&&;

    template <class _Tag>
    using __decay_signature =
      __transform<__q<__decay_rvalue_ref>, __mcompose<__q<completion_signatures>, __qf<_Tag>>>;

    template <class _SchedulerId>
    struct __environ {
      struct __t
        : __env::__prop<stdexec::__t<_SchedulerId>(
            get_completion_scheduler_t<set_value_t>,
            get_completion_scheduler_t<set_stopped_t>)> {
        using __id = __environ;

        template <same_as<get_domain_t> _Key>
        friend auto tag_invoke(_Key, const __t& __self) noexcept {
          return query_or(get_domain, __self.__value_, default_domain());
        }
      };
    };

    template <class... _Ts>
    using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Ts> && ...)>;

    template <class _CvrefSender, class _Env>
    using __all_values_and_errors_nothrow_decay_copyable = //
      __mand<
        __try_error_types_of_t<_CvrefSender, _Env, __q<__all_nothrow_decay_copyable>>,
        __try_value_types_of_t<_CvrefSender, _Env, __q<__all_nothrow_decay_copyable>, __q<__mand>>>;

    template <class _CvrefSender, class _Env>
    using __with_error_t = //
      __if<
        __all_values_and_errors_nothrow_decay_copyable<_CvrefSender, _Env>,
        completion_signatures<>,
        __with_exception_ptr>;

    template <class _Scheduler, class _CvrefSender, class _Env>
    using __completions_t = //
      __try_make_completion_signatures<
        _CvrefSender,
        _Env,
        __try_make_completion_signatures<
          schedule_result_t<_Scheduler>,
          _Env,
          __with_error_t<_CvrefSender, _Env>,
          __mconst<completion_signatures<>>>,
        __decay_signature<set_value_t>,
        __decay_signature<set_error_t>>;

    struct schedule_from_t {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const {
        using _Env = __t<__environ<__id<__decay_t<_Scheduler>>>>;
        auto __env = _Env{{(_Scheduler&&) __sched}};
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain, __make_sexpr<schedule_from_t>(std::move(__env), (_Sender&&) __sndr));
      }

      using _Sender = __1;
      using _Env = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(schedule_from_t, get_completion_scheduler_t<set_value_t>(_Env&), _Sender)>;
    };

    struct __schedule_from_impl : __sexpr_defaults {
      template <class, class _Data, class _Child>
      using __env_ = __env::__env_join_t<_Data, env_of_t<_Child>>;

      template <class _Sender>
      using __env_t = __mapply<__q<__env_>, _Sender>;

      template <class _Sender>
      using __scheduler_t =
        __decay_t<__call_result_t<get_completion_scheduler_t<set_value_t>, __env_t<_Sender>>>;

      template <class _Sender, class _Receiver>
      using __receiver_t = //
        stdexec::__t< __receiver1<
          stdexec::__id<__scheduler_t<_Sender>>,
          stdexec::__id<__variant_for_t<__child_of<_Sender>, env_of_t<_Receiver>>>,
          stdexec::__id<_Receiver>>>;

      template <class _Sender, class _Receiver>
      using __operation_t =         //
        stdexec::__t< __operation1< //
          stdexec::__id<__scheduler_t<_Sender>>,
          stdexec::__cvref_id<__child_of<_Sender>>,
          stdexec::__id<_Receiver>>>;

      static constexpr auto get_attrs = //
        []<class _Data, class _Child>(const _Data& __data, const _Child& __child) noexcept {
        return __join_env(__data, stdexec::get_env(__child));
      };

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&&, const _Env&) noexcept
        -> __completions_t<__scheduler_t<_Sender>, __child_of<_Sender>, _Env> {
        static_assert(sender_expr_for<_Sender, schedule_from_t>);
        return {};
      };

      static constexpr auto connect =                                              //
        []<class _Sender, receiver _Receiver>(_Sender && __sndr, _Receiver __rcvr) //
        -> __operation_t<_Sender, _Receiver>                                       //
        requires sender_to<__child_of<_Sender>, __receiver_t<_Sender, _Receiver>>
      {
        static_assert(sender_expr_for<_Sender, schedule_from_t>);
        return __sexpr_apply(
          (_Sender&&) __sndr,
          [&]<class _Data, class _Child>(
            __ignore, _Data&& __data, _Child&& __child) -> __operation_t<_Sender, _Receiver> {
            auto __sched = get_completion_scheduler<set_value_t>(__data);
            return {__sched, (_Child&&) __child, (_Receiver&&) __rcvr};
          });
      };
    };
  } // namespace __schedule_from

  using __schedule_from::schedule_from_t;
  inline constexpr schedule_from_t schedule_from{};

  template <>
  struct __sexpr_impl<schedule_from_t> : __schedule_from::__schedule_from_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.transfer]
  namespace __transfer {
    using __schedule_from::__environ;

    template <class _Env>
    using __scheduler_t = __result_of<get_completion_scheduler<set_value_t>, _Env>;

    template <class _Sender>
    using __lowered_t = //
      __result_of<schedule_from, __scheduler_t<__data_of<_Sender>>, __child_of<_Sender>>;

    struct transfer_t {
      template <sender _Sender, scheduler _Scheduler>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const {
        auto __domain = __get_early_domain(__sndr);
        using _Env = __t<__environ<__id<__decay_t<_Scheduler>>>>;
        return stdexec::transform_sender(
          __domain, __make_sexpr<transfer_t>(_Env{{(_Scheduler&&) __sched}}, (_Sender&&) __sndr));
      }

      template <scheduler _Scheduler>
      STDEXEC_ATTRIBUTE((always_inline)) //
      __binder_back<transfer_t, __decay_t<_Scheduler>> operator()(_Scheduler&& __sched) const {
        return {{}, {}, {(_Scheduler&&) __sched}};
      }

      //////////////////////////////////////////////////////////////////////////////////////////////
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<
          tag_invoke_t(
            transfer_t,
            get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
            _Sender,
            get_completion_scheduler_t<set_value_t>(_Env)),
          tag_invoke_t(transfer_t, _Sender, get_completion_scheduler_t<set_value_t>(_Env))>;

      template <class _Env>
      static auto __transform_sender_fn(const _Env&) {
        return [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
          auto __sched = get_completion_scheduler<set_value_t>(__data);
          return schedule_from(std::move(__sched), (_Child&&) __child);
        };
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply((_Sender&&) __sndr, __transform_sender_fn(__env));
      }
    };

    struct __transfer_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data, class _Child>(const _Data& __data, const _Child& __child) noexcept -> decltype(auto) {
        return __join_env(__data, stdexec::get_env(__child));
      };
    };
  } // namespace __transfer

  using __transfer::transfer_t;
  inline constexpr transfer_t transfer{};

  template <>
  struct __sexpr_impl<transfer_t> : __transfer::__transfer_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  namespace __transfer_just {
    // This is a helper for finding legacy cusutomizations of transfer_just.
    inline auto __transfer_just_tag_invoke() {
      return []<class... _Ts>(_Ts&&... __ts) -> tag_invoke_result_t<transfer_just_t, _Ts...> {
        return tag_invoke(transfer_just, (_Ts&&) __ts...);
      };
    }

    template <class _Env>
    auto __make_transform_fn(const _Env& __env) {
      return [&]<class _Scheduler, class... _Values>(_Scheduler&& __sched, _Values&&... __vals) {
        return transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched);
      };
    }

    template <class _Env>
    auto __transform_sender_fn(const _Env& __env) {
      return [&]<class _Data>(__ignore, _Data&& __data) {
        return __apply(__make_transform_fn(__env), (_Data&&) __data);
      };
    }

    struct transfer_just_t {
      using _Data = __0;
      using __legacy_customizations_t = //
        __types<__apply_t(decltype(__transfer_just_tag_invoke()), _Data)>;

      template <scheduler _Scheduler, __movable_value... _Values>
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_just_t>(std::tuple{(_Scheduler&&) __sched, (_Values&&) __vals...}));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply((_Sender&&) __sndr, __transform_sender_fn(__env));
      }
    };

    inline auto __make_env_fn() noexcept {
      return []<class _Scheduler>(const _Scheduler& __sched, const auto&...) noexcept {
        using _Env = __t<__schedule_from::__environ<__id<_Scheduler>>>;
        return _Env{{__sched}};
      };
    }

    struct __transfer_just_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data) noexcept {
          return __apply(__make_env_fn(), __data);
        };
    };
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  inline constexpr transfer_just_t transfer_just{};

  template <>
  struct __sexpr_impl<transfer_just_t> : __transfer_just::__transfer_just_impl { };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // __write adaptor
  namespace __write_ {
    template <class _ReceiverId, class _Env>
    struct __operation_base {
      using _Receiver = __t<_ReceiverId>;
      _Receiver __rcvr_;
      const _Env __env_;
    };

    inline constexpr auto __get_env = [](auto* __op) noexcept -> decltype(auto) {
      return (__op->__env_);
    };

    template <class _ReceiverId, class _Env>
    using __receiver_t = //
      __t<__detail::__receiver_with< &__operation_base<_ReceiverId, _Env>::__rcvr_, __get_env>>;

    template <class _CvrefSenderId, class _ReceiverId, class _Env>
    struct __operation : __operation_base<_ReceiverId, _Env> {
      using _CvrefSender = __cvref_t<_CvrefSenderId>;
      using _Receiver = __t<_ReceiverId>;
      using __base_t = __operation_base<_ReceiverId, _Env>;
      using __receiver_t = __write_::__receiver_t<_ReceiverId, _Env>;
      connect_result_t<_CvrefSender, __receiver_t> __state_;

      __operation(_CvrefSender&& __sndr, _Receiver&& __rcvr, auto&& __env)
        : __base_t{(_Receiver&&) __rcvr, (decltype(__env)) __env}
        , __state_{stdexec::connect((_CvrefSender&&) __sndr, __receiver_t{{}, this})} {
      }

      friend void tag_invoke(start_t, __operation& __self) noexcept {
        start(__self.__state_);
      }
    };

    struct __write_t {
      template <sender _Sender, class... _Envs>
      auto operator()(_Sender&& __sndr, _Envs... __envs) const {
        return __make_sexpr<__write_t>(__join_env(std::move(__envs)...), (_Sender&&) __sndr);
      }

      template <class... _Envs>
      STDEXEC_ATTRIBUTE((always_inline)) //
      auto operator()(_Envs... __envs) const -> __binder_back<__write_t, _Envs...> {
        return {{}, {}, {std::move(__envs)...}};
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, const auto& __data, __ignore) noexcept {
          return __join_env(__data, (_Env&&) __env);
        };
      }

      template <sender_expr_for<__write_t> _Self, class _Env>
      static auto transform_env(const _Self& __self, _Env&& __env) noexcept {
        return __sexpr_apply(__self, __transform_env_fn((_Env&&) __env));
      }
    };

    struct __write_impl : __sexpr_defaults {
      template <class _Self, class _Receiver>
      using __receiver_t = __write_::__receiver_t<__id<_Receiver>, __decay_t<__data_of<_Self>>>;

      template <class _Self, class _Receiver>
      using __operation_t =
        __operation<__cvref_id<__child_of<_Self>>, __id<_Receiver>, __decay_t<__data_of<_Self>>>;

      static constexpr auto connect =                                          //
        []<class _Self, receiver _Receiver>(_Self && __self, _Receiver __rcvr) //
        -> __operation_t<_Self, _Receiver>                                     //
        requires sender_to<__child_of<_Self>, __receiver_t<_Self, _Receiver>>
      {
        static_assert(sender_expr_for<_Self, __write_t>);
        return __sexpr_apply(
          (_Self&&) __self,
          [&]<class _Env, class _Child>(__write_t, _Env&& __env, _Child&& __child) //
          -> __operation_t<_Self, _Receiver> {
            return {(_Child&&) __child, (_Receiver&&) __rcvr, (_Env&&) __env};
          });
      };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept
        -> stdexec::__completion_signatures_of_t<
          __child_of<_Self>,
          __env::__env_join_t<const __decay_t<__data_of<_Self>>&, _Env>> {
        static_assert(sender_expr_for<_Self, __write_t>);
        return {};
      };
    };
  } // namespace __write_

  using __write_::__write_t;
  inline constexpr __write_t __write{};

  template <>
  struct __sexpr_impl<__write_t> : __write_::__write_impl { };

  namespace __detail {
    template <class _Scheduler>
    STDEXEC_ATTRIBUTE((always_inline))
    auto __mkenv_sched(_Scheduler __sched) {
      auto __env = __join_env(__mkprop(__sched, get_scheduler), __mkprop(get_domain));

      struct __env_t : decltype(__env) { };

      return __env_t{__env};
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on {
    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __operation;

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __operation_base : __immovable {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      _Scheduler __scheduler_;
      _Sender __sndr_;
      _Receiver __rcvr_;
    };

    inline constexpr auto __sched_prop = [](auto* __op) noexcept {
      return __mkprop(__op->__scheduler_, get_scheduler);
    };
    inline constexpr auto __domain_prop = [](auto*) noexcept {
      return __mkprop(get_domain);
    };

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    using __receiver_ref_t = __t<__detail::__receiver_with<
      &__operation_base<_SchedulerId, _SenderId, _ReceiverId>::__rcvr_,
      __sched_prop,
      __domain_prop>>;

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __receiver {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : receiver_adaptor<__t> {
        using __id = __receiver;
        using __receiver_ref_t = __on::__receiver_ref_t<_SchedulerId, _SenderId, _ReceiverId>;
        stdexec::__t<__operation<_SchedulerId, _SenderId, _ReceiverId>>* __op_state_;

        _Receiver&& base() && noexcept {
          return (_Receiver&&) __op_state_->__rcvr_;
        }

        const _Receiver& base() const & noexcept {
          return __op_state_->__rcvr_;
        }

        void set_value() && noexcept {
          // cache this locally since *this is going bye-bye.
          auto* __op_state = __op_state_;
          try {
            // This line will invalidate *this:
            start(__op_state->__data_.template emplace<1>(__conv{[__op_state] {
              return connect((_Sender&&) __op_state->__sndr_, __receiver_ref_t{{}, __op_state});
            }}));
          } catch (...) {
            set_error((_Receiver&&) __op_state->__rcvr_, std::current_exception());
          }
        }
      };
    };

    template <class _SchedulerId, class _SenderId, class _ReceiverId>
    struct __operation {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_SchedulerId, _SenderId, _ReceiverId> {
        using _Base = __operation_base<_SchedulerId, _SenderId, _ReceiverId>;
        using __id = __operation;
        using __receiver_t = stdexec::__t<__receiver<_SchedulerId, _SenderId, _ReceiverId>>;
        using __receiver_ref_t = __on::__receiver_ref_t<_SchedulerId, _SenderId, _ReceiverId>;
        using __schedule_sender_t = __result_of<schedule, _Scheduler>;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(std::get<0>(__self.__data_));
        }

        template <class _Sender2, class _Receiver2>
        __t(_Scheduler __sched, _Sender2&& __sndr, _Receiver2&& __rcvr)
          : _Base{{}, (_Scheduler&&) __sched, (_Sender2&&) __sndr, (_Receiver2&&) __rcvr}
          , __data_{std::in_place_index<0>, __conv{[this] {
                      return connect(schedule(this->__scheduler_), __receiver_t{{}, this});
                    }}} {
        }

        std::variant<
          connect_result_t<__schedule_sender_t, __receiver_t>,
          connect_result_t<_Sender, __receiver_ref_t>>
          __data_;
      };
    };

    template <class _SchedulerId, class _SenderId>
    struct __sender;

    struct on_t;

    template <class _SchedulerId, class _SenderId>
    struct __sender {
      using _Scheduler = stdexec::__t<_SchedulerId>;
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        using __id = __sender;
        using sender_concept = sender_t;

        using __schedule_sender_t = __result_of<schedule, _Scheduler>;
        template <class _ReceiverId>
        using __receiver_ref_t = __on::__receiver_ref_t<_SchedulerId, _SenderId, _ReceiverId>;
        template <class _ReceiverId>
        using __receiver_t = stdexec::__t<__receiver<_SchedulerId, _SenderId, _ReceiverId>>;
        template <class _ReceiverId>
        using __operation_t = stdexec::__t<__operation<_SchedulerId, _SenderId, _ReceiverId>>;

        _Scheduler __scheduler_;
        _Sender __sndr_;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires constructible_from<_Sender, __copy_cvref_t<_Self, _Sender>>
                && sender_to<__schedule_sender_t, __receiver_t<stdexec::__id<_Receiver>>>
                && sender_to<_Sender, __receiver_ref_t<stdexec::__id<_Receiver>>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<stdexec::__id<_Receiver>> {
          return {
            ((_Self&&) __self).__scheduler_, ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
        }

        friend auto tag_invoke(get_env_t, const __t& __self) noexcept -> env_of_t<const _Sender&> {
          return get_env(__self.__sndr_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __try_make_completion_signatures<
            __schedule_sender_t,
            _Env,
            __try_make_completion_signatures<
              __copy_cvref_t<_Self, _Sender>,
              __make_env_t<_Env, __with<get_scheduler_t, _Scheduler>>,
              completion_signatures<set_error_t(std::exception_ptr)>>,
            __mconst<completion_signatures<>>> {
          return {};
        }

        // BUGBUG better would be to port the `on` algorithm to __sexpr
        template <class _Self, class _Fun, class _OnTag = on_t>
        static auto apply(_Self&& __self, _Fun __fun) -> __call_result_t<
          _Fun,
          _OnTag,
          __copy_cvref_t<_Self, _Scheduler>,
          __copy_cvref_t<_Self, _Sender>> {
          return ((_Fun&&) __fun)(
            _OnTag(), ((_Self&&) __self).__scheduler_, ((_Self&&) __self).__sndr_);
        }
      };
    };

    struct on_t {
      using _Sender = __1;
      using _Scheduler = __0;
      using __legacy_customizations_t = __types< tag_invoke_t(on_t, _Scheduler, _Sender)>;

      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const {
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain, __make_sexpr<on_t>((_Scheduler&&) __sched, (_Sender&&) __sndr));
      }

      template <class _Scheduler, class _Sender>
      using __sender_t =
        __t<__sender<stdexec::__id<__decay_t<_Scheduler>>, stdexec::__id<__decay_t<_Sender>>>>;

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, auto __sched, __ignore) noexcept {
          return __join_env(__detail::__mkenv_sched(__sched), (_Env&&) __env);
        };
      }

      template <class _Sender, class _Env>
      static auto transform_env(const _Sender& __sndr, _Env&& __env) noexcept {
        return __sexpr_apply(__sndr, __transform_env_fn((_Env&&) __env));
      }

      template <class _Sender, class _Env>
        requires __is_not_instance_of<__id<__decay_t<_Sender>>, __sender>
      static auto transform_sender(_Sender&& __sndr, const _Env&) {
        return __sexpr_apply(
          (_Sender&&) __sndr,
          []<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
            return __sender_t<_Data, _Child>{(_Data&&) __data, (_Child&&) __child};
          });
      }
    };
  } // namespace __on

  using __on::on_t;
  inline constexpr on_t on{};

  // BUGBUG this will also be unnecessary when `on` returns a __sexpr
  namespace __detail {
    template <class _SchedulerId, class _SenderId>
    extern __mconst<__on::__sender<__t<_SchedulerId>, __name_of<__t<_SenderId>>>>
      __name_of_v<__on::__sender<_SchedulerId, _SenderId>>;
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.into_variant]
  namespace __into_variant {
    template <class _Sender, class _Env>
      requires sender_in<_Sender, _Env>
    using __into_variant_result_t = value_types_of_t<_Sender, _Env>;

    template <class _Sender, class _Env>
    using __variant_t = __try_value_types_of_t<_Sender, _Env>;

    template <class _Variant>
    using __variant_completions =
      completion_signatures< set_value_t(_Variant), set_error_t(std::exception_ptr)>;

    template <class _Sender, class _Env>
    using __compl_sigs = //
      __try_make_completion_signatures<
        _Sender,
        _Env,
        __meval<__variant_completions, __variant_t<_Sender, _Env>>,
        __mconst<completion_signatures<>>>;

    struct into_variant_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<into_variant_t>(__(), std::move(__sndr)));
      }

      STDEXEC_ATTRIBUTE((always_inline)) //
      auto operator()() const noexcept {
        return __binder_back<into_variant_t>{};
      }
    };

    struct __into_variant_impl : __sexpr_defaults {
      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(_Self&&, _Receiver&) noexcept {
          using __variant_t = value_types_of_t<__child_of<_Self>, env_of_t<_Receiver>>;
          return __mtype<__variant_t>();
        };

      static constexpr auto complete = //
        []<class _State, class _Receiver, class _Tag, class... _Args>(__ignore, _State, _Receiver& __rcvr, _Tag, _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_Tag, set_value_t>) {
          using __variant_t = __t<_State>;
          try {
            set_value((_Receiver&&) __rcvr, __variant_t{std::tuple<_Args&&...>{(_Args&&) __args...}});
          } catch (...) {
            set_error((_Receiver&&) __rcvr, std::current_exception());
          }
        } else {
          _Tag()((_Receiver&&) __rcvr, (_Args&&) __args...);
        }
      };

      static constexpr auto get_completion_signatures =       //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept //
        -> __compl_sigs<__child_of<_Self>, _Env> {
        static_assert(sender_expr_for<_Self, into_variant_t>);
        return {};
      };
    };
  } // namespace __into_variant

  using __into_variant::into_variant_t;
  inline constexpr into_variant_t into_variant{};

  template <>
  struct __sexpr_impl<into_variant_t> : __into_variant::__into_variant_impl { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  namespace __when_all {
    enum __state_t {
      __started,
      __error,
      __stopped
    };

    struct __on_stop_request {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _Env>
    auto __mkenv(_Env&& __env, in_place_stop_source& __stop_source) noexcept {
      return __join_env(__mkprop(__stop_source.get_token(), get_stop_token), (_Env&&) __env);
    }

    template <class _Env>
    using __env_t = //
      decltype(__mkenv(__declval<_Env>(), __declval<in_place_stop_source&>()));

    template <class _Tp>
    using __decay_rvalue_ref = __decay_t<_Tp>&&;

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __mvalid<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    template <
      __mstring _Context = "In stdexec::when_all()..."__csz,
      __mstring _Diagnostic =
        "The given sender can complete successfully in more that one way. "
        "Use stdexec::when_all_with_variant() instead."__csz>
    struct _INVALID_WHEN_ALL_ARGUMENT_;

    template <class _Sender, class _Env>
    using __too_many_value_completions_error =
      __mexception<_INVALID_WHEN_ALL_ARGUMENT_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>>;

    template <class _Sender, class _Env, class _ValueTuple, class... _Rest>
    using __value_tuple_t = __minvoke<
      __if_c< (0 == sizeof...(_Rest)), __mconst<_ValueTuple>, __q<__too_many_value_completions_error>>,
      _Sender,
      _Env>;

    template <class _Env, class _Sender>
    using __single_values_of_t = //
      __try_value_types_of_t<
        _Sender,
        _Env,
        __transform<__q<__decay_rvalue_ref>, __q<__types>>,
        __mbind_front_q<__value_tuple_t, _Sender, _Env>>;

    template <class _Env, class... _Senders>
    using __set_values_sig_t = //
      __meval<
        completion_signatures,
        __minvoke< __mconcat<__qf<set_value_t>>, __single_values_of_t<_Env, _Senders>...>>;

    template <class... _Args>
    using __all_nothrow_decay_copyable = __mbool<(__nothrow_decay_copyable<_Args> && ...)>;

    template <class _Env, class... _Senders>
    using __all_value_and_error_args_nothrow_decay_copyable = //
      __mand< __compl_sigs::__maybe_for_all_sigs<
        __completion_signatures_of_t<_Senders, _Env>,
        __q<__all_nothrow_decay_copyable>,
        __q<__mand>>...>;

    template <class _Env, class... _Senders>
    using __completions_t = //
      __concat_completion_signatures_t<
        __if<
          __all_value_and_error_args_nothrow_decay_copyable<_Env, _Senders...>,
          completion_signatures<set_stopped_t()>,
          completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr&&)>>,
        __minvoke<
          __with_default< __mbind_front_q<__set_values_sig_t, _Env>, completion_signatures<>>,
          _Senders...>,
        __try_make_completion_signatures<
          _Senders,
          _Env,
          completion_signatures<>,
          __mconst<completion_signatures<>>,
          __mcompose<__q<completion_signatures>, __qf<set_error_t>, __q<__decay_rvalue_ref>>>...>;

    struct __not_an_error { };

    struct __tie_fn {
      template <class... _Ty>
      std::tuple<_Ty&...> operator()(_Ty&... __vals) noexcept {
        return std::tuple<_Ty&...>{__vals...};
      }
    };

    template <class _Tag, class _Receiver>
    auto __complete_fn(_Tag, _Receiver& __rcvr) noexcept {
      return [&]<class... _Ts>(_Ts&... __ts) noexcept {
        if constexpr (!same_as<__types<_Ts...>, __types<__not_an_error>>) {
          _Tag()((_Receiver&&) __rcvr, (_Ts&&) __ts...);
        }
      };
    }

    template <class _Receiver, class _ValuesTuple>
    void __set_values(_Receiver& __rcvr, _ValuesTuple& __values) noexcept {
      __apply(
        [&](auto&... __opt_vals) noexcept -> void {
          __apply(
            __complete_fn(set_value, __rcvr), //
            std::tuple_cat(__apply(__tie_fn{}, *__opt_vals)...));
        },
        __values);
    }

    template <class _Env, class _Sender>
    using __values_opt_tuple_t = //
      __value_types_of_t<
        _Sender,
        __env_t<_Env>,
        __mcompose<__q<std::optional>, __q<__decayed_tuple>>,
        __q<__msingle>>;

    template <class _Env, __max1_sender<__env_t<_Env>>... _Senders>
    struct __traits {
      // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
      using __values_tuple = //
        __minvoke<
          __with_default<
            __transform< __mbind_front_q<__values_opt_tuple_t, _Env>, __q<std::tuple>>,
            __ignore>,
          _Senders...>;

      using __nullable_variant_t_ = __munique<__mbind_front_q<std::variant, __not_an_error>>;

      using __error_types = //
        __minvoke<
          __mconcat<__transform<__q<__decay_t>, __nullable_variant_t_>>,
          error_types_of_t<_Senders, __env_t<_Env>, __types>... >;

      using __errors_variant = //
        __if<
          __all_value_and_error_args_nothrow_decay_copyable<_Env, _Senders...>,
          __error_types,
          __minvoke<__push_back_unique<__q<std::variant>>, __error_types, std::exception_ptr>>;
    };

    struct _INVALID_ARGUMENTS_TO_WHEN_ALL_ { };

    template <class _ErrorsVariant, class _ValuesTuple, class _StopToken>
    struct __when_all_state {
      using __stop_callback_t = typename _StopToken::template callback_type<__on_stop_request>;

      template <class _Receiver>
      void __arrive(_Receiver& __rcvr) noexcept {
        if (0 == --__count_) {
          __complete(__rcvr);
        }
      }

      template <class _Receiver>
      void __complete(_Receiver& __rcvr) noexcept {
        // Stop callback is no longer needed. Destroy it.
        __on_stop_.reset();
        // All child operations have completed and arrived at the barrier.
        switch (__state_.load(std::memory_order_relaxed)) {
        case __started:
          if constexpr (!same_as<_ValuesTuple, __ignore>) {
            // All child operations completed successfully:
            __when_all::__set_values(__rcvr, __values_);
          }
          break;
        case __error:
          if constexpr (!same_as<_ErrorsVariant, std::variant<std::monostate>>) {
            // One or more child operations completed with an error:
            std::visit(__complete_fn(set_error, __rcvr), __errors_);
          }
          break;
        case __stopped:
          stdexec::set_stopped((_Receiver&&) __rcvr);
          break;
        default:;
        }
      }

      std::atomic<std::size_t> __count_;
      in_place_stop_source __stop_source_{};
      // Could be non-atomic here and atomic_ref everywhere except __completion_fn
      std::atomic<__state_t> __state_{__started};
      _ErrorsVariant __errors_{};
      STDEXEC_ATTRIBUTE((no_unique_address)) _ValuesTuple __values_{};
      std::optional<__stop_callback_t> __on_stop_{};
    };

    template <class _Env>
    static auto __mk_state_fn(const _Env& __env) noexcept {
      return [&]<__max1_sender<__env_t<_Env>>... _Child>(__ignore, __ignore, _Child&&...) {
        using _Traits = __traits<_Env, _Child...>;
        using _ErrorsVariant = typename _Traits::__errors_variant;
        using _ValuesTuple = typename _Traits::__values_tuple;
        using _State = __when_all_state<_ErrorsVariant, _ValuesTuple, stop_token_of_t<_Env>>;
        return _State{
          sizeof...(_Child),
          in_place_stop_source{},
          __started,
          _ErrorsVariant{},
          _ValuesTuple{},
          std::nullopt};
      };
    }

    template <class _Env>
    using __mk_state_fn_t = decltype(__when_all::__mk_state_fn(__declval<_Env>()));

    struct when_all_t {
      // Used by the default_domain to find legacy customizations:
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(when_all_t, _Sender...)>;

      // TODO: improve diagnostic when senders have different domains
      template <sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto operator()(_Senders&&... __sndrs) const {
        auto __domain = __domain::__common_domain_t<_Senders...>();
        return stdexec::transform_sender(
          __domain, __make_sexpr<when_all_t>(__(), (_Senders&&) __sndrs...));
      }
    };

    struct __when_all_impl : __sexpr_defaults {
      template <class _Self, class _Env>
      using __error_t = __mexception<
        _INVALID_ARGUMENTS_TO_WHEN_ALL_,
        __children_of<_Self, __q<_WITH_SENDERS_>>,
        _WITH_ENVIRONMENT_<_Env>>;

      template <class _Self, class _Env>
      using __completions = //
        __children_of<_Self, __mbind_front_q<__completions_t, __env_t<_Env>>>;

      static constexpr auto get_attrs = //
        []<class... _Child>(__ignore, const _Child&...) noexcept {
        using _Domain = __domain::__common_domain_t<_Child...>;
        if constexpr (same_as<_Domain, default_domain>) {
          return empty_env();
        } else {
          return __mkprop(_Domain(), get_domain);
        }
        STDEXEC_UNREACHABLE();
      };

      static constexpr auto get_completion_signatures = //
        []<class _Self, class _Env>(_Self&&, _Env&&) noexcept {
          static_assert(sender_expr_for<_Self, when_all_t>);
          return __minvoke<__mtry_catch<__q<__completions>, __q<__error_t>>, _Self, _Env>();
        };

      static constexpr auto get_env = //
        []<class _State, class _Receiver>(
          __ignore,
          _State& __state,
          const _Receiver& __rcvr) noexcept //
        -> __env_t<env_of_t<const _Receiver&>> {
        return __mkenv(stdexec::get_env(__rcvr), __state.__stop_source_);
      };

      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(_Self&& __self, _Receiver& __rcvr)
        -> __sexpr_apply_result_t<_Self, __mk_state_fn_t<env_of_t<_Receiver>>> {
        return __sexpr_apply((_Self&&) __self, __when_all::__mk_state_fn(stdexec::get_env(__rcvr)));
      };

      static constexpr auto start = //
        []<class _State, class _Receiver, class... _Operations>(
          _State& __state,
          _Receiver& __rcvr,
          _Operations&... __child_ops) noexcept -> void {
        // register stop callback:
        __state.__on_stop_.emplace(
          get_stop_token(stdexec::get_env(__rcvr)), __on_stop_request{__state.__stop_source_});
        if (__state.__stop_source_.stop_requested()) {
          // Stop has already been requested. Don't bother starting
          // the child operations.
          stdexec::set_stopped(std::move(__rcvr));
        } else {
          (stdexec::start(__child_ops), ...);
          if constexpr (sizeof...(__child_ops) == 0) {
            __state.__complete(__rcvr);
          }
        }
      };

      template <class _State, class _Receiver, class _Error>
      static void __set_error(_State& __state, _Receiver& __rcvr, _Error&& __err) noexcept {
        // TODO: What memory orderings are actually needed here?
        if (__error != __state.__state_.exchange(__error)) {
          __state.__stop_source_.request_stop();
          // We won the race, free to write the error into the operation
          // state without worry.
          if constexpr (__nothrow_decay_copyable<_Error>) {
            __state.__errors_.template emplace<__decay_t<_Error>>((_Error&&) __err);
          } else {
            try {
              __state.__errors_.template emplace<__decay_t<_Error>>((_Error&&) __err);
            } catch (...) {
              __state.__errors_.template emplace<std::exception_ptr>(std::current_exception());
            }
          }
        }
      }

      static constexpr auto complete = //
        []<class _Index, class _State, class _Receiver, class _SetTag, class... _Args>(
          _Index,
          _State& __state,
          _Receiver& __rcvr,
          _SetTag,
          _Args&&... __args) noexcept -> void {
        if constexpr (same_as<_SetTag, set_error_t>) {
          __set_error(__state, __rcvr, (_Args&&) __args...);
        } else if constexpr (same_as<_SetTag, set_stopped_t>) {
          __state_t __expected = __started;
          // Transition to the "stopped" state if and only if we're in the
          // "started" state. (If this fails, it's because we're in an
          // error state, which trumps cancellation.)
          if (__state.__state_.compare_exchange_strong(__expected, __stopped)) {
            __state.__stop_source_.request_stop();
          }
        } else if constexpr (!same_as<decltype(_State::__values_), __ignore>) {
          // We only need to bother recording the completion values
          // if we're not already in the "error" or "stopped" state.
          if (__state.__state_ == __started) {
            auto& __opt_values = std::get<__v<_Index>>(__state.__values_);
            static_assert(
              same_as<decltype(*__opt_values), __decayed_tuple<_Args...>&>,
              "One of the senders in this when_all() is fibbing about what types it sends");
            if constexpr ((__nothrow_decay_copyable<_Args> && ...)) {
              __opt_values.emplace((_Args&&) __args...);
            } else {
              try {
                __opt_values.emplace((_Args&&) __args...);
              } catch (...) {
                __set_error(__state, __rcvr, std::current_exception());
              }
            }
          }
        }

        __state.__arrive(__rcvr);
      };
    };

    struct when_all_with_variant_t {
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(when_all_with_variant_t, _Sender...)>;

      template <sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto operator()(_Senders&&... __sndrs) const {
        auto __domain = __domain::__common_domain_t<_Senders...>();
        return stdexec::transform_sender(
          __domain, __make_sexpr<when_all_with_variant_t>(__(), (_Senders&&) __sndrs...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        // transform the when_all_with_variant into a regular when_all (looking for
        // early when_all customizations), then transform it again to look for
        // late customizations.
        return __sexpr_apply(
          (_Sender&&) __sndr, [&]<class... _Child>(__ignore, __ignore, _Child&&... __child) {
            return when_all_t()(into_variant((_Child&&) __child)...);
          });
      }
    };

    struct __when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class... _Child>(__ignore, const _Child&...) noexcept {
        using _Domain = __domain::__common_domain_t<_Child...>;
        if constexpr (same_as<_Domain, default_domain>) {
          return empty_env();
        } else {
          return __mkprop(_Domain(), get_domain);
        }
        STDEXEC_UNREACHABLE();
      };
    };

    struct transfer_when_all_t {
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(
          transfer_when_all_t,
          get_completion_scheduler_t<set_value_t>(const _Env&),
          _Sender...)>;

      template <scheduler _Scheduler, sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const {
        using _Env = __t<__schedule_from::__environ<__id<__decay_t<_Scheduler>>>>;
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_when_all_t>(
            _Env{{(_Scheduler&&) __sched}}, (_Senders&&) __sndrs...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        // transform the transfer_when_all into a regular transform | when_all
        // (looking for early customizations), then transform it again to look for
        // late customizations.
        return __sexpr_apply(
          (_Sender&&) __sndr,
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return transfer(
              when_all_t()((_Child&&) __child...), get_completion_scheduler<set_value_t>(__data));
          });
      }
    };

    struct __transfer_when_all_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data, const auto&...) noexcept -> const _Data& {
        return __data;
      };
    };

    struct transfer_when_all_with_variant_t {
      using _Env = __0;
      using _Sender = __1;
      using __legacy_customizations_t = //
        __types<tag_invoke_t(
          transfer_when_all_with_variant_t,
          get_completion_scheduler_t<set_value_t>(const _Env&),
          _Sender...)>;

      template <scheduler _Scheduler, sender... _Senders>
        requires __domain::__has_common_domain<_Senders...>
      auto operator()(_Scheduler&& __sched, _Senders&&... __sndrs) const {
        using _Env = __t<__schedule_from::__environ<__id<__decay_t<_Scheduler>>>>;
        auto __domain = query_or(get_domain, __sched, default_domain());
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<transfer_when_all_with_variant_t>(
            _Env{{(_Scheduler&&) __sched}}, (_Senders&&) __sndrs...));
      }

      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        // transform the transfer_when_all_with_variant into regular transform_when_all
        // and into_variant calls/ (looking for early customizations), then transform it
        // again to look for late customizations.
        return __sexpr_apply(
          (_Sender&&) __sndr,
          [&]<class _Data, class... _Child>(__ignore, _Data&& __data, _Child&&... __child) {
            return transfer_when_all_t()(
              get_completion_scheduler<set_value_t>((_Data&&) __data),
              into_variant((_Child&&) __child)...);
          });
      }
    };

    struct __transfer_when_all_with_variant_impl : __sexpr_defaults {
      static constexpr auto get_attrs = //
        []<class _Data>(const _Data& __data, const auto&...) noexcept -> const _Data& {
        return __data;
      };
    };
  } // namespace __when_all

  using __when_all::when_all_t;
  inline constexpr when_all_t when_all{};

  using __when_all::when_all_with_variant_t;
  inline constexpr when_all_with_variant_t when_all_with_variant{};

  using __when_all::transfer_when_all_t;
  inline constexpr transfer_when_all_t transfer_when_all{};

  using __when_all::transfer_when_all_with_variant_t;
  inline constexpr transfer_when_all_with_variant_t transfer_when_all_with_variant{};

  template <>
  struct __sexpr_impl<when_all_t> : __when_all::__when_all_impl { };

  template <>
  struct __sexpr_impl<when_all_with_variant_t> : __when_all::__when_all_with_variant_impl { };

  template <>
  struct __sexpr_impl<transfer_when_all_t> : __when_all::__transfer_when_all_impl { };

  template <>
  struct __sexpr_impl<transfer_when_all_with_variant_t>
    : __when_all::__transfer_when_all_with_variant_impl { };

  namespace __read {
    template <class _Tag, class _ReceiverId>
    using __result_t = __call_result_t<_Tag, env_of_t<stdexec::__t<_ReceiverId>>>;

    template <class _Tag, class _ReceiverId>
    concept __nothrow_t = __nothrow_callable<_Tag, env_of_t<stdexec::__t<_ReceiverId>>>;

    inline constexpr __mstring __query_failed_diag =
      "The current execution environment doesn't have a value for the given query."__csz;

    template <class _Tag>
    struct _WITH_QUERY_;

    template <class _Tag, class _Env>
    using __query_failed_error = //
      __mexception<              //
        _NOT_CALLABLE_<"In stdexec::read()..."__csz, __query_failed_diag>,
        _WITH_QUERY_<_Tag>,
        _WITH_ENVIRONMENT_<_Env>>;

    template <class _Tag, class _Env>
      requires __callable<_Tag, _Env>
    using __completions_t = //
      __if_c<
        __nothrow_callable<_Tag, _Env>,
        completion_signatures<set_value_t(__call_result_t<_Tag, _Env>)>,
        completion_signatures<
          set_value_t(__call_result_t<_Tag, _Env>),
          set_error_t(std::exception_ptr)>>;

    template <class _Tag, class _Ty>
    struct __state {
      using __query = _Tag;
      using __result = _Ty;
      std::optional<_Ty> __result_;
    };

    template <class _Tag, class _Ty>
      requires same_as<_Ty, _Ty&&>
    struct __state<_Tag, _Ty> {
      using __query = _Tag;
      using __result = _Ty;
    };

    struct __read_t {
      template <class _Tag>
      constexpr auto operator()(_Tag) const noexcept {
        return __make_sexpr<__read_t>(_Tag());
      }
    };

    struct __read_impl : __sexpr_defaults {
      template <class _Tag, class _Env>
      using __completions_t =
        __minvoke<__mtry_catch_q<__read::__completions_t, __q<__query_failed_error>>, _Tag, _Env>;

      static constexpr auto get_completion_signatures =            //
        []<class _Self, class _Env>(const _Self&, _Env&&) noexcept //
        -> __completions_t<__data_of<_Self>, _Env> {
        static_assert(sender_expr_for<_Self, __read_t>);
        return {};
      };

      static constexpr auto get_state = //
        []<class _Self, class _Receiver>(const _Self&, _Receiver& __rcvr) noexcept {
          using __query = __data_of<_Self>;
          using __result = __call_result_t<__query, env_of_t<_Receiver>>;
          return __state<__query, __result>();
        };

      static constexpr auto start = //
        []<class _State, class _Receiver>(_State& __state, _Receiver& __rcvr) noexcept -> void {
        using __query = typename _State::__query;
        using __result = typename _State::__result;
        if constexpr (same_as<__result, __result&&>) {
          // The query returns a reference type; pass it straight through to the receiver.
          stdexec::__set_value_invoke(std::move(__rcvr), __query(), stdexec::get_env(__rcvr));
        } else {
          constexpr bool _Nothrow = __nothrow_callable<__query, env_of_t<_Receiver>>;
          auto __query_fn = [&]() noexcept(_Nothrow) -> __result&& {
            __state.__result_.emplace(__conv{[&]() noexcept(_Nothrow) {
              return __query()(stdexec::get_env(__rcvr));
            }});
            return std::move(*__state.__result_);
          };
          stdexec::__set_value_invoke(std::move(__rcvr), __query_fn);
        }
      };
    };
  } // namespace __read

  inline constexpr __read::__read_t read{};

  template <>
  struct __sexpr_impl<__read::__read_t> : __read::__read_impl { };

  namespace __queries {
    inline auto get_scheduler_t::operator()() const noexcept {
      return read(get_scheduler);
    }

    template <class _Env>
      requires tag_invocable<get_scheduler_t, const _Env&>
    inline auto get_scheduler_t::operator()(const _Env& __env) const noexcept
      -> tag_invoke_result_t<get_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_scheduler_t, const _Env&>>);
      return tag_invoke(get_scheduler_t{}, __env);
    }

    inline auto get_delegatee_scheduler_t::operator()() const noexcept {
      return read(get_delegatee_scheduler);
    }

    template <class _Env>
      requires tag_invocable<get_delegatee_scheduler_t, const _Env&>
    inline auto get_delegatee_scheduler_t::operator()(const _Env& __t) const noexcept
      -> tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&> {
      static_assert(nothrow_tag_invocable<get_delegatee_scheduler_t, const _Env&>);
      static_assert(scheduler<tag_invoke_result_t<get_delegatee_scheduler_t, const _Env&>>);
      return tag_invoke(get_delegatee_scheduler_t{}, std::as_const(__t));
    }

    inline auto get_allocator_t::operator()() const noexcept {
      return read(get_allocator);
    }

    inline auto get_stop_token_t::operator()() const noexcept {
      return read(get_stop_token);
    }

    template <__completion_tag _CPO>
    template <__has_completion_scheduler_for<_CPO> _Queryable>
    auto get_completion_scheduler_t<_CPO>::operator()(const _Queryable& __queryable) const noexcept
      -> tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&> {
      static_assert(
        nothrow_tag_invocable<get_completion_scheduler_t<_CPO>, const _Queryable&>,
        "get_completion_scheduler<_CPO> should be noexcept");
      static_assert(
        scheduler<tag_invoke_result_t<get_completion_scheduler_t<_CPO>, const _Queryable&>>);
      return tag_invoke(*this, __queryable);
    }
  } // namespace __queries

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  namespace __on_v2 {
    inline constexpr __mstring __on_context = "In stdexec::on(Scheduler, Sender)..."__csz;
    inline constexpr __mstring __no_scheduler_diag =
      "stdexec::on() requires a scheduler to transition back to."__csz;
    inline constexpr __mstring __no_scheduler_details =
      "The provided environment lacks a value for the get_scheduler() query."__csz;

    template <
      __mstring _Context = __on_context,
      __mstring _Diagnostic = __no_scheduler_diag,
      __mstring _Details = __no_scheduler_details>
    struct _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_ { };

    struct on_t;

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-local-typedefs")

    struct __no_scheduler_in_environment {
      // Issue a custom diagnostic if the environment doesn't provide a scheduler.
      template <class _Sender, class _Env>
      static auto transform_sender(_Sender&&, const _Env&) {
        struct __no_scheduler_in_environment {
          using sender_concept = sender_t;
          using completion_signatures = //
            __mexception<
              _CANNOT_RESTORE_EXECUTION_CONTEXT_AFTER_ON_<>,
              _WITH_SENDER_<_Sender>,
              _WITH_ENVIRONMENT_<_Env>>;
        };

        return __no_scheduler_in_environment{};
      }
    };

    STDEXEC_PRAGMA_POP()

    template <class _Ty, class = __name_of<__decay_t<_Ty>>>
    struct __always {
      _Ty __val_;

      _Ty operator()() noexcept {
        return static_cast<_Ty&&>(__val_);
      }
    };

    template <class _Ty>
    __always(_Ty) -> __always<_Ty>;

    struct on_t : __no_scheduler_in_environment {
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const {
        // BUGBUG __get_early_domain, or get_domain(__sched), or ...?
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<on_t>((_Scheduler&&) __sched, (_Sender&&) __sndr));
      }

      template <class _Env>
      STDEXEC_ATTRIBUTE((always_inline))
      static auto __transform_env_fn(_Env&& __env) noexcept {
        return [&](__ignore, auto __sched, __ignore) noexcept {
          return __join_env(__detail::__mkenv_sched(__sched), (_Env&&) __env);
        };
      }

      template <class _Sender, class _Env>
      static auto transform_env(const _Sender& __sndr, _Env&& __env) noexcept {
        return __sexpr_apply(__sndr, __transform_env_fn((_Env&&) __env));
      }

      using __no_scheduler_in_environment::transform_sender;

      template <class _Sender, class _Env>
        requires __callable<get_scheduler_t, const _Env&>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        return __sexpr_apply(
          (_Sender&&) __sndr,
          [&]<class _Scheduler, class _Child>(__ignore, _Scheduler __sched, _Child&& __child) {
            auto __old = get_scheduler(__env);
            return transfer(
              let_value(transfer_just(std::move(__sched)), __always{(_Child&&) __child}),
              std::move(__old));
          });
      }
    };

    template <class _Scheduler, class _Closure>
    struct __continue_on_data {
      _Scheduler __sched_;
      _Closure __clsur_;
    };
    template <class _Scheduler, class _Closure>
    __continue_on_data(_Scheduler, _Closure) -> __continue_on_data<_Scheduler, _Closure>;

    struct continue_on_t : __no_scheduler_in_environment {
      template <sender _Sender, scheduler _Scheduler, __sender_adaptor_closure_for<_Sender> _Closure>
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched, _Closure&& __clsur) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain,
          __make_sexpr<continue_on_t>(
            __continue_on_data{(_Scheduler&&) __sched, (_Closure&&) __clsur}, (_Sender&&) __sndr));
      }

      template <scheduler _Scheduler, __sender_adaptor_closure _Closure>
      STDEXEC_ATTRIBUTE((always_inline)) //
      auto operator()(_Scheduler&& __sched, _Closure&& __clsur) const
        -> __binder_back<continue_on_t, __decay_t<_Scheduler>, __decay_t<_Closure>> {
        return {
          {},
          {},
          {(_Scheduler&&) __sched, (_Closure&&) __clsur}
        };
      }

      using __no_scheduler_in_environment::transform_sender;

      template <class _Sender, class _Env>
        requires __callable<get_scheduler_t, const _Env&>
      static auto transform_sender(_Sender&& __sndr, const _Env& __env) {
        auto __old = get_scheduler(__env);
        return __sexpr_apply(
          (_Sender&&) __sndr,
          [&]<class _Data, class _Child>(__ignore, _Data&& __data, _Child&& __child) {
            auto&& [__sched, __clsur] = (_Data&&) __data;
            using _Closure = decltype(__clsur);
            return __write(
              transfer(
                ((_Closure&&) __clsur)(
                  transfer(__write((_Child&&) __child, __detail::__mkenv_sched(__old)), __sched)),
                __old),
              __detail::__mkenv_sched(__sched));
          });
      }
    };
  } // __on_v2

  namespace v2 {
    using __on_v2::on_t;
    inline constexpr on_t on{};

    using __on_v2::continue_on_t;
    inline constexpr continue_on_t continue_on{};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  namespace __sync_wait {
    inline auto __make_env(run_loop& __loop) noexcept {
      return __env::__env_fn{
        [&](__one_of<get_scheduler_t, get_delegatee_scheduler_t> auto) noexcept {
          return __loop.get_scheduler();
        }};
    }

    struct __env : __result_of<__make_env, run_loop&> {
      __env();

      __env(run_loop& __loop) noexcept
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
            __state_->__data_.template emplace<2>((_Error&&) __err);
          else if constexpr (__decays_to<_Error, std::error_code>)
            __state_->__data_.template emplace<2>(
              std::make_exception_ptr(std::system_error(__err)));
          else
            __state_->__data_.template emplace<2>(std::make_exception_ptr((_Error&&) __err));
          __loop_->finish();
        }

        template <same_as<set_value_t> _Tag, class... _As>
          requires constructible_from<std::tuple<_Values...>, _As...>
        friend void tag_invoke(_Tag, __t&& __rcvr, _As&&... __as) noexcept {
          try {
            __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
            __rcvr.__loop_->finish();
          } catch (...) {
            __rcvr.__set_error(std::current_exception());
          }
        }

        template <same_as<set_error_t> _Tag, class _Error>
        friend void tag_invoke(_Tag, __t&& __rcvr, _Error __err) noexcept {
          __rcvr.__set_error((_Error&&) __err);
        }

        friend void tag_invoke(set_stopped_t __d, __t&& __rcvr) noexcept {
          __rcvr.__state_->__data_.template emplace<3>(__d);
          __rcvr.__loop_->finish();
        }

        friend __env tag_invoke(get_env_t, const __t& __rcvr) noexcept {
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
      using __t = __sync_wait_result_t<__result_of<into_variant, _Sender>>;
    };
    template <class _Sender>
    using __variant_for_t = __t<__variant_for<_Sender>>;

    inline constexpr __mstring __sync_wait_context_diag = //
      "In stdexec::sync_wait()..."__csz;
    inline constexpr __mstring __too_many_successful_completions_diag =
      "The argument to stdexec::sync_wait() is a sender that can complete successfully in more "
      "than one way. Use stdexec::sync_wait_with_variant() instead."__csz;

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
    concept __valid_sync_wait_argument = __ok< __minvoke<
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
        if constexpr (!__ok<_Completions>) {
          return _Completions();
        } else {
          constexpr __mstring __diag =
            "The stdexec::sender_in<Sender, Environment> concept check has failed."__csz;
          return __sync_wait_error<__diag, _Sender>();
        }
      } else if constexpr (!__valid_sync_wait_argument<_Sender>) {
        return __sync_wait_error<__too_many_successful_completions_diag, _Sender>();
      } else if constexpr (!sender_to<_Sender, __sync_receiver_for_t<_Sender>>) {
        constexpr __mstring __diag =
          "Failed to connect the given sender to sync_wait's internal receiver. "
          "The stdexec::connect(Sender, Receiver) expression is ill-formed."__csz;
        return __sync_wait_error<__diag, _Sender>();
      } else {
        constexpr __mstring __diag = "Unknown concept check failure."__csz;
        return __sync_wait_error<__diag, _Sender>();
      }
      STDEXEC_UNREACHABLE();
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
        using _SD = __early_domain_of_t<_Sender>;
        constexpr bool __has_custom_impl = __callable<apply_sender_t, _SD, sync_wait_t, _Sender>;
        using _Domain = __if_c<__has_custom_impl, _SD, default_domain>;
        return stdexec::apply_sender(_Domain(), *this, (_Sender&&) __sndr);
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

      // The default implementation goes here:
      template <class _Sender>
        requires sender_to<_Sender, __sync_receiver_for_t<_Sender>>
      auto apply_sender(_Sender&& __sndr) const -> std::optional<__sync_wait_result_t<_Sender>> {
        using state_t = __sync_wait_result_impl<_Sender, __q<__state>>;
        state_t __state{};
        run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state = connect((_Sender&&) __sndr, __receiver_t<_Sender>{&__state, &__loop});
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
        return stdexec::apply_sender(__domain, *this, (_Sender&&) __sndr);
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
        return sync_wait_t()(into_variant((_Sender&&) __sndr));
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

  template <auto _Reason = "You cannot pipe one sender into another."__csz>
  struct _CANNOT_PIPE_INTO_A_SENDER_ { };

  template <class _Sender>
  using __bad_pipe_sink_t = __mexception<_CANNOT_PIPE_INTO_A_SENDER_<>, _WITH_SENDER_<_Sender>>;
} // namespace stdexec

#if STDEXEC_MSVC()
namespace stdexec {
  // MSVCBUG https://developercommunity.visualstudio.com/t/Incorrect-codegen-in-await_suspend-aroun/10454102

  // MSVC incorrectly allocates the return buffer for await_suspend calls within the suspended coroutine
  // frame. When the suspended coroutine is destroyed within await_suspend, the continuation coroutine handle
  // is not only used after free, but also overwritten by the debug malloc implementation when NRVO is in play.

  // This workaround delays the destruction of the suspended coroutine by wrapping the continuation in another
  // coroutine which destroys the former and transfers execution to the original continuation.

  // The wrapping coroutine is thread-local and is reused within the thread for each destroy-and-continue sequence.
  // The wrapping coroutine itself is destroyed at thread exit.

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

#define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue) \
  (::stdexec::__destroy_and_continue_msvc::__impl(__destroy, __continue))
#else
#define STDEXEC_DESTROY_AND_CONTINUE(__destroy, __continue) (__destroy.destroy(), __continue)
#endif

// For issuing a meaningful diagnostic for the erroneous `snd1 | snd2`.
template <stdexec::sender _Sender>
  requires stdexec::__ok<stdexec::__bad_pipe_sink_t<_Sender>>
auto operator|(stdexec::__ignore_sender, _Sender&&) noexcept -> stdexec::__ignore_sender;

#include "__detail/__p2300.hpp"

STDEXEC_PRAGMA_POP()
