/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include "../../stdexec/execution.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(1302)
STDEXEC_PRAGMA_IGNORE_EDG(497)

namespace exec {
  struct _FAILURE_TO_CONNECT_ {
    template <class... _Message>
    struct _WHAT_ {
      struct __t {
        using __id = _WHAT_;
        using sender_concept = stdexec::sender_t;
        using completion_signatures = _WHAT_;
      };
    };
    template <class... _Message>
    using __f = stdexec::__t<_WHAT_<_Message...>>;
  };

  struct _TYPE_IS_NOT_A_VALID_SENDER_WITH_CURRENT_ENVIRONMENT_ {
    template <class _Sender, class _Env>
    struct _WITH_ { };
  };

  namespace __stl {
    using namespace stdexec;

    struct __data_placeholder;

    template <class...>
    inline constexpr bool __never_true = false;

    template <class _Env>
    struct __receiver_placeholder {
      template <__completion_tag _Tag, class... _As>
      friend void tag_invoke(_Tag, __receiver_placeholder, _As&&...) noexcept {
        static_assert(
          __never_true<_Tag, _As...>, "we should never be instantiating the body of this function");
      }

      template <same_as<__receiver_placeholder> _Self>
      [[noreturn]] friend _Env tag_invoke(get_env_t, _Self) noexcept {
        static_assert(
          __never_true<_Self>, "we should never be instantiating the body of this function");
        std::terminate();
      }
    };

    template <class _Kernel, class _Receiver, class _Completions>
    using __data_t =
      decltype(__declval<_Kernel&>().get_data(__declval<_Receiver&>(), (_Completions*) nullptr));

    struct __with_dummy_transform_sender {
      int transform_sender;
    };

    template <class _Kernel>
    struct __test_for_transform_sender
      : _Kernel
      , __with_dummy_transform_sender { };

    template <class _Kernel>
    concept __lacks_transform_sender = //
      requires {
        {
          &__test_for_transform_sender<_Kernel>::transform_sender
        } -> same_as<int __test_for_transform_sender<_Kernel>::*>;
      };

    template <class _Kernel, class _Sender, class _Env>
    using __tfx_sender_ = decltype(__declval<_Kernel&>().transform_sender(
      __declval<_Sender>(),
      __declval<__data_placeholder&>(),
      __declval<__receiver_placeholder<_Env>&>()));

    struct __sender_transform_failed {
      using __t = __sender_transform_failed;
    };

    template <class _Kernel, class _Sender, class _Env>
    auto __transform_sender_t_() {
      if constexpr (__lacks_transform_sender<_Kernel>) {
        return __mtype<_Sender>{};
      } else {
        if constexpr (__mvalid<__tfx_sender_, _Kernel, _Sender, _Env>) {
          return __mtype<__tfx_sender_<_Kernel, _Sender, _Env>>{};
        } else {
          return __sender_transform_failed{};
        }
      }
      STDEXEC_UNREACHABLE();
    }

    template <class _Kernel, class _Sender, class _Data, class _Receiver>
    auto __transform_sender(_Kernel& __kernel, _Sender&& __sndr, _Data& __data, _Receiver& __rcvr) {
      if constexpr (__lacks_transform_sender<_Kernel>) {
        return (_Sender&&) __sndr;
      } else {
        return __kernel.transform_sender((_Sender&&) __sndr, __data, __rcvr);
      }
      STDEXEC_UNREACHABLE();
    }

    template <class _Kernel, class _Sender, class _Env>
    using __transform_sender_t =
      stdexec::__t<decltype(__stl::__transform_sender_t_<_Kernel, _Sender, _Env>())>;

    template <class _Kernel, class _Receiver, class _Tag, class... _As>
    using __set_result_t = decltype(__declval<_Kernel&>().set_result(
      _Tag{},
      __declval<__data_placeholder&>(),
      __declval<_Receiver&>(),
      __declval<_As>()...));

    template <class _Kernel, class _Env>
    using __env_t = decltype(__declval<_Kernel&>().get_env(__declval<_Env>()));

    template <class _Kernel, class _Env, class _Tag, class... _As>
    auto __completions_from_sig(_Tag (*)(_As...))
      -> __set_result_t<_Kernel, __receiver_placeholder<_Env>, _Tag, _As...>;

    template <class _Kernel, class _Env, class _Sig>
    using __completions_from_sig_t = decltype(__stl::__completions_from_sig((_Sig*) nullptr));

    template <class... _Completions>
    auto __all_completions(_Completions*...)
      -> __minvoke< __mconcat<__munique<__q<completion_signatures>>>, _Completions...>;

    template <class _Kernel, class _Env, class... _Sigs>
    auto __compute_completions_(completion_signatures<_Sigs...>*)
      -> decltype(__stl::__all_completions(
        static_cast<__completions_from_sig_t<_Kernel, _Env, _Sigs>>(nullptr)...));

    template <class _Kernel, class _Env, class _NoCompletions>
    auto __compute_completions_(_NoCompletions*) -> _NoCompletions;

    template <class _Kernel, class _Env, class _Completions>
    using __compute_completions_t =
      decltype(__stl::__compute_completions_<_Kernel, _Env>((_Completions*) nullptr));

    template <class _Kernel, class _Data, class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __state {
        template <class... _Sigs>
        __state(_Kernel __kernel, _Receiver __rcvr, completion_signatures<_Sigs...>* __cmpl)
          : __rcvr_((_Receiver&&) __rcvr)
          , __kernel_{(_Kernel&&) __kernel}
          , __data_(__kernel_.get_data(__rcvr_, __cmpl)) {
        }

        _Receiver __rcvr_;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Kernel __kernel_;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Data __data_;
      };

      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;
        __state* __state_;

        template < //
          __completion_tag _Tag,
          same_as<__t> _Self,
          class... _As>
          requires __mvalid<__set_result_t, _Kernel, _Receiver, _Tag, _As...>
        friend void tag_invoke(_Tag __tag, _Self __self, _As&&... __as) noexcept {
          __state& __st = *__self.__state_;
          (void) __st.__kernel_.set_result(__tag, __st.__data_, __st.__rcvr_, (_As&&) __as...);
        }

        template <same_as<get_env_t> _Tag, same_as<__t> _Self>
        friend auto tag_invoke(_Tag, _Self __self) noexcept
          -> __env_t<_Kernel, env_of_t<_Receiver>> {
          __state& __st = *__self.__state_;
          static_assert(noexcept(__st.__kernel_.get_env(stdexec::get_env(__st.__rcvr_))));
          return __st.__kernel_.get_env(stdexec::get_env(__st.__rcvr_));
        }
      };
    };

    template <class _Sender, class _Kernel, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      using __env_t = __stl::__env_t<_Kernel, env_of_t<_Receiver>>;
      using __sender_t = __transform_sender_t<_Kernel, _Sender, env_of_t<_Receiver>>;
      using __base_completions_t = completion_signatures_of_t<__sender_t, __env_t>;
      using __completions_t = __compute_completions_t<_Kernel, __env_t, __base_completions_t>;
      using __data_t = __stl::__data_t<_Kernel, _Receiver, __completions_t>;
      using __receiver_id = __receiver<_Kernel, __data_t, _ReceiverId>;
      using __receiver_t = stdexec::__t<__receiver_id>;
      using __state_t = typename __receiver_id::__state;

      struct __t : __immovable {
        using __id = __operation;
        __state_t __state_;
        connect_result_t<__sender_t, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Kernel __kernel, _Receiver __rcvr)
          : __state_{(_Kernel&&) __kernel, (_Receiver&&) __rcvr, (__completions_t*) nullptr}
          , __op_(stdexec::connect(
              __stl::__transform_sender(
                __state_.__kernel_,
                (_Sender&&) __sndr,
                __state_.__data_,
                __state_.__rcvr_),
              __receiver_t{&__state_})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__state_.__kernel_.start(
            __self.__op_, __self.__state_.__data_, __self.__state_.__rcvr_);
        }
      };
    };

    template <class _Self>
    __minvoke<__id_<>, _Self> __is_derived_sender_(const _Self&);

    template <class _Self, class _DerivedId>
    concept __is_derived_sender = //
      requires(_Self&& __self) {
        { __is_derived_sender_((_Self&&) __self) } -> same_as<_DerivedId>;
      };

    template <class _DerivedId, class _Sender, class _Kernel>
    struct __sender {
      template <class _Self, class _Receiver>
      using __operation_t = //
        stdexec::__t<
          __operation< __copy_cvref_t<_Self, _Sender>, _Kernel, stdexec::__id<_Receiver>>>;

      struct __t {
        using __id = _DerivedId;
        using sender_concept = stdexec::sender_t;
        _Sender __sndr_;
        _Kernel __kernel_;

        template <class... _As>
          requires constructible_from<_Kernel, _As...>
        __t(_Sender __sndr, _As&&... __as)
          : __sndr_((_Sender&&) __sndr)
          , __kernel_{(_As&&) __as...} {
        }

        template <__is_derived_sender<_DerivedId> _Self, receiver _Receiver>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return {((_Self&&) __self).__sndr_, ((_Self&&) __self).__kernel_, (_Receiver&&) __rcvr};
        }

        template <class _Self, class _Env>
        using __new_sender_t = __transform_sender_t<_Kernel, __copy_cvref_t<_Self, _Sender>, _Env>;

        template <class _Env>
        using __new_env_t = __env_t<_Kernel, _Env>;

        template <class _NewSender, class _NewEnv>
        using __pre_completions_t =
          __call_result_t<get_completion_signatures_t, _NewSender, _NewEnv>;

        template <class _NewEnv, class _PreCompletions>
        using __completions_t = __compute_completions_t<_Kernel, _NewEnv, _PreCompletions>;

        template <class _Env>
        using __diagnostic_t =    //
          __minvoke<              //
            _FAILURE_TO_CONNECT_, //
            _TYPE_IS_NOT_A_VALID_SENDER_WITH_CURRENT_ENVIRONMENT_::_WITH_<_DerivedId, _Env>>;

        template <__is_derived_sender<_DerivedId> _Self, class _Env>
        static auto __new_completion_sigs_type() {
          using _NewSender = __new_sender_t<_Self, _Env>;
          if constexpr (sender<_NewSender>) {
            using _NewEnv = __new_env_t<_Env>;
            if constexpr (__mvalid<__pre_completions_t, _NewSender, _NewEnv>) {
              using _Completions =
                __completions_t<_NewEnv, __pre_completions_t<_NewSender, _NewEnv>>;
              if constexpr (__valid_completion_signatures<_Completions, _Env>) {
                return (_Completions(*)()) nullptr;
              } else {
                // assume this is an error message and return it directly
                return (_Completions(*)()) nullptr;
              }
            } else {
              return (__diagnostic_t<_Env>(*)()) nullptr;
            }
          } else if constexpr (same_as<_NewSender, __sender_transform_failed>) {
            return (__diagnostic_t<_Env>(*)()) nullptr;
          } else {
            // assume this is an error message and return it directly
            return (_NewSender(*)()) nullptr;
          }
        }

        template <class _Self, class _Env>
        using __new_completions_t = decltype(__new_completion_sigs_type<_Self, _Env>()());

        template <__is_derived_sender<_DerivedId> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
          -> __new_completions_t<_Self, _Env> {
          return {};
        }

        friend auto tag_invoke(stdexec::get_env_t, const __t& __self) noexcept
          -> env_of_t<const _Sender&> {
          return stdexec::get_env(__self.__sndr_);
        }
      };
    };
  } // namespace __stl

  template <class _DerivedId, class _Sender, class _Kernel>
  using __sender_facade = __stl::__sender<_DerivedId, _Sender, _Kernel>;

  struct __default_kernel {
    struct __no_data { };

    template <class _Sender>
    static _Sender&& transform_sender(           //
      _Sender&& __sndr,                          //
      [[maybe_unused]] stdexec::__ignore __data, //
      [[maybe_unused]] stdexec::__ignore __rcvr) noexcept {
      return (_Sender&&) __sndr;
    }

    template <class _Env>
    static _Env get_env(_Env&& __env) noexcept {
      static_assert(stdexec::__nothrow_move_constructible<_Env>);
      return (_Env&&) __env;
    }

    static __no_data get_data(                   //
      [[maybe_unused]] stdexec::__ignore __rcvr, //
      [[maybe_unused]] void* __compl_sigs) noexcept {
      return {};
    }

    template <class _Op>
    static void start(                                      //
      _Op& __op,                                            //
      [[maybe_unused]] stdexec::__ignore __data,            //
      [[maybe_unused]] stdexec::__ignore __rcvr) noexcept { //
      stdexec::start(__op);
    }

    template <class _Tag, class _Receiver, class... _As>
    static auto set_result(                      //
      _Tag __tag,                                //
      [[maybe_unused]] stdexec::__ignore __data, //
      _Receiver& __rcvr,                         //
      _As&&... __as) noexcept                    //
      -> stdexec::completion_signatures<_Tag(_As...)>* {
      __tag((_Receiver&&) __rcvr, (_As&&) __as...);
      return {};
    }
  };
} // namespace exec

STDEXEC_PRAGMA_POP()
