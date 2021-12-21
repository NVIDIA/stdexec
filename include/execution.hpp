/*
 * Copyright (c) NVIDIA
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

#include <any>
#include <atomic>
#include <barrier>
#include <cassert>
#include <condition_variable>
#include <stdexcept>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>

#include <__utility.hpp>
#include <functional.hpp>
#include <concepts.hpp>
#include <coroutine.hpp>
#include <stop_token.hpp>

#if defined(__clang__)
#define _STRINGIZE(__arg) #__arg
#define _PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define _PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define _PRAGMA_IGNORE(__arg) _Pragma(_STRINGIZE(GCC diagnostic ignored __arg))
#else
#define _PRAGMA_PUSH()
#define _PRAGMA_POP()
#define _PRAGMA_IGNORE(__arg)
#endif

_PRAGMA_PUSH()
_PRAGMA_IGNORE("-Wundefined-internal")
// inline namespace reopened as a non-inline namespace:
_PRAGMA_IGNORE("-Winline-namespace-reopened-noninline")

namespace std::execution {
  template <template <template <class...> class, template <class...> class> class>
    struct __test_has_values;

  template <template <template <class...> class> class>
    struct __test_has_errors;

  template <class _T>
    concept __has_sender_types = requires {
      typename __test_has_values<_T::template value_types>;
      typename __test_has_errors<_T::template error_types>;
      typename bool_constant<_T::sends_done>;
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  inline namespace __receiver_cpo {
    inline constexpr struct set_value_t {
      template <class _Receiver, class... _As>
        requires tag_invocable<set_value_t, _Receiver, _As...>
      void operator()(_Receiver&& __rcvr, _As&&... __as) const
        noexcept(nothrow_tag_invocable<set_value_t, _Receiver, _As...>) {
        (void) tag_invoke(set_value_t{}, (_Receiver&&) __rcvr, (_As&&) __as...);
      }
    } set_value{};

    inline constexpr struct set_error_t {
      template <class _Receiver, class _Error>
        requires tag_invocable<set_error_t, _Receiver, _Error>
      void operator()(_Receiver&& __rcvr, _Error&& __err) const
        noexcept(nothrow_tag_invocable<set_error_t, _Receiver, _Error>) {
        (void) tag_invoke(set_error_t{}, (_Receiver&&) __rcvr, (_Error&&) __err);
      }
    } set_error {};

    inline constexpr struct set_done_t {
      template <class _Receiver>
        requires tag_invocable<set_done_t, _Receiver>
      void operator()(_Receiver&& __rcvr) const
        noexcept(nothrow_tag_invocable<set_done_t, _Receiver>) {
        (void) tag_invoke(set_done_t{}, (_Receiver&&) __rcvr);
      }
    } set_done{};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template <class _Receiver, class _Error = exception_ptr>
    concept receiver =
      move_constructible<remove_cvref_t<_Receiver>> &&
      constructible_from<remove_cvref_t<_Receiver>, _Receiver> &&
      requires(remove_cvref_t<_Receiver>&& __rcvr, _Error&& __err) {
        { set_done(std::move(__rcvr)) } noexcept;
        { set_error(std::move(__rcvr), (_Error&&) __err) } noexcept;
      };

  template <class _Receiver, class... _An>
    concept receiver_of =
      receiver<_Receiver> &&
      requires(remove_cvref_t<_Receiver>&& __rcvr, _An&&... an) {
        set_value((remove_cvref_t<_Receiver>&&) __rcvr, (_An&&) an...);
      };

  // NOT TO SPEC
  template <class _Receiver, class..._As>
    inline constexpr bool nothrow_receiver_of =
      receiver_of<_Receiver, _As...> &&
      nothrow_tag_invocable<set_value_t, _Receiver, _As...>;

  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  // NOT TO SPEC
  namespace __completion_signatures {
    template <same_as<set_value_t> _Tag, class _Ty = __q<__types>, class... _Args>
      __types<__minvoke<_Ty, _Args...>> __test(_Tag(*)(_Args...));
    template <same_as<set_error_t> _Tag, class _Ty = __q<__types>, class _Error>
      __types<__minvoke1<_Ty, _Error>> __test(_Tag(*)(_Error));
    template <same_as<set_done_t> _Tag, class _Ty = __q<__types>>
      __types<__minvoke<_Ty>> __test(_Tag(*)());
    template <class, class = void>
      __types<> __test(...);

    template <class _Sig, class _Tag, class _Ty = __q<__types>>
      using __signal_args_t =
        decltype(__test<_Tag, _Ty>((_Sig*) nullptr));

    template <class _Sig>
      concept __completion_signal =
        requires { typename __id<decltype(__test((_Sig*) nullptr))>; };

    template <class... _Sigs>
      struct completion_signatures {
        struct type {
          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types =
              __minvoke<
                __concat<__q<_Variant>>,
                __signal_args_t<_Sigs, set_value_t, __q<_Tuple>>...>;

          template <template <class...> class _Variant>
            using error_types =
              __minvoke<
                __concat<__q<_Variant>>,
                __signal_args_t<_Sigs, set_error_t, __q1<__id>>...>;

          static constexpr bool sends_done =
            __minvoke<
              __concat<__count>,
              __signal_args_t<_Sigs, set_done_t>...>::value != 0;
        };
      };
  } // namespace __completion_signatures

  template <__completion_signatures::__completion_signal... _Sigs>
    using completion_signatures =
      __t<__completion_signatures::completion_signatures<_Sigs...>>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.traits]
  namespace __sender_base {
    struct sender_base {};
  }
  using __sender_base::sender_base;

  inline namespace __sender_traits {
    namespace __impl {
      struct __default_context {
        friend void tag_invoke(set_value_t, __default_context&&, auto&&...) {}
        friend void tag_invoke(set_error_t, __default_context&&, auto&&) noexcept {}
        friend void tag_invoke(set_done_t, __default_context&&) noexcept {}
      };

      template <class _Sender>
        struct __typed_sender {
          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types = typename _Sender::template value_types<_Tuple, _Variant>;
          template <template <class...> class _Variant>
            using error_types = typename _Sender::template error_types<_Variant>;
          static constexpr bool sends_done = _Sender::sends_done;
        };

      struct get_sender_traits_t {
        template <class _Sender, class _Receiver>
        constexpr auto operator()(_Sender&&, _Receiver&&) const noexcept {
          static_assert(sizeof(_Sender), "Incomplete type used with get_sender_traits");
          static_assert(sizeof(_Receiver), "Incomplete type used with get_sender_traits");
          if constexpr (tag_invocable<get_sender_traits_t, _Sender, _Receiver>) {
            if constexpr (is_void_v<tag_invoke_result_t<get_sender_traits_t, _Sender, _Receiver>>) {
              return sender_base{};
            } else {
              return tag_invoke_result_t<get_sender_traits_t, _Sender, _Receiver>{};
            }
          } else if constexpr (__has_sender_types<remove_cvref_t<_Sender>>) {
            return __typed_sender<remove_cvref_t<_Sender>>{};
          } else if constexpr (derived_from<remove_cvref_t<_Sender>, sender_base>) {
            return sender_base{};
          } else if constexpr (__awaitable<_Sender>) {
            using _Result = __await_result_t<_Sender>;
            if constexpr (is_void_v<_Result>) {
              return completion_signatures<set_value_t(), set_error_t(exception_ptr)>{};
            } else {
              return completion_signatures<set_value_t(_Result), set_error_t(exception_ptr)>{};
            }
          } else {
            struct __no_sender_traits{
              using __this_is_not_a_sender = _Sender;
            };
            return __no_sender_traits{};
          }
        }

        template <class _Sender>
        constexpr auto operator()(_Sender&& __sndr) const noexcept {
          return (*this)(__sndr, __default_context{});
        }
      };
    } // namespace __impl
    using __impl::__default_context;

    using __impl::get_sender_traits_t;
    inline constexpr get_sender_traits_t get_sender_traits {};
  } // namespace __sender_traits

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.traits]
  template <class _Sender, class _Receiver = __default_context>
  struct sender_traits
    : __call_result_t<get_sender_traits_t, _Sender, _Receiver> {};

  template <class _Sender>
    concept __invalid_sender_traits =
      same_as<typename sender_traits<_Sender>::__this_is_not_a_sender, _Sender>;

  template <class _Sender>
    concept __valid_sender_traits =
      !__invalid_sender_traits<_Sender>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  // NOT TO SPEC (YET)
  template <class _Sender>
    concept sender =
      move_constructible<remove_cvref_t<_Sender>> &&
      __valid_sender_traits<_Sender>;

  template <class _Sender, class _Receiver = __default_context>
    concept typed_sender =
      sender<_Sender> &&
      __has_sender_types<sender_traits<_Sender, _Receiver>>;

  struct __not_a_variant {
    __not_a_variant() = delete;
  };
  template <class... _Ts>
    using __variant =
      __minvoke<
        __if<
          __bool<sizeof...(_Ts) != 0>,
          __transform<__q1<decay_t>, __munique<__q<variant>>>,
          __constant<__not_a_variant>>,
        _Ts...>;

  template <class... _Ts>
    using __decayed_tuple = tuple<decay_t<_Ts>...>;

  template <class _Sender,
            class _Receiver = __default_context,
            template <class...> class _Tuple = __decayed_tuple,
            template <class...> class _Variant = __variant>
      requires typed_sender<_Sender, _Receiver>
    using value_types_of_t =
      typename sender_traits<_Sender, _Receiver>::template
        value_types<_Tuple, _Variant>;

  template <class _Sender,
            class _Receiver = __default_context,
            template <class...> class _Variant = __variant>
      requires typed_sender<_Sender, _Receiver>
    using error_types_of_t =
      typename sender_traits<_Sender, _Receiver>::template
        error_types<_Variant>;

  template <class _Sender,
            class _Receiver = __default_context,
            class _Tuple = __q<__decayed_tuple>,
            class _Variant = __q<__variant>>
      requires typed_sender<_Sender, _Receiver>
    using __value_types_of_t =
      value_types_of_t<
        _Sender, _Receiver, _Tuple::template __f, _Variant::template __f>;

  template <class _Sender,
            class _Receiver = __default_context,
            class _Variant = __q<__variant>>
      requires typed_sender<_Sender, _Receiver>
    using __error_types_of_t =
      error_types_of_t<_Sender, _Receiver, _Variant::template __f>;

  template <class _Sender, class _Receiver>
    using __sends_done =
      __bool<sender_traits<_Sender, _Receiver>::sends_done>;

  template <class _Sender, class... _Ts>
    concept sender_of =
      typed_sender<_Sender> &&
      same_as<
        __types<_Ts...>,
        value_types_of_t<_Sender, __default_context, __types, __single_t>>;

  template <class _Sender, class _Receiver = __default_context>
    using __single_sender_value_t =
      value_types_of_t<_Sender, _Receiver, __single_or_void_t, __single_t>;

  template <class _Sender, class _Receiver = __default_context>
    concept __single_typed_sender =
      typed_sender<_Sender, _Receiver> &&
      __valid<__single_sender_value_t, _Sender, _Receiver>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  inline namespace __schedule {
    inline constexpr struct schedule_t {
      template <class _Scheduler>
        requires tag_invocable<schedule_t, _Scheduler> &&
          sender<tag_invoke_result_t<schedule_t, _Scheduler>>
      auto operator()(_Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<schedule_t, _Scheduler>) {
        return tag_invoke(schedule_t{}, (_Scheduler&&) __sched);
      }
    } schedule {};
  }

  inline namespace __scheduler_queries {
    namespace __impl {
      struct forwarding_scheduler_query_t {
        template <class _Tag>
        constexpr bool operator()(_Tag __tag) const noexcept {
          if constexpr (nothrow_tag_invocable<forwarding_scheduler_query_t, _Tag> &&
                        is_invocable_r_v<bool, tag_t<tag_invoke>,
                                         forwarding_scheduler_query_t, _Tag>) {
            return tag_invoke(*this, (_Tag&&) __tag);
          } else {
            return false;
          }
        }
      };
    } // namespace __impl

    using __impl::forwarding_scheduler_query_t;
    inline constexpr forwarding_scheduler_query_t forwarding_scheduler_query{};

    template <class _Tag>
      concept __scheduler_query =
        forwarding_scheduler_query(_Tag{});
  } // namespace __scheduler_queries

  inline namespace __sender_queries {
    namespace __impl {
      template <__one_of<set_value_t, set_error_t, set_done_t> _CPO>
        struct get_completion_scheduler_t;
    }
    using __impl::get_completion_scheduler_t;
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.schedulers]
  template <class _Scheduler>
    concept scheduler =
      copy_constructible<remove_cvref_t<_Scheduler>> &&
      equality_comparable<remove_cvref_t<_Scheduler>> &&
      requires(_Scheduler&& __sched, const get_completion_scheduler_t<set_value_t> __tag) {
        { schedule((_Scheduler&&) __sched) } -> sender_of;
        { tag_invoke(__tag, schedule((_Scheduler&&) __sched)) } -> same_as<remove_cvref_t<_Scheduler>>;
      };

  // NOT TO SPEC
  template <scheduler _Scheduler>
    using schedule_result_t = __call_result_t<schedule_t, _Scheduler>;

  // [execution.general.queries], general queries
  inline namespace __general_queries {
    namespace __impl {
      // TODO: implement allocator concept
      template <class _T0>
        concept __allocator = true;

      struct get_scheduler_t {
        template <class _T>
          requires nothrow_tag_invocable<get_scheduler_t, __cref_t<_T>> &&
            scheduler<tag_invoke_result_t<get_scheduler_t, __cref_t<_T>>>
        auto operator()(_T&& __t) const
          noexcept(nothrow_tag_invocable<get_scheduler_t, __cref_t<_T>>)
          -> tag_invoke_result_t<get_scheduler_t, __cref_t<_T>> {
          return tag_invoke(get_scheduler_t{}, std::as_const(__t));
        }
        // NOT TO SPEC
        auto operator()() const noexcept;
      };

      struct get_delegee_scheduler_t {
        template <class _T>
          requires nothrow_tag_invocable<get_delegee_scheduler_t, __cref_t<_T>> &&
            scheduler<tag_invoke_result_t<get_delegee_scheduler_t, __cref_t<_T>>>
        auto operator()(_T&& __t) const
          noexcept(nothrow_tag_invocable<get_delegee_scheduler_t, __cref_t<_T>>)
          -> tag_invoke_result_t<get_delegee_scheduler_t, __cref_t<_T>> {
          return tag_invoke(get_delegee_scheduler_t{}, std::as_const(__t));
        }
      };

      struct get_allocator_t {
        template <class _T>
          requires nothrow_tag_invocable<get_allocator_t, __cref_t<_T>> &&
            __allocator<tag_invoke_result_t<get_allocator_t, __cref_t<_T>>>
        tag_invoke_result_t<get_allocator_t, __cref_t<_T>> operator()(_T&& __t) const
          noexcept(nothrow_tag_invocable<get_allocator_t, __cref_t<_T>>) {
          return tag_invoke(get_allocator_t{}, std::as_const(__t));
        }
        // NOT TO SPEC
        auto operator()() const noexcept;
      };

      struct get_stop_token_t {
        template <class _T>
          requires tag_invocable<get_stop_token_t, __cref_t<_T>> &&
            stoppable_token<tag_invoke_result_t<get_stop_token_t, __cref_t<_T>>>
        tag_invoke_result_t<get_stop_token_t, __cref_t<_T>> operator()(_T&& __t) const
          noexcept(nothrow_tag_invocable<get_stop_token_t, __cref_t<_T>>) {
          return tag_invoke(get_stop_token_t{}, std::as_const(__t));
        }
        never_stop_token operator()(auto&&) const noexcept {
          return {};
        }
        // NOT TO SPEC
        auto operator()() const noexcept;
      };
    } // namespace __impl

    using __impl::get_allocator_t;
    using __impl::get_scheduler_t;
    using __impl::get_delegee_scheduler_t;
    using __impl::get_stop_token_t;
    inline constexpr get_scheduler_t get_scheduler{};
    inline constexpr get_delegee_scheduler_t get_delegee_scheduler{};
    inline constexpr get_allocator_t get_allocator{};
    inline constexpr get_stop_token_t get_stop_token{};
  } // namespace __general_queries

  template <class _T>
    using stop_token_of_t =
      remove_cvref_t<decltype(get_stop_token(__declval<_T>()))>;

  // [execution.receivers.queries], receiver queries
  inline namespace __receiver_queries {
    namespace __impl {
      struct forwarding_receiver_query_t {
        template <class _Tag>
        constexpr bool operator()(_Tag __tag) const noexcept {
          if constexpr (nothrow_tag_invocable<forwarding_receiver_query_t, _Tag> &&
                        is_invocable_r_v<bool, tag_t<tag_invoke>,
                                         forwarding_receiver_query_t, _Tag>) {
            return tag_invoke(*this, (_Tag&&) __tag);
          } else {
            return __none_of<_Tag, set_value_t, set_error_t, set_done_t>;
          }
        }
      };
    }

    using __impl::forwarding_receiver_query_t;
    inline constexpr forwarding_receiver_query_t forwarding_receiver_query{};

    template <class _Tag>
      concept __receiver_query =
        forwarding_receiver_query(_Tag{});
  } // namespace __receiver_queries

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  inline namespace __start {
    inline constexpr struct start_t {
      template <class _O>
        requires tag_invocable<start_t, _O&>
      void operator()(_O& o) const noexcept(nothrow_tag_invocable<start_t, _O&>) {
        (void) tag_invoke(start_t{}, o);
      }
    } start {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template <class _O>
    concept operation_state =
      destructible<_O> &&
      is_object_v<_O> &&
      requires (_O& o) {
        {start(o)} noexcept;
      };

  inline namespace __as_awaitable {
    struct as_awaitable_t;
    extern const as_awaitable_t as_awaitable;
  }

  /////////////////////////////////////////////////////////////////////////////
  // __connect_awaitable_
  inline namespace __connect_awaitable_ {
    namespace __impl {
      struct __promise_base {
        __coro::suspend_always initial_suspend() noexcept {
          return {};
        }
        [[noreturn]] __coro::suspend_always final_suspend() noexcept {
          terminate();
        }
        [[noreturn]] void unhandled_exception() noexcept {
          terminate();
        }
        [[noreturn]] void return_void() noexcept {
          terminate();
        }
        template <class _Fun>
        auto yield_value(_Fun&& __fun) noexcept {
          struct awaiter {
            _Fun&& __fun_;
            bool await_ready() noexcept {
              return false;
            }
            void await_suspend(__coro::coroutine_handle<>)
              noexcept(__nothrow_callable<_Fun>) {
              // If this throws, the runtime catches the exception,
              // resumes the __connect_awaitable coroutine, and immediately
              // rethrows the exception. The end result is that an
              // exception_ptr to the exception gets passed to set_error.
              ((_Fun &&) __fun_)();
            }
            [[noreturn]] void await_resume() noexcept {
              terminate();
            }
          };
          return awaiter{(_Fun &&) __fun};
        }
      };

      struct __operation_base {
        __coro::coroutine_handle<> __coro_;

        explicit __operation_base(__coro::coroutine_handle<> __hcoro) noexcept
          : __coro_(__hcoro) {}

        __operation_base(__operation_base&& __other) noexcept
          : __coro_(std::exchange(__other.__coro_, {})) {}

        ~__operation_base() {
          if (__coro_)
            __coro_.destroy();
        }

        friend void tag_invoke(start_t, __operation_base& __self) noexcept {
          __self.__coro_.resume();
        }
      };

      template <class _ReceiverId>
        struct __promise;

      template <class _ReceiverId>
        struct __operation : __operation_base {
          using promise_type = __promise<_ReceiverId>;
          using __operation_base::__operation_base;
        };

      template <class _ReceiverId>
        struct __promise : __promise_base {
          using _Receiver = __t<_ReceiverId>;

          template <class _T0>
          explicit __promise(_T0&, _Receiver& __rcvr) noexcept
            : __rcvr_(__rcvr)
          {}

          __coro::coroutine_handle<> unhandled_done() noexcept {
            set_done(std::move(__rcvr_));
            // Returning noop_coroutine here causes the __connect_awaitable
            // coroutine to never resume past the point where it co_await's
            // the awaitable.
            return __coro::noop_coroutine();
          }

          __operation<_ReceiverId> get_return_object() noexcept {
            return __operation<_ReceiverId>{
              __coro::coroutine_handle<__promise>::from_promise(*this)};
          }

          template <class _Awaitable>
          _Awaitable&& await_transform(_Awaitable&& __await) noexcept {
            return (_Awaitable&&) __await;
          }

          template <class _Awaitable>
            requires tag_invocable<as_awaitable_t, _Awaitable, __promise&>
          auto await_transform(_Awaitable&& __await)
              noexcept(nothrow_tag_invocable<as_awaitable_t, _Awaitable, __promise&>)
              -> tag_invoke_result_t<as_awaitable_t, _Awaitable, __promise&> {
            return tag_invoke(as_awaitable, (_Awaitable&&) __await, *this);
          }

          // Pass through receiver queries
          template <__receiver_query _CPO, class... _As>
            requires __callable<_CPO, const _Receiver&, _As...>
          friend decltype(auto) tag_invoke(_CPO cpo, const __promise& __self, _As&&... __as)
            noexcept(__nothrow_callable<_CPO, const _Receiver&, _As...>) {
            return ((_CPO&&) cpo)(as_const(__self.__rcvr_), (_As&&) __as...);
          }

          _Receiver& __rcvr_;
        };

      template <class _Receiver>
        using __promise_t = __promise<__x<remove_cvref_t<_Receiver>>>;

      template <class _Receiver>
        using __operation_t = __operation<__x<remove_cvref_t<_Receiver>>>;
    } // namespace __impl

    inline constexpr struct __fn {
     private:
      template <class _Receiver, class... _Args>
        using __nothrow_ = bool_constant<nothrow_receiver_of<_Receiver, _Args...>>;

      template <class _Awaitable, class _Receiver>
      static __impl::__operation_t<_Receiver> __co_impl(_Awaitable __await, _Receiver __rcvr) {
        using __result_t = __await_result_t<_Awaitable, __impl::__promise_t<_Receiver>>;
        exception_ptr __eptr;
        try {
          // This is a bit mind bending control-flow wise.
          // We are first evaluating the co_await expression.
          // Then the result of that is passed into a lambda
          // that curries a reference to the result into another
          // lambda which is then returned to 'co_yield'.
          // The 'co_yield' expression then invokes this lambda
          // after the coroutine is suspended so that it is safe
          // for the receiver to destroy the coroutine.
          auto __fun = [&]<bool _NoThrow>(bool_constant<_NoThrow>, auto&&... __as) noexcept {
            return [&]() noexcept(_NoThrow) -> void {
              set_value((_Receiver&&) __rcvr, (add_rvalue_reference_t<__result_t>) __as...);
            };
          };
          if constexpr (is_void_v<__result_t>)
            co_yield (co_await (_Awaitable &&) __await, __fun(__nothrow_<_Receiver>{}));
          else
            co_yield __fun(__nothrow_<_Receiver, __result_t>{}, co_await (_Awaitable &&) __await);
        } catch (...) {
          __eptr = current_exception();
        }
        co_yield [&]() noexcept -> void {
          set_error((_Receiver&&) __rcvr, (exception_ptr&&) __eptr);
        };
      }
     public:
      template <receiver _Receiver, __awaitable<__impl::__promise_t<_Receiver>> _Awaitable>
        requires receiver_of<_Receiver, __await_result_t<_Awaitable, __impl::__promise_t<_Receiver>>>
      __impl::__operation_t<_Receiver> operator()(_Awaitable&& __await, _Receiver&& __rcvr) const {
        return __co_impl((_Awaitable&&) __await, (_Receiver&&) __rcvr);
      }
      template <receiver _Receiver, __awaitable<__impl::__promise_t<_Receiver>> _Awaitable>
        requires same_as<void, __await_result_t<_Awaitable, __impl::__promise_t<_Receiver>>> &&
          receiver_of<_Receiver>
      __impl::__operation_t<_Receiver> operator()(_Awaitable&& __await, _Receiver&& __rcvr) const {
        return __co_impl((_Awaitable&&) __await, (_Receiver&&) __rcvr);
      }
    } __connect_awaitable{};
  } // namespace __connect_awaitable_

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  inline namespace __connect {
    inline constexpr struct connect_t {
      template <sender _Sender, receiver _Receiver>
        requires tag_invocable<connect_t, _Sender, _Receiver> &&
          operation_state<tag_invoke_result_t<connect_t, _Sender, _Receiver>>
      auto operator()(_Sender&& __sndr, _Receiver&& __rcvr) const
        noexcept(nothrow_tag_invocable<connect_t, _Sender, _Receiver>)
        -> tag_invoke_result_t<connect_t, _Sender, _Receiver> {
        return tag_invoke(connect_t{}, (_Sender&&) __sndr, (_Receiver&&) __rcvr);
      }
      template <class _Awaitable, receiver _Receiver>
        requires (!tag_invocable<connect_t, _Awaitable, _Receiver>) &&
          __awaitable<_Awaitable, __connect_awaitable_::__impl::__promise_t<_Receiver>>
      __connect_awaitable_::__impl::__operation_t<_Receiver> operator()(_Awaitable&& __await, _Receiver&& __rcvr) const {
        return __connect_awaitable((_Awaitable&&) __await, (_Receiver&&) __rcvr);
      }
    } connect {};
  }

  template <class _Sender, class _Receiver>
    using connect_result_t = __call_result_t<connect_t, _Sender, _Receiver>;

  template <class _Sender, class _Receiver>
    concept __has_nothrow_connect =
      noexcept(connect(__declval<_Sender>(), __declval<_Receiver>()));

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  template <class _Sender, class _Receiver>
    concept sender_to =
      sender<_Sender> &&
      receiver<_Receiver> &&
      requires (_Sender&& __sndr, _Receiver&& __rcvr) {
        connect((_Sender&&) __sndr, (_Receiver&&) __rcvr);
      };

  template <
      class _Sender,
      class _Receiver,
      class _Fun,
      class _Tfx = __q1<__id>,
      class _Continuation = __q<__types>>
    using __tfx_sender_values =
      __value_types_of_t<
        _Sender,
        _Receiver,
        __transform<_Tfx, __bind_front_q<invoke_result_t, _Fun>>,
        _Continuation>;

  template <
      class _Sender,
      class _Receiver,
      class _Fun,
      class _Tfx = __q1<__id>,
      class _Continuation = __q<__types>>
    using __tfx_sender_errors =
      __error_types_of_t<
        _Sender,
        _Receiver,
        __transform<
          __compose<__bind_front_q<invoke_result_t, _Fun>, _Tfx, __defer<__id>>,
          _Continuation>>;

  template <
      class,
      class,
      class _Fun,
      class = __q1<__id>,
      class _Continuation = __q<__types>>
    using __tfx_sender_done =
      __minvoke<_Continuation, invoke_result_t<_Fun>>;

  template <class _Fun, class _Sender, class _WhichTfx, class _Tfx = __q1<__id>>
    concept __invocable_with_xxx_from =
      sender<_Sender> &&
        (!typed_sender<_Sender> ||
         __valid<__minvoke, _WhichTfx, _Sender, __default_context, _Fun, _Tfx>);

  template <class _Fun, class _Sender, class _Tfx = __q1<__id>>
    concept __invocable_with_values_from =
      __invocable_with_xxx_from<_Fun, _Sender, __defer<__tfx_sender_values>, _Tfx>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.queries], sender queries
  inline namespace __sender_queries {
    namespace __impl {
      template <__one_of<set_value_t, set_error_t, set_done_t> _CPO>
        struct get_completion_scheduler_t {
          template <sender _Sender>
            requires tag_invocable<get_completion_scheduler_t, const _Sender&> &&
              scheduler<tag_invoke_result_t<get_completion_scheduler_t, const _Sender&>>
          auto operator()(const _Sender& __sndr) const noexcept
              -> tag_invoke_result_t<get_completion_scheduler_t, const _Sender&> {
            // NOT TO SPEC:
            static_assert(
              nothrow_tag_invocable<get_completion_scheduler_t, const _Sender&>,
              "get_completion_scheduler<_CPO> should be noexcept");
            return tag_invoke(*this, __sndr);
          }
        };

      struct forwarding_sender_query_t {
        template <class _Tag>
        constexpr bool operator()(_Tag __tag) const noexcept {
          if constexpr (nothrow_tag_invocable<forwarding_sender_query_t, _Tag> &&
                        is_invocable_r_v<bool, tag_t<tag_invoke>,
                                         forwarding_sender_query_t, _Tag>) {
            return tag_invoke(*this, (_Tag&&) __tag);
          } else {
            return false;
          }
        }
      };
    } // namespace __impl

    using __impl::get_completion_scheduler_t;
    using __impl::forwarding_sender_query_t;

    template <__one_of<set_value_t, set_error_t, set_done_t> _CPO>
      inline constexpr get_completion_scheduler_t<_CPO> get_completion_scheduler{};

    inline constexpr forwarding_sender_query_t forwarding_sender_query{};

    template <class _Tag>
      concept __sender_query =
        forwarding_sender_query(_Tag{});
  } // namespace __sender_queries

  template <class _Sender, class _CPO>
    concept __has_completion_scheduler =
      __callable<get_completion_scheduler_t<_CPO>, _Sender>;

  template <class _Sender, class _CPO>
    using __completion_scheduler_for =
      __call_result_t<get_completion_scheduler_t<_CPO>, _Sender>;

  template <class _Fun, class _CPO, class _Sender, class... _As>
    concept __tag_invocable_with_completion_scheduler =
      __has_completion_scheduler<_Sender, _CPO> &&
      tag_invocable<_Fun, __completion_scheduler_for<_Sender, _CPO>, _Sender, _As...>;

  /////////////////////////////////////////////////////////////////////////////
  // execution::as_awaitable [execution.coro_utils.as_awaitable]
  inline namespace __as_awaitable {
    namespace __impl {
      struct __void {};
      template <class _Value>
        using __value_or_void_t =
          conditional_t<is_void_v<_Value>, __void, _Value>;
      template <class _Value>
        using __expected_t =
          variant<monostate, __value_or_void_t<_Value>, exception_ptr>;

      template <class _Value>
        struct __receiver_base {
          template <class... _Us>
            requires constructible_from<_Value, _Us...> ||
              (is_void_v<_Value> && sizeof...(_Us) == 0)
          friend void tag_invoke(set_value_t, __receiver_base&& __self, _Us&&... __us)
              noexcept(is_nothrow_constructible_v<_Value, _Us...> ||
                  is_void_v<_Value>) {
            __self.__result_->template emplace<1>((_Us&&) __us...);
            __self.__continuation_.resume();
          }

          template <class _Error>
          friend void tag_invoke(set_error_t, __receiver_base&& __self, _Error&& __err) noexcept {
            if constexpr (__decays_to<_Error, exception_ptr>)
              __self.__result_->template emplace<2>((_Error&&) __err);
            else if constexpr (__decays_to<_Error, error_code>)
              __self.__result_->template emplace<2>(make_exception_ptr(system_error(__err)));
            else
              __self.__result_->template emplace<2>(make_exception_ptr((_Error&&) __err));
            __self.__continuation_.resume();
          }

          __expected_t<_Value>* __result_;
          __coro::coroutine_handle<> __continuation_;
        };

      template <typename _PromiseId, typename _Value>
        struct __sender_awaitable_base {
          using _Promise = __t<_PromiseId>;
          struct __receiver : __receiver_base<_Value> {
            friend void tag_invoke(set_done_t, __receiver&& __self) noexcept {
              auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
                __self.__continuation_.address());
              __continuation.promise().unhandled_done().resume();
            }

            // Forward other tag_invoke overloads to the promise
            template <class... _As, __callable<_Promise&, _As...> _CPO>
            friend auto tag_invoke(_CPO cpo, const __receiver& __self, _As&&... __as)
                noexcept(__nothrow_callable<_CPO, _Promise&, _As...>)
                -> __call_result_t<_CPO, _Promise&, _As...> {
              auto __continuation = __coro::coroutine_handle<_Promise>::from_address(
                __self.__continuation_.address());
              return ((_CPO&&) cpo)(__continuation.promise(), (_As&&) __as...);
            }
          };

        bool await_ready() const noexcept {
          return false;
        }

        _Value await_resume() {
          switch (__result_.index()) {
          case 0: // receiver contract not satisfied
            assert(!"_Should never get here");
            break;
          case 1: // set_value
            if constexpr (!is_void_v<_Value>)
              return (_Value&&) std::get<1>(__result_);
            else
              return;
          case 2: // set_error
            std::rethrow_exception(std::get<2>(__result_));
          }
          terminate();
        }

       protected:
        __expected_t<_Value> __result_;
      };

      template <typename _PromiseId, typename _SenderId>
      struct __sender_awaitable
        : __sender_awaitable_base<_PromiseId, __single_sender_value_t<__t<_SenderId>, __t<_PromiseId>>> {
       private:
        using _Promise = __t<_PromiseId>;
        using _Sender = __t<_SenderId>;
        using _Base = __sender_awaitable_base<_PromiseId, __single_sender_value_t<_Sender, _Promise>>;
        using __receiver = typename _Base::__receiver;
        connect_result_t<_Sender, __receiver> __op_state_;
       public:
        __sender_awaitable(_Sender&& sender, __coro::coroutine_handle<_Promise> __hcoro)
            noexcept(__has_nothrow_connect<_Sender, __receiver>)
          : __op_state_(connect((_Sender&&)sender, __receiver{{&this->__result_, __hcoro}}))
        {}

        void await_suspend(__coro::coroutine_handle<_Promise>) noexcept {
          start(__op_state_);
        }
      };
      template <class _Promise, class _Sender>
        using __sender_awaitable_t =
          __sender_awaitable<__x<_Promise>, __x<remove_cvref_t<_Sender>>>;

      template <class _T, class _Promise>
        concept __custom_tag_invoke_awaiter =
          tag_invocable<as_awaitable_t, _T, _Promise&> &&
          __awaitable<tag_invoke_result_t<as_awaitable_t, _T, _Promise&>, _Promise>;

      template <class _Sender, class _Promise>
        using __receiver =
          typename __sender_awaitable_base<
            __x<_Promise>,
            __single_sender_value_t<_Sender, _Promise>
          >::__receiver;

      template <class _Sender, class _Promise>
        concept __awaitable_sender =
          __single_typed_sender<_Sender, _Promise> &&
          sender_to<_Sender, __receiver<_Sender, _Promise>> &&
          requires (_Promise& __promise) {
            { __promise.unhandled_done() } -> convertible_to<__coro::coroutine_handle<>>;
          };
    } // namespace __impl

    inline constexpr struct as_awaitable_t {
      template <class _T, class _Promise>
      static constexpr bool __is_noexcept() noexcept {
        if constexpr (__impl::__custom_tag_invoke_awaiter<_T, _Promise>) {
          return nothrow_tag_invocable<as_awaitable_t, _T, _Promise&>;
        } else if constexpr (__awaitable<_T>) {
          return true;
        } else if constexpr (__impl::__awaitable_sender<_T, _Promise>) {
          using _Sender = __impl::__sender_awaitable_t<_Promise, _T>;
          return is_nothrow_constructible_v<_Sender, _T, __coro::coroutine_handle<_Promise>>;
        } else {
          return true;
        }
      }
      template <class _T, class _Promise>
      decltype(auto) operator()(_T&& __t, _Promise& __promise) const
          noexcept(__is_noexcept<_T, _Promise>()) {
        if constexpr (__impl::__custom_tag_invoke_awaiter<_T, _Promise>) {
          return tag_invoke(*this, (_T&&) __t, __promise);
        } else if constexpr (__awaitable<_T>) {
          return (_T&&) __t;
        } else if constexpr (__impl::__awaitable_sender<_T, _Promise>) {
          auto __hcoro = __coro::coroutine_handle<_Promise>::from_promise(__promise);
          return __impl::__sender_awaitable_t<_Promise, _T>{(_T&&) __t, __hcoro};
        } else {
          return (_T&&) __t;
        }
      }
    } as_awaitable{};
  } // namespace __as_awaitable

  inline namespace __with_awaitable_senders {
    namespace __impl {
      struct __with_awaitable_senders_base {
        template <class _OtherPromise>
        void set_continuation(__coro::coroutine_handle<_OtherPromise> __hcoro) noexcept {
          static_assert(!is_void_v<_OtherPromise>);
          __continuation_ = __hcoro;
          if constexpr (requires(_OtherPromise& __other) { __other.unhandled_done(); }) {
            __done_callback_ = [](void* __address) noexcept -> __coro::coroutine_handle<> {
              // This causes the rest of the coroutine (the part after the co_await
              // of the sender) to be skipped and invokes the calling coroutine's
              // done handler.
              return __coro::coroutine_handle<_OtherPromise>::from_address(__address)
                  .promise().unhandled_done();
            };
          }
          // If _OtherPromise doesn't implement unhandled_done(), then if a "done" unwind
          // reaches this point, it's considered an unhandled exception and terminate()
          // is called.
        }

        __coro::coroutine_handle<> continuation() const noexcept {
          return __continuation_;
        }

        __coro::coroutine_handle<> unhandled_done() noexcept {
          return (*__done_callback_)(__continuation_.address());
        }

       private:
        __coro::coroutine_handle<> __continuation_{};
        __coro::coroutine_handle<> (*__done_callback_)(void*) noexcept =
          [](void*) noexcept -> __coro::coroutine_handle<> {
            std::terminate();
          };
      };
    } // namespace __impl

    template <class _Promise>
    struct with_awaitable_senders : __impl::__with_awaitable_senders_base {
      template <class _Value>
      decltype(auto) await_transform(_Value&& __val) {
        static_assert(derived_from<_Promise, with_awaitable_senders>);
        return as_awaitable((_Value&&) __val, static_cast<_Promise&>(*this));
      }
    };
  }

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __submit
  inline namespace __submit_ {
    namespace __impl {
      template <class _SenderId, class _ReceiverId>
        struct __operation {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          struct __receiver {
            __operation* __op_state_;
            // Forward all the receiver ops, and delete the operation state.
            template <__one_of<set_value_t, set_error_t, set_done_t> _Tag, class... _As>
              requires __callable<_Tag, _Receiver, _As...>
            friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as)
                noexcept(__nothrow_callable<_Tag, _Receiver, _As...>) {
              // Delete the state as cleanup:
              unique_ptr<__operation> __g{__self.__op_state_};
              return __tag((_Receiver&&) __self.__op_state_->__rcvr_, (_As&&) __as...);
            }
            // Forward all receiever queries.
            template <__none_of<set_value_t, set_error_t, set_done_t> _Tag, class... _As>
              requires __callable<_Tag, const _Receiver&, _As...>
            friend decltype(auto) tag_invoke(_Tag __tag, const __receiver& __self, _As&&... __as)
                noexcept(__nothrow_callable<_Tag, const _Receiver&, _As...>) {
              return ((_Tag&&) __tag)((const _Receiver&) __self.__op_state_->__rcvr_, (_As&&) __as...);
            }
          };
          _Receiver __rcvr_;
          connect_result_t<_Sender, __receiver> __op_state_;
          __operation(_Sender&& __sndr, __decays_to<_Receiver> auto&& __rcvr)
            : __rcvr_((decltype(__rcvr)&&) __rcvr)
            , __op_state_(connect((_Sender&&) __sndr, __receiver{this}))
          {}
        };
    } // namespace __impl

    inline constexpr struct __submit_t {
      template <receiver _Receiver, sender_to<_Receiver> _Sender>
      void operator()(_Sender&& __sndr, _Receiver&& __rcvr) const noexcept(false) {
        start((new __impl::__operation<__x<_Sender>, __x<decay_t<_Receiver>>>{
            (_Sender&&) __sndr, (_Receiver&&) __rcvr})->__op_state_);
      }
    } __submit {};
  } // namespace __submit_

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  inline namespace __start_detached {
    namespace __impl {
      struct __detached_receiver {
        friend void tag_invoke(set_value_t, __detached_receiver&&, auto&&...) noexcept {}
        [[noreturn]]
        friend void tag_invoke(set_error_t, __detached_receiver&&, auto&&) noexcept {
          terminate();
        }
        friend void tag_invoke(set_done_t, __detached_receiver&&) noexcept {}
      };
    } // namespace __impl

    inline constexpr struct start_detached_t {
      template <sender _Sender>
        requires tag_invocable<start_detached_t, _Sender>
      void operator()(_Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<start_detached_t, _Sender>) {
        (void) tag_invoke(start_detached_t{}, (_Sender&&) __sndr);
      }
      template <sender _Sender>
        requires (!tag_invocable<start_detached_t, _Sender>) &&
          sender_to<_Sender, __impl::__detached_receiver>
      void operator()(_Sender&& __sndr) const noexcept(false) {
        __submit((_Sender&&) __sndr, __impl::__detached_receiver{});
      }
    } start_detached {};
  } // namespace __start_detached

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  inline namespace __just {
    namespace __impl {
      template <class _CPO, class... _Ts>
        struct __sender {
          tuple<_Ts...> __vals_;
          template <class _Receiver>
            using __is_nothrow =
              __bool<noexcept(_CPO{}(__declval<_Receiver>(), __declval<_Ts>()...))>;

          template <class _ReceiverId>
          struct __operation {
            using _Receiver = __t<_ReceiverId>;
            tuple<_Ts...> __vals_;
            _Receiver __rcvr_;

            friend void tag_invoke(start_t, __operation& __op_state) noexcept
                requires __v<__is_nothrow<_Receiver>> {
              std::apply([&__op_state](_Ts&... __ts) {
                _CPO{}((_Receiver&&) __op_state.__rcvr_, (_Ts&&) __ts...);
              }, __op_state.__vals_);
            }

            friend void tag_invoke(start_t, __operation& __op_state) noexcept try {
              std::apply([&__op_state](_Ts&... __ts) {
                _CPO{}((_Receiver&&) __op_state.__rcvr_, (_Ts&&) __ts...);
              }, __op_state.__vals_);
            } catch(...) {
              set_error((_Receiver&&) __op_state.__rcvr_, current_exception());
            }
          };

          template <receiver_of<_Ts...> _Receiver>
            requires (copy_constructible<_Ts> &&...)
          friend auto tag_invoke(connect_t, const __sender& __sndr, _Receiver&& __rcvr)
            noexcept((is_nothrow_copy_constructible_v<_Ts> &&...))
            -> __operation<__x<remove_cvref_t<_Receiver>>> {
            return {__sndr.__vals_, (_Receiver&&) __rcvr};
          }

          template <receiver_of<_Ts...> _Receiver>
          friend auto tag_invoke(connect_t, __sender&& __sndr, _Receiver&& __rcvr)
            noexcept((is_nothrow_move_constructible_v<_Ts> &&...))
            -> __operation<__x<remove_cvref_t<_Receiver>>> {
            return {((__sender&&) __sndr).__vals_, (_Receiver&&) __rcvr};
          }
        };

        template <class... _Ts>
        completion_signatures<set_value_t(_Ts...), set_error_t(exception_ptr)>
        tag_invoke(get_sender_traits_t, const __sender<set_value_t, _Ts...>&, auto&&) noexcept;

        template <class _Tag, class... _Ts>
        completion_signatures<_Tag(_Ts...)>
        tag_invoke(get_sender_traits_t, const __sender<_Tag, _Ts...>&, auto&&) noexcept;
    }

    inline constexpr struct __just_t {
      template <__movable_value... _Ts>
      __impl::__sender<set_value_t, decay_t<_Ts>...> operator()(_Ts&&... __ts) const
        noexcept((is_nothrow_constructible_v<decay_t<_Ts>, _Ts> &&...)) {
        return {{(_Ts&&) __ts...}};
      }
    } just {};

    inline constexpr struct __just_error_t {
      template <__movable_value _Error>
      __impl::__sender<set_error_t, _Error> operator()(_Error&& __err) const
        noexcept(is_nothrow_constructible_v<decay_t<_Error>, _Error>) {
        return {{(_Error&&) __err}};
      }
    } just_error {};

    inline constexpr struct __just_done_t {
      __impl::__sender<set_done_t> operator()() const noexcept {
        return {{}};
      }
    } just_done {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  inline namespace __execute_ {
    namespace __impl {
      template <class _Fun>
        struct __as_receiver {
          _Fun __fun_;
          friend void tag_invoke(set_value_t, __as_receiver&& __rcvr) noexcept(__nothrow_callable<_Fun&>) {
            __rcvr.__fun_();
          }
          [[noreturn]]
          friend void tag_invoke(set_error_t, __as_receiver&&, exception_ptr) noexcept {
            terminate();
          }
          friend void tag_invoke(set_done_t, __as_receiver&&) noexcept {}
        };
    }

    inline constexpr struct execute_t {
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> && move_constructible<_Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const
        noexcept(noexcept(
          submit(schedule((_Scheduler&&) __sched), __impl::__as_receiver<_Fun>{(_Fun&&) __fun}))) {
        (void) submit(schedule((_Scheduler&&) __sched), __impl::__as_receiver<_Fun>{(_Fun&&) __fun});
      }
      template <scheduler _Scheduler, class _Fun>
        requires __callable<_Fun&> &&
          move_constructible<_Fun> &&
          tag_invocable<execute_t, _Scheduler, _Fun>
      void operator()(_Scheduler&& __sched, _Fun __fun) const
        noexcept(nothrow_tag_invocable<execute_t, _Scheduler, _Fun>) {
        (void) tag_invoke(execute_t{}, (_Scheduler&&) __sched, (_Fun&&) __fun);
      }
    } execute {};
  }

  // NOT TO SPEC:
  namespace __closure {
    template <__class _D>
      struct sender_adaptor_closure;
  }
  using __closure::sender_adaptor_closure;

  template <class _T>
    concept __sender_adaptor_closure =
      derived_from<remove_cvref_t<_T>, sender_adaptor_closure<remove_cvref_t<_T>>> &&
      move_constructible<remove_cvref_t<_T>> &&
      constructible_from<remove_cvref_t<_T>, _T>;

  namespace __closure {
    template <class _T0, class _T1>
    struct __compose : sender_adaptor_closure<__compose<_T0, _T1>> {
      [[no_unique_address]] _T0 __t0_;
      [[no_unique_address]] _T1 __t1_;

      template <sender _Sender>
        requires __callable<_T0, _Sender> && __callable<_T1, __call_result_t<_T0, _Sender>>
      __call_result_t<_T1, __call_result_t<_T0, _Sender>> operator()(_Sender&& __sndr) && {
        return ((_T1&&) __t1_)(((_T0&&) __t0_)((_Sender&&) __sndr));
      }

      template <sender _Sender>
        requires __callable<const _T0&, _Sender> && __callable<const _T1&, __call_result_t<const _T0&, _Sender>>
      __call_result_t<_T1, __call_result_t<_T0, _Sender>> operator()(_Sender&& __sndr) const & {
        return __t1_(__t0_((_Sender&&) __sndr));
      }
    };

    template <__class _D>
      struct sender_adaptor_closure
      {};

    template <__sender_adaptor_closure _T0, __sender_adaptor_closure _T1>
    __compose<remove_cvref_t<_T0>, remove_cvref_t<_T1>> operator|(_T0&& __t0, _T1&& __t1) {
      return {(_T0&&) __t0, (_T1&&) __t1};
    }

    template <sender _Sender, __sender_adaptor_closure _Closure>
      requires __callable<_Closure, _Sender>
    __call_result_t<_Closure, _Sender> operator|(_Sender&& __sndr, _Closure&& __clsur) {
      return ((_Closure&&) __clsur)((_Sender&&) __sndr);
    }

    template <class _Fun, class... _As>
    struct __binder_back : sender_adaptor_closure<__binder_back<_Fun, _As...>> {
      [[no_unique_address]] _Fun __fun_;
      tuple<_As...> __as_;

      template <sender _Sender>
        requires __callable<_Fun, _Sender, _As...>
      __call_result_t<_Fun, _Sender, _As...> operator()(_Sender&& __sndr) &&
        noexcept(__nothrow_callable<_Fun, _Sender, _As...>) {
        return std::apply([&__sndr, this](_As&... __as) {
            return ((_Fun&&) __fun_)((_Sender&&) __sndr, (_As&&) __as...);
          }, __as_);
      }

      template <sender _Sender>
        requires __callable<const _Fun&, _Sender, const _As&...>
      __call_result_t<const _Fun&, _Sender, const _As&...> operator()(_Sender&& __sndr) const &
        noexcept(__nothrow_callable<const _Fun&, _Sender, const _As&...>) {
        return std::apply([&__sndr, this](const _As&... __as) {
            return __fun_((_Sender&&) __sndr, __as...);
          }, __as_);
      }
    };
  } // namespace __closure
  using __closure::__binder_back;

  namespace __tag_invoke_adaptors {
    // A derived-to-base cast that works even when the base is not
    // accessible from derived.
    template <class _T, class _U>
      __member_t<_U, _T> __c_cast(_U&& u) noexcept requires __decays_to<_T, _T> {
        static_assert(is_reference_v<__member_t<_U, _T>>);
        static_assert(is_base_of_v<_T, remove_reference_t<_U>>);
        return (__member_t<_U, _T>) (_U&&) u;
      }
    namespace __no {
      struct __nope {};
      struct __receiver : __nope {};
      void tag_invoke(set_error_t, __receiver, exception_ptr) noexcept;
      void tag_invoke(set_done_t, __receiver) noexcept;
    }
    using __not_a_receiver = __no::__receiver;

    template <class _Base>
      struct __adaptor {
        struct __t {
          template <class _T1>
            requires constructible_from<_Base, _T1>
          explicit __t(_T1&& __base) : __base_((_T1&&) __base) {}

         private:
          [[no_unique_address]] _Base __base_;

         protected:
          _Base& base() & noexcept { return __base_; }
          const _Base& base() const & noexcept { return __base_; }
          _Base&& base() && noexcept { return (_Base&&) __base_; }
        };
      };
    template <derived_from<__no::__nope> _Base>
      struct __adaptor<_Base> {
        struct __t : __no::__nope { };
      };
    template <class _Base>
      using __adaptor_base = typename __adaptor<_Base>::__t;

    template <class _Sender, class _Receiver>
      concept __has_connect =
        requires(_Sender&& __sndr, _Receiver&& __rcvr) {
          ((_Sender&&) __sndr).connect((_Receiver&&) __rcvr);
        };

    template <__class _Derived, sender _Base>
      struct __sender_adaptor {
        class __t : __adaptor_base<_Base> {
          using connect = void;

          template <__decays_to<_Derived> _Self, receiver _Receiver>
            requires __has_connect<_Self, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            noexcept(noexcept(((_Self&&) __self).connect((_Receiver&&) __rcvr)))
            -> decltype(((_Self&&) __self).connect((_Receiver&&) __rcvr)) {
            return ((_Self&&) __self).connect((_Receiver&&) __rcvr);
          }

          template <__decays_to<_Derived> _Self, receiver _Receiver>
            requires requires {typename decay_t<_Self>::connect;} &&
              sender_to<__member_t<_Self, _Base>, _Receiver>
          friend decltype(auto) tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            noexcept(__has_nothrow_connect<__member_t<_Self, _Base>, _Receiver>) {
            execution::connect(((_Self&&) __self).base(), (_Receiver&&) __rcvr);
          }

          template <__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Base&, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const _Derived& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Base&, _As...>) {
            return ((_Tag&&) __tag)(__self.base(), (_As&&) __as...);
          }

         protected:
          using __adaptor_base<_Base>::base;

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };

    template <class _Receiver, class... _As>
      concept __has_set_value =
        requires(_Receiver&& __rcvr, _As&&... __as) {
          ((_Receiver&&) __rcvr).set_value((_As&&) __as...);
        };

    template <class _Receiver, class _Error>
      concept __has_set_error =
        requires(_Receiver&& __rcvr, _Error&& __err) {
          ((_Receiver&&) __rcvr).set_error((_Error&&) __err);
        };

    template <class _Receiver>
      concept __has_set_done =
        requires(_Receiver&& __rcvr) {
          ((_Receiver&&) __rcvr).set_done();
        };

    template <__class _Derived, receiver _Base>
      struct __receiver_adaptor {
        class __t : __adaptor_base<_Base> {
          friend _Derived;
          using set_value = void;
          using set_error = void;
          using set_done = void;

          static constexpr bool __has_base = !derived_from<_Base, __no::__nope>;

          template <class _D>
            using __base_from_derived_t = decltype(__declval<_D>().base());

          using __get_base_t =
            __if<
              __bool<__has_base>,
              __bind_back<__defer<__member_t>, _Base>,
              __q1<__base_from_derived_t>>;

          template <class _D>
            using __base_t = __minvoke1<__get_base_t, _D&&>;

          template <class _D>
            static __base_t<_D> __get_base(_D&& __self) noexcept {
              if constexpr (__has_base) {
                return __c_cast<__t>((_D&&) __self).base();
              } else {
                return ((_D&&) __self).base();
              }
            }

          template <class... _As>
            requires __has_set_value<_Derived, _As...>
          friend void tag_invoke(set_value_t, _Derived&& __self, _As&&... __as)
            noexcept(noexcept(((_Derived&&) __self).set_value((_As&&) __as...))) {
            ((_Derived&&) __self).set_value((_As&&) __as...);
          }

          template <class _D = _Derived, class... _As>
            requires requires {typename _D::set_value;} &&
              receiver_of<__base_t<_D>, _As...>
          friend void tag_invoke(set_value_t, _Derived&& __self, _As&&... __as)
            noexcept(nothrow_receiver_of<__base_t<_D>, _As...>) {
            execution::set_value(__get_base((_Derived&&) __self), (_As&&) __as...);
          }

          template <class _Error>
            requires __has_set_error<_Derived, _Error>
          friend void tag_invoke(set_error_t, _Derived&& __self, _Error&& __err) noexcept {
            static_assert(noexcept(((_Derived&&) __self).set_error((_Error&&) __err)));
            ((_Derived&&) __self).set_error((_Error&&) __err);
          }

          template <class _Error, class _D = _Derived>
            requires requires {typename _D::set_error;} &&
              receiver<__base_t<_D>, _Error>
          friend void tag_invoke(set_error_t, _Derived&& __self, _Error&& __err) noexcept {
            execution::set_error(__get_base((_Derived&&) __self), (_Error&&) __err);
          }

          template <class _D = _Derived>
            requires __has_set_done<_D>
          friend void tag_invoke(set_done_t, _Derived&& __self) noexcept {
            static_assert(noexcept(((_Derived&&) __self).set_done()));
            ((_Derived&&) __self).set_done();
          }

          template <class _D = _Derived>
            requires requires {typename _D::set_done;}
          friend void tag_invoke(set_done_t, _Derived&& __self) noexcept {
            execution::set_done(__get_base((_Derived&&) __self));
          }

          template <__receiver_query _Tag, class _D = _Derived, class... _As>
            requires __callable<_Tag, __base_t<const _D&>, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const _Derived& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, __base_t<const _D&>, _As...>) {
            return ((_Tag&&) __tag)(__get_base(__self), (_As&&) __as...);
          }

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };

    template <class _OpState>
      concept __has_start =
        requires(_OpState& __op_state) {
          __op_state.start();
        };

    template <__class _Derived, operation_state _Base>
      struct __operation_state_adaptor {
        class __t : __adaptor_base<_Base> {
          using start = void;

          template <class _D = _Derived>
            requires __has_start<_D>
          friend void tag_invoke(start_t, _Derived& __self) noexcept {
            static_assert(noexcept(__self.start()));
            __self.start();
          }

          template <class _D = _Derived>
            requires requires {typename _D::start;}
          friend void tag_invoke(start_t, _Derived& __self) noexcept {
            execution::start(__c_cast<__t>(__self).base());
          }

          template <__none_of<start_t> _Tag, class... _As>
            requires __callable<_Tag, const _Base&, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const _Derived& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Base&, _As...>) {
            return ((_Tag&&) __tag)(__c_cast<__t>(__self).base(), (_As&&) __as...);
          }

         protected:
          using __adaptor_base<_Base>::base;

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };

    template <class _Scheduler>
      concept __has_schedule =
        requires(_Scheduler&& __sched) {
          ((_Scheduler&&) __sched).schedule();
        };

    template <__class _Derived, scheduler _Base>
      struct __scheduler_adaptor {
        class __t : __adaptor_base<_Base> {
          using schedule = void;

          template <__decays_to<_Derived> _Self>
            requires __has_schedule<_Self>
          friend auto tag_invoke(schedule_t, _Self&& __self)
            noexcept(noexcept(((_Self&&) __self).schedule()))
            -> decltype(((_Self&&) __self).schedule()) {
            return ((_Self&&) __self).schedule();
          }

          template <__decays_to<_Derived> _Self>
            requires requires {typename decay_t<_Self>::schedule;} &&
              scheduler<__member_t<_Self, _Base>>
          friend auto tag_invoke(schedule_t, _Self&& __self)
            noexcept(noexcept(execution::schedule(__declval<__member_t<_Self, _Base>>())))
            -> schedule_result_t<_Self> {
            return execution::schedule(__c_cast<__t>((_Self&&) __self).base());
          }

          template <__none_of<schedule_t> _Tag, same_as<_Derived> _Self, class... _As>
            requires __callable<_Tag, const _Base&, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const _Self& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Base&, _As...>) {
            return ((_Tag&&) __tag)(__c_cast<__t>(__self).base(), (_As&&) __as...);
          }

         protected:
          using __adaptor_base<_Base>::base;

         public:
          __t() = default;
          using __adaptor_base<_Base>::__adaptor_base;
        };
      };
  } // namespace __tag_invoke_adaptors

  // NOT TO SPEC
  template <__class _Derived, sender _Base>
    using sender_adaptor =
      typename __tag_invoke_adaptors::__sender_adaptor<_Derived, _Base>::__t;

  template <__class _Derived, receiver _Base = __tag_invoke_adaptors::__not_a_receiver>
    using receiver_adaptor =
      typename __tag_invoke_adaptors::__receiver_adaptor<_Derived, _Base>::__t;

  // NOT TO SPEC
  template <__class _Derived, operation_state _Base>
    using operation_state_adaptor =
      typename __tag_invoke_adaptors::__operation_state_adaptor<_Derived, _Base>::__t;

  // NOT TO SPEC
  template <__class _Derived, scheduler _Base>
    using scheduler_adaptor =
      typename __tag_invoke_adaptors::__scheduler_adaptor<_Derived, _Base>::__t;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  inline namespace __then {
    namespace __impl {
      template <class _ReceiverId, class _Fun>
        class __receiver
          : receiver_adaptor<__receiver<_ReceiverId, _Fun>, __t<_ReceiverId>> {
          using _Receiver = __t<_ReceiverId>;
          friend receiver_adaptor<__receiver, _Receiver>;
          [[no_unique_address]] _Fun __f_;

          // Customize set_value by invoking the invocable and passing the result
          // to the base class
          template <class... _As>
            requires invocable<_Fun, _As...> &&
              receiver_of<_Receiver, invoke_result_t<_Fun, _As...>>
          void set_value(_As&&... __as) && noexcept try {
            execution::set_value(
                ((__receiver&&) *this).base(),
                std::invoke((_Fun&&) __f_, (_As&&) __as...));
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }
          // Handle the case when the invocable returns void
          template <class _R2 = _Receiver, class... _As>
            requires invocable<_Fun, _As...> &&
              same_as<void, invoke_result_t<_Fun, _As...>> &&
              receiver_of<_R2>
          void set_value(_As&&... __as) && noexcept try {
            invoke((_Fun&&) __f_, (_As&&) __as...);
            execution::set_value(((__receiver&&) *this).base());
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }

         public:
          explicit __receiver(_Receiver __rcvr, _Fun __fun)
            : receiver_adaptor<__receiver, _Receiver>((_Receiver&&) __rcvr)
            , __f_((_Fun&&) __fun)
          {}
        };

      template <class, class, class>
        struct __traits {};

      template <class _Sender, class _Receiver, class _Fun>
          requires typed_sender<_Sender, __receiver<__x<_Receiver>, _Fun>> &&
            __valid<__tfx_sender_values, _Sender, __receiver<__x<_Receiver>, _Fun>, _Fun>
        struct __traits<_Sender, _Receiver, _Fun> {
          using __receiver_t = __receiver<__x<_Receiver>, _Fun>;

          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types =
              __tfx_sender_values<
                _Sender,
                __receiver_t,
                _Fun,
                __q1<__id>,
                __transform<
                  __q<__types>,
                  __replace<
                    __types<void>,
                    __types<>,
                    __transform<__uncurry<__q<_Tuple>>, __q<_Variant>>>>>;

          template <template <class...> class _Variant>
            using error_types =
              __minvoke2<
                __push_back_unique<__q<_Variant>>,
                error_types_of_t<_Sender, __receiver_t, __types>,
                exception_ptr>;

          static constexpr bool sends_done = sender_traits<_Sender, __receiver_t>::sends_done;
        };

      template <class _SenderId, class _Fun>
        class __sender : sender_adaptor<__sender<_SenderId, _Fun>, __t<_SenderId>> {
          using _Sender = __t<_SenderId>;
          friend sender_adaptor<__sender, _Sender>;
          template <class _Receiver>
            using __receiver = __receiver<__x<remove_cvref_t<_Receiver>>, _Fun>;

          [[no_unique_address]] _Fun __fun_;

          template <receiver _Receiver>
            requires sender_to<_Sender, __receiver<_Receiver>>
          decltype(auto) connect(_Receiver&& __rcvr) &&
            noexcept(__has_nothrow_connect<_Sender, __receiver<_Receiver>>) {
            return execution::connect(
                ((__sender&&) *this).base(),
                __receiver<_Receiver>{(_Receiver&&) __rcvr, (_Fun&&) __fun_});
          }

          template <class _Receiver>
          friend constexpr __traits<_Sender, _Receiver, _Fun>
          tag_invoke(get_sender_traits_t, const __sender&, _Receiver&&) noexcept {
            return {};
          }

         public:
          explicit __sender(_Sender __sndr, _Fun __fun)
            : sender_adaptor<__sender, _Sender>{(_Sender&&) __sndr}
            , __fun_((_Fun&&) __fun)
          {}
        };
    }

    inline constexpr struct then_t {
      template <class _Sender, class _Fun>
        using __sender = __impl::__sender<__x<remove_cvref_t<_Sender>>, _Fun>;

      template <sender _Sender, __invocable_with_values_from<_Sender> _Fun>
        requires __tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Fun>) {
        auto __sched = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(then_t{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <sender _Sender, __invocable_with_values_from<_Sender> _Fun>
        requires (!__tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>) &&
          tag_invocable<then_t, _Sender, _Fun>
      sender auto operator()(_Sender&& __sndr, _Fun __fun) const
        noexcept(nothrow_tag_invocable<then_t, _Sender, _Fun>) {
        return tag_invoke(then_t{}, (_Sender&&) __sndr, (_Fun&&) __fun);
      }
      template <sender _Sender, __invocable_with_values_from<_Sender> _Fun>
        requires (!__tag_invocable_with_completion_scheduler<then_t, set_value_t, _Sender, _Fun>) &&
          (!tag_invocable<then_t, _Sender, _Fun>) &&
          sender<__sender<_Sender, _Fun>>
      __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
        return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
      }
      template <class _Fun>
      __binder_back<then_t, _Fun> operator()(_Fun __fun) const {
        return {{}, {}, {(_Fun&&) __fun}};
      }
    } then {};
  }

  //////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.let_value]
  // [execution.senders.adaptors.let_error]
  // [execution.senders.adaptors.let_done]
  inline namespace __let {
    namespace __impl {
      using __nullable_variant_t = __munique<__bind_front<__q<variant>, __>>;

      template <class... _Ts>
        struct __as_tuple {
          __decayed_tuple<_Ts...> operator()(_Ts...) const;
        };

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __receiver;

      template <class... _Ts>
        struct __which_tuple_ : _Ts... {
          using _Ts::operator()...;
        };

      struct __which_tuple_base {
        template <class... _Ts>
          __decayed_tuple<_Ts...> operator()(_Ts&&...) const;
      };

      template <sender, class, class>
        struct __which_tuple : __which_tuple_base {};

      template <class _Sender, class _Receiver>
          requires typed_sender<_Sender, _Receiver>
        struct __which_tuple<_Sender, _Receiver, set_value_t>
          : value_types_of_t<_Sender, _Receiver, __as_tuple, __which_tuple_> {};

      template <class _Sender, class _Receiver>
          requires typed_sender<_Sender, _Receiver>
        struct __which_tuple<_Sender, _Receiver, set_error_t>
          : __error_types_of_t<
              _Sender,
              _Receiver,
              __transform<__q<__as_tuple>, __q<__which_tuple_>>> {};

      template <class _Fun>
        struct __applyable_fn {
          __ operator()(auto&&...) const;
          template <class... _As>
              requires invocable<_Fun, _As...>
            invoke_result_t<_Fun, _As...> operator()(_As&&...) const;
        };

      template <class _Fun, class _Tuple>
        concept __applyable =
          requires (__applyable_fn<_Fun> __fun, _Tuple&& __tupl) {
            {std::apply(__fun, (_Tuple&&) __tupl)} -> __none_of<__>;
          };
      template <class _Fun, class _Tuple>
          requires __applyable<_Fun, _Tuple>
        using __apply_result_t =
          decltype(std::apply(__applyable_fn<_Fun>{}, __declval<_Tuple>()));

      template <class _T>
        using __decay_ref = decay_t<_T>&;

      template <class _Fun, class... _As>
        using __result_sender_t = __call_result_t<_Fun, __decay_ref<_As>...>;

      template <class _Sender, class _Receiver, class _Fun, class _Let,
          class = __receiver<__x<_Sender>, __x<_Receiver>, _Fun, _Let>>
        struct __storage {
          any __args_;
          any __op_state3_;
        };

      // Storage for let_value:
      template <class _Sender, class _Receiver, class _Fun, class _LetReceiver>
          requires typed_sender<_Sender, _LetReceiver>
        struct __storage<_Sender, _Receiver, _Fun, set_value_t, _LetReceiver> {
          template <class... _As>
            using __op_state_for_t =
              connect_result_t<__result_sender_t<_Fun, _As...>, _Receiver>;

          // Compute a variant of tuples to hold all the values of the input
          // sender:
          using __args_t =
            __value_types_of_t<_Sender, _LetReceiver, __q<__decayed_tuple>, __nullable_variant_t>;
          __args_t __args_;

          // Compute a variant of operation states:
          using __op_state3_t =
            __value_types_of_t<_Sender, _LetReceiver, __q<__op_state_for_t>, __nullable_variant_t>;
          __op_state3_t __op_state3_;
        };

      // Storage for let_error:
      template <class _Sender, class _Receiver, class _Fun, class _LetReceiver>
          requires typed_sender<_Sender, _LetReceiver>
        struct __storage<_Sender, _Receiver, _Fun, set_error_t, _LetReceiver> {
          template <class _Error>
            using __op_state_for_t =
              connect_result_t<__result_sender_t<_Fun, _Error>, _Receiver>;

          // Compute a variant of tuples to hold all the errors of the input
          // sender:
          using __args_t =
            __error_types_of_t<
              _Sender,
              _LetReceiver,
              __transform<__q<__decayed_tuple>, __nullable_variant_t>>;
          __args_t __args_;

          // Compute a variant of operation states:
          using __op_state3_t =
            __error_types_of_t<
              _Sender,
              _LetReceiver,
              __transform<__q1<__op_state_for_t>, __nullable_variant_t>>;
          __op_state3_t __op_state3_;
        };

      // Storage for let_done
      template <class _Sender, class _Receiver, class _Fun, class _LetReceiver>
        struct __storage<_Sender, _Receiver, _Fun, set_done_t, _LetReceiver> {
          variant<tuple<>> __args_;
          variant<monostate, connect_result_t<__call_result_t<_Fun>, _Receiver>> __op_state3_;
        };

      template <class _Receiver>
        struct __typed_senders {
          template <typed_sender<_Receiver>...>
            struct __f;
        };

      template <class _T0, class _T1>
        using __or = __bool<(__v<_T0> || __v<_T1>)>;

      // Call the _Continuation with the result of calling _Fun with
      // every set of values:
      template <class _Sender, class _Receiver, class _Fun, class _Continuation>
        using __value_senders_of =
          __tfx_sender_values<_Sender, _Receiver, _Fun, __q1<__decay_ref>, _Continuation>;

      // Call the _Continuation with the result of calling _Fun with
      // every error:
      template <class _Sender, class _Receiver, class _Fun, class _Continuation>
        using __error_senders_of =
          __tfx_sender_errors<_Sender, _Receiver, _Fun, __q1<__decay_ref>, _Continuation>;

      // Call the _Continuation with the result of calling _Fun:
      template <class _Sender, class _Receiver, class _Fun, class _Continuation>
        using __done_senders_of =
          __tfx_sender_done<_Sender, _Receiver, _Fun, __q1<__decay_ref>, _Continuation>;

      // A let_xxx sender is typed if and only if the input sender is a typed
      // sender, and if all the possible return types of the function are also
      // typed senders.
      template <class _Sender, class _Receiver, class _Fun, class _Let,
          class = __receiver<__x<_Sender>, __x<_Receiver>, _Fun, _Let>>
        struct __traits {};

      template <class _Sender, class _Receiver, class _Fun, class _LetReceiver>
          requires typed_sender<_Sender, _LetReceiver> &&
            __valid<__value_senders_of, _Sender, _LetReceiver, _Fun, __typed_senders<_Receiver>>
        struct __traits<_Sender, _Receiver, _Fun, set_value_t, _LetReceiver>
        {
          template <class _Continuation>
            using __result_senders_t =
              __value_senders_of<_Sender, _LetReceiver, _Fun, _Continuation>;

          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types =
              __result_senders_t<
                __transform<
                  __bind_back<__defer<__value_types_of_t>, _Receiver, __q<_Tuple>, __q<__types>>,
                  __concat<__munique<__q<_Variant>>>>>;

          template <template <class...> class _Variant>
            using error_types =
              __result_senders_t<
                __transform<
                  __bind_back<__defer<__error_types_of_t>, _Receiver, __q<__types>>,
                  __bind_front<
                    __concat<__munique<__q<_Variant>>>,
                    __types<exception_ptr>,
                    error_types_of_t<_Sender, _LetReceiver, __types>>>>;

          static constexpr bool sends_done =
            __result_senders_t<
              __transform<
                __bind_back_q1<__sends_done, _Receiver>,
                __right_fold<
                  __sends_done<_Sender, _LetReceiver>,
                  __q2<__or>>>>::value;
        };

      template <class _Sender, class _Receiver, class _Fun, class _LetReceiver>
          requires typed_sender<_Sender, _LetReceiver> &&
            __valid<__error_senders_of, _Sender, _LetReceiver, _Fun, __typed_senders<_Receiver>>
        struct __traits<_Sender, _Receiver, _Fun, set_error_t, _LetReceiver>
        {
          template <class _Continuation>
            using __result_senders_t =
              __error_senders_of<_Sender, _LetReceiver, _Fun, _Continuation>;

          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types =
              __result_senders_t<
                __transform<
                  __bind_back<__defer<__value_types_of_t>, _Receiver, __q<_Tuple>, __q<__types>>,
                  __bind_front<
                    __concat<__munique<__q<_Variant>>>,
                    value_types_of_t<_Sender, _LetReceiver, _Tuple, __types>>>>;

          template <template <class...> class _Variant>
            using error_types =
              __result_senders_t<
                __transform<
                  __bind_back<__defer<__error_types_of_t>, _Receiver, __q<__types>>,
                  __bind_front<
                    __concat<__munique<__q<_Variant>>>,
                    __types<exception_ptr>>>>;

          static constexpr bool sends_done =
            __result_senders_t<
              __transform<
                __bind_back_q1<__sends_done, _Receiver>,
                __right_fold<
                  __sends_done<_Sender, _LetReceiver>,
                  __q2<__or>>>>::value;
        };

      template <class _Sender, class _Receiver, class _Fun, class _LetReceiver>
          requires typed_sender<_Sender, _LetReceiver> &&
            __valid<__done_senders_of, _Sender, _LetReceiver, _Fun, __typed_senders<_Receiver>>
        struct __traits<_Sender, _Receiver, _Fun, set_done_t, _LetReceiver>
        {
          template <class _Continuation>
            using __result_senders_t =
              __done_senders_of<_Sender, _LetReceiver, _Fun, _Continuation>;

          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types =
              __result_senders_t<
                __transform<
                  __bind_back<__defer<__value_types_of_t>, _Receiver, __q<_Tuple>, __q<__types>>,
                  __bind_front<
                    __concat<__munique<__q<_Variant>>>,
                    value_types_of_t<_Sender, _LetReceiver, _Tuple, __types>>>>;

          template <template <class...> class _Variant>
            using error_types =
              __result_senders_t<
                __transform<
                  __bind_back<__defer<__error_types_of_t>, _Receiver, __q<__types>>,
                  __bind_front<
                    __concat<__munique<__q<_Variant>>>,
                    __types<exception_ptr>,
                    error_types_of_t<_Sender, _LetReceiver, __types>>>>;

          static constexpr bool sends_done =
            __result_senders_t<
              __transform<
                __bind_back_q1<__sends_done, _Receiver>,
                __right_fold<false_type, __q2<__or>>>>::value;
        };

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __operation;

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __receiver {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          _Receiver&& base() && noexcept { return (_Receiver&&) __op_state_->__rcvr_;}
          const _Receiver& base() const & noexcept { return __op_state_->__rcvr_;}

          template <class... _As>
            using __which_tuple_t =
              __call_result_t<__which_tuple<_Sender, __receiver, _Let>, _As...>;

          template <class... _As>
            using __op_state_for_t =
              connect_result_t<__result_sender_t<_Fun, _As...>, _Receiver>;

          // For the purposes of the receiver concept, this receiver must be able
          // to accept exception_ptr, even if the input sender can never complete
          // with set_error.
          template <__decays_to<exception_ptr> _Error>
            friend void tag_invoke(set_error_t, __receiver&&, _Error&& __err) noexcept
              requires same_as<_Let, set_error_t> && (!__valid<__which_tuple_t, _Error>);

          template <__one_of<_Let> _Tag, class... _As>
              requires __applyable<_Fun, __which_tuple_t<_As...>&> &&
                sender_to<__apply_result_t<_Fun, __which_tuple_t<_As...>&>, _Receiver>
            friend void tag_invoke(_Tag, __receiver&& __self, _As&&... __as) noexcept try {
              using __tuple_t = __which_tuple_t<_As...>;
              using __op_state_t = __mapply<__q<__op_state_for_t>, __tuple_t>;
              auto& __args =
                __self.__op_state_->__storage_.__args_.template emplace<__tuple_t>((_As&&) __as...);
              start(__self.__op_state_->__storage_.__op_state3_.template emplace<__op_state_t>(
                __conv{[&] {
                  return connect(std::apply(std::move(__self.__op_state_->__fun_), __args), std::move(__self).base());
                }}
              ));
            } catch(...) {
              set_error(std::move(__self).base(), current_exception());
            }

          template <__one_of<set_value_t, set_error_t, set_done_t> _Tag, class... _As>
              requires __none_of<_Tag, _Let> && __callable<_Tag, _Receiver, _As...>
            friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept try {
              __tag(std::move(__self).base(), (_As&&) __as...);
            } catch(...) {
              set_error(std::move(__self).base(), current_exception());
            }

          template <__receiver_query _Tag, class... _As>
              requires __callable<_Tag, const _Receiver&, _As...>
            friend decltype(auto) tag_invoke(_Tag __tag, const __receiver& __self, _As&&... __as)
                noexcept(__nothrow_callable<_Tag, const _Receiver&, _As...>) {
              return ((_Tag&&) __tag)(__self.base(), (_As&&) __as...);
            }

          __operation<_SenderId, _ReceiverId, _Fun, _Let>* __op_state_;
        };

      template <class _SenderId, class _ReceiverId, class _Fun, class _Let>
        struct __operation {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_t = __receiver<_SenderId, _ReceiverId, _Fun, _Let>;

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            start(__self.__op_state2_);
          }

          template <class _Receiver2>
            __operation(_Sender&& __sndr, _Receiver2&& __rcvr, _Fun __fun)
              : __op_state2_(connect((_Sender&&) __sndr, __receiver_t{this}))
              , __rcvr_((_Receiver2&&) __rcvr)
              , __fun_((_Fun&&) __fun)
            {}

          connect_result_t<_Sender, __receiver_t> __op_state2_;
          _Receiver __rcvr_;
          _Fun __fun_;
          [[no_unique_address]] __storage<_Sender, _Receiver, _Fun, _Let> __storage_;
        };

      template <class _SenderId, class _Fun, class _LetId>
        struct __sender {
          using _Sender = __t<_SenderId>;
          using _Let = __t<_LetId>;
          template <class _Self, class _Receiver>
            using __operation_t =
              __operation<
                __x<__member_t<_Self, _Sender>>,
                __x<remove_cvref_t<_Receiver>>,
                _Fun,
                _Let>;
          template <class _Self, class _Receiver>
            using __receiver_t =
              __receiver<
                __x<__member_t<_Self, _Sender>>,
                __x<remove_cvref_t<_Receiver>>,
                _Fun,
                _Let>;

          template <__decays_to<__sender> _Self, receiver _Receiver>
              requires sender_to<__member_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation_t<_Self, _Receiver> {
              return __operation_t<_Self, _Receiver>{
                  ((_Self&&) __self).__sndr_,
                  (_Receiver&&) __rcvr,
                  ((_Self&&) __self).__fun_
              };
            }

          template <__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>) {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

          template <__decays_to<__sender> _Self, class _Receiver>
          friend constexpr auto tag_invoke(get_sender_traits_t, _Self&&, _Receiver&&) noexcept
              -> __traits<__member_t<_Self, _Sender>, _Receiver, _Fun, _Let> {
            return {};
          }

          _Sender __sndr_;
          _Fun __fun_;
        };

      template <class _LetTag, class _SetTag, class _Which>
        struct __let_xxx_t {
          using type = _SetTag;
          template <class _Sender, class _Fun>
            using __sender = __impl::__sender<__x<remove_cvref_t<_Sender>>, _Fun, _LetTag>;

          template <sender _Sender, __invocable_with_xxx_from<_Sender, _Which, __q1<__impl::__decay_ref>> _Fun>
            requires __tag_invocable_with_completion_scheduler<_LetTag, _SetTag, _Sender, _Fun>
          sender auto operator()(_Sender&& __sndr, _Fun __fun) const
            noexcept(nothrow_tag_invocable<_LetTag, __completion_scheduler_for<_Sender, _SetTag>, _Sender, _Fun>) {
            auto __sched = get_completion_scheduler<_SetTag>(__sndr);
            return tag_invoke(_LetTag{}, std::move(__sched), (_Sender&&) __sndr, (_Fun&&) __fun);
          }
          template <sender _Sender, __invocable_with_xxx_from<_Sender, _Which, __q1<__impl::__decay_ref>> _Fun>
            requires (!__tag_invocable_with_completion_scheduler<_LetTag, _SetTag, _Sender, _Fun>) &&
              tag_invocable<_LetTag, _Sender, _Fun>
          sender auto operator()(_Sender&& __sndr, _Fun __fun) const
            noexcept(nothrow_tag_invocable<_LetTag, _Sender, _Fun>) {
            return tag_invoke(_LetTag{}, (_Sender&&) __sndr, (_Fun&&) __fun);
          }
          template <sender _Sender, __invocable_with_xxx_from<_Sender, _Which, __q1<__impl::__decay_ref>> _Fun>
            requires (!__tag_invocable_with_completion_scheduler<_LetTag, _SetTag, _Sender, _Fun>) &&
              (!tag_invocable<_LetTag, _Sender, _Fun>) &&
              sender<__sender<_Sender, _Fun>>
          __sender<_Sender, _Fun> operator()(_Sender&& __sndr, _Fun __fun) const {
            return __sender<_Sender, _Fun>{(_Sender&&) __sndr, (_Fun&&) __fun};
          }
          template <class _Fun>
          __binder_back<_LetTag, _Fun> operator()(_Fun __fun) const {
            return {{}, {}, {(_Fun&&) __fun}};
          }
        };
    } // namespace __impl

    inline constexpr struct let_value_t
      : __let::__impl::__let_xxx_t<let_value_t, set_value_t, __defer<__tfx_sender_values>>
    {} let_value {};

    inline constexpr struct let_error_t
      : __let::__impl::__let_xxx_t<let_error_t, set_error_t, __defer<__tfx_sender_errors>>
    {} let_error {};

    inline constexpr struct let_done_t
      : __let::__impl::__let_xxx_t<let_done_t, set_done_t, __defer<__tfx_sender_done>>
    {} let_done {};
  } // namespace __let

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.done_as_optional]
  // [execution.senders.adaptors.done_as_error]
  inline namespace __done_as_xxx {
    namespace __impl {
      template <class _Ty, class _Sender, class _Receiver>
        concept __constructible_from =
          __single_typed_sender<_Sender> &&
          constructible_from<optional<__single_sender_value_t<_Sender>>, _Ty>;

      template <class _SenderId, class _ReceiverId>
        struct __operation;

      template <class _SenderId, class _ReceiverId>
        struct __receiver : receiver_adaptor<__receiver<_SenderId, _ReceiverId>> {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          _Receiver&& base() && noexcept { return (_Receiver&&) __op_->__rcvr_; }
          const _Receiver& base() const & noexcept { return __op_->__rcvr_; }

          template <__constructible_from<_Sender, __receiver> _Ty>
            void set_value(_Ty&& __a) && noexcept try {
              using _Value = __single_sender_value_t<_Sender>;
              execution::set_value(((__receiver&&) *this).base(), optional<_Value>{(_Ty&&) __a});
            } catch(...) {
              execution::set_error(((__receiver&&) *this).base(), current_exception());
            }

          void set_done() && noexcept {
            using _Value = __single_sender_value_t<_Sender>;
            execution::set_value(((__receiver&&) *this).base(), optional<_Value>{nullopt});
          }

          __operation<_SenderId, _ReceiverId>* __op_;
        };

      template <class _SenderId, class _ReceiverId>
        struct __operation {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_t = __receiver<_SenderId, _ReceiverId>;

          __operation(_Sender&& __sndr, _Receiver&& __rcvr)
            : __op_state_(connect((_Sender&&) __sndr, __receiver_t{{}, this}))
            , __rcvr_((_Receiver&&) __rcvr)
          {}

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            start(__self.__op_state_);
          }

          connect_result_t<_Sender, __receiver_t> __op_state_;
          _Receiver __rcvr_;
        };

      template <sender, class>
        struct __traits {};

      template <class _Sender, class _Receiver>
          requires __single_typed_sender<_Sender, _Receiver>
        struct __traits<_Sender, _Receiver> {
          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types = _Variant<_Tuple<optional<__single_sender_value_t<_Sender, _Receiver>>>>;

          template <template <class...> class _Variant>
            using error_types =
              __minvoke<
                __push_back_unique<__q<_Variant>>,
                __error_types_of_t<_Sender, _Receiver>,
                exception_ptr>;

          static constexpr bool sends_done = false;
        };

      template <class _SenderId>
        struct __sender {
          using _Sender = __t<_SenderId>;
          template <class _Self, class _Receiver>
            using __operation_t =
              __operation<__x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>>;
          template <class _Self, class _Receiver>
            using __receiver_t =
              __receiver<__x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>>;
          template <class _Self, class _Receiver>
            using __traits_t =
              __traits<__member_t<_Self, _Sender>, decay_t<_Receiver>>;

          template <__decays_to<__sender> _Self, receiver _Receiver>
              requires __single_typed_sender<__member_t<_Self, _Sender>> &&
                sender_to<__member_t<_Self, _Sender>, __receiver_t<_Self, _Receiver>>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation_t<_Self, _Receiver> {
              return __operation_t<_Self, _Receiver>{((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
            }

          template <__sender_query _Tag, class... _As>
              requires __callable<_Tag, const _Sender&, _As...>
            friend decltype(auto) tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
              noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>) {
              return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
            }

          template <__decays_to<__sender> _Self, class _Receiver>
            friend auto tag_invoke(get_sender_traits_t, _Self&&, _Receiver&&) noexcept
              -> __traits_t<_Sender, _Receiver>;

          _Sender __sndr_;
        };
    } // namespace __impl

    inline constexpr struct __done_as_optional_t {
      template <sender _Sender>
        auto operator()(_Sender&& __sndr) const
          -> __impl::__sender<__x<decay_t<_Sender>>> {
          return {(_Sender&&) __sndr};
        }
      __binder_back<__done_as_optional_t> operator()() const noexcept {
        return {};
      }
    } done_as_optional {};

    inline constexpr struct __done_as_error_t {
      template <sender _Sender, __movable_value _Error>
        auto operator()(_Sender&& __sndr, _Error __err) const {
          return (_Sender&&) __sndr
            | let_done([__err2 = (_Error&&) __err] () mutable {
                return just_error((_Error&&) __err2);
              });
        }
      template <__movable_value _Error>
        auto operator()(_Error __err) const
          -> __binder_back<__done_as_error_t, _Error> {
          return {{}, {}, {(_Error&&) __err}};
        }
    } done_as_error {};
  } // namespace __done_as_xxx

  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  inline namespace __loop {
    class run_loop;

    namespace __impl {
      struct __task {
        virtual void __execute_() noexcept = 0;
        __task* __next_ = nullptr;
      };

      template <typename _ReceiverId>
        class __operation final : __task {
          using _Receiver = __t<_ReceiverId>;

          friend void tag_invoke(start_t, __operation& __op_state) noexcept {
            __op_state.__start_();
          }

          void __execute_() noexcept override try {
            if (get_stop_token(__rcvr_).stop_requested()) {
              set_done((_Receiver&&) __rcvr_);
            } else {
              set_value((_Receiver&&) __rcvr_);
            }
          } catch(...) {
            set_error((_Receiver&&) __rcvr_, current_exception());
          }

          void __start_() noexcept;

          [[no_unique_address]] _Receiver __rcvr_;
          run_loop* const __loop_;

         public:
          template <typename _Receiver2>
          explicit __operation(_Receiver2&& __rcvr, run_loop* __loop)
            : __rcvr_((_Receiver2 &&) __rcvr)
            , __loop_(__loop) {}
        };
    } // namespace __impl

    class run_loop {
      template <class>
        friend class __impl::__operation;
     public:
      class __scheduler {
        struct __schedule_task
          : completion_signatures<set_value_t(), set_error_t(exception_ptr), set_done_t()> {
         private:
          friend __scheduler;

          template <typename _Receiver>
          friend __impl::__operation<__x<decay_t<_Receiver>>>
          tag_invoke(connect_t, const __schedule_task& __self, _Receiver&& __rcvr) {
            return __impl::__operation<__x<decay_t<_Receiver>>>{(_Receiver &&) __rcvr, __self.__loop_};
          }

          template <class _CPO>
          friend __scheduler
          tag_invoke(get_completion_scheduler_t<_CPO>, const __schedule_task& __self) noexcept {
            return __scheduler{__self.__loop_};
          }

          explicit __schedule_task(run_loop* __loop) noexcept
            : __loop_(__loop)
          {}

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* __loop) noexcept : __loop_(__loop) {}

       public:
        friend __schedule_task tag_invoke(schedule_t, const __scheduler& __self) noexcept {
          return __self.__schedule();
        }

        bool operator==(const __scheduler&) const noexcept = default;

       private:
        __schedule_task __schedule() const noexcept {
          return __schedule_task{__loop_};
        }

        run_loop* __loop_;
      };

      __scheduler get_scheduler() {
        return __scheduler{this};
      }

      void run();

      void finish();

     private:
      void __push_back_(__impl::__task* __task);
      __impl::__task* __pop_front_();

      mutex __mutex_;
      condition_variable __cv_;
      __impl::__task* __head_ = nullptr;
      __impl::__task* __tail_ = nullptr;
      bool __stop_ = false;
    };

    namespace __impl {
      template <typename _ReceiverId>
      inline void __operation<_ReceiverId>::__start_() noexcept try {
        __loop_->__push_back_(this);
      } catch(...) {
        set_error((_Receiver&&) __rcvr_, current_exception());
      }
    }

    inline void run_loop::run() {
      while (auto* __task = __pop_front_()) {
        __task->__execute_();
      }
    }

    inline void run_loop::finish() {
      unique_lock __lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::__push_back_(__impl::__task* __task) {
      unique_lock __lock{__mutex_};
      if (__head_ == nullptr) {
        __head_ = __task;
      } else {
        __tail_->__next_ = __task;
      }
      __tail_ = __task;
      __task->__next_ = nullptr;
      __cv_.notify_one();
    }

    inline __impl::__task* run_loop::__pop_front_() {
      unique_lock __lock{__mutex_};
      while (__head_ == nullptr) {
        if (__stop_)
          return nullptr;
        __cv_.wait(__lock);
      }
      auto* __task = __head_;
      __head_ = __task->__next_;
      if (__head_ == nullptr)
        __tail_ = nullptr;
      return __task;
    }
  } // namespace __loop

  // NOT TO SPEC
  using run_loop = __loop::run_loop;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.schedule_from]
  inline namespace __schedule_from {
    namespace __impl {
      // Compute a data structure to store the source sender's completion.
      // The primary template assumes a non-typed sender, which uses type
      // erasure to store the completion information.
      struct __completion_storage_non_typed {
        template <receiver _Receiver>
          struct __f {
            any __tuple_;
            void (*__complete_)(_Receiver& __rcvr, any& __tupl) noexcept;

            template <class _Tuple, class... _Args>
              void emplace(_Args&&... __args) {
                __tuple_.emplace<_Tuple>((_Args&&) __args...);
                __complete_ = [](_Receiver& __rcvr, any& __tupl) noexcept {
                  try {
                    std::apply([&](auto __tag, auto&&... __args) -> void {
                      __tag((_Receiver&&) __rcvr, (decltype(__args)&&) __args...);
                    }, any_cast<_Tuple>(__tupl));
                  } catch(...) {
                    set_error((_Receiver&&) __rcvr, current_exception());
                  }
                };
              }

            void complete(_Receiver& __rcvr) noexcept {
              __complete_(__rcvr, __tuple_);
            }
          };
        };

      template <class _Sender, class _ScheduleFromReceiver>
        struct __completion_storage : __completion_storage_non_typed {};

      // This specialization is for typed senders, where the completion
      // information can be stored in situ within a variant in the operation
      // state
      template <class _Sender, class _ScheduleFromReceiver>
          requires typed_sender<_Sender, _ScheduleFromReceiver>
        struct __completion_storage<_Sender, _ScheduleFromReceiver> {
          // Compute a variant type that is capable of storing the results of the
          // input sender when it completes. The variant has type:
          //   variant<
          //     tuple<set_done_t>,
          //     tuple<set_value_t, decay_t<_Values1>...>,
          //     tuple<set_value_t, decay_t<_Values2>...>,
          //        ...
          //     tuple<set_error_t, decay_t<_Error1>>,
          //     tuple<set_error_t, decay_t<_Error2>>,
          //        ...
          //   >
          template <class... _Ts>
            using __bind_tuples =
              __bind_front_q<variant, tuple<set_done_t>, _Ts...>;

          using __bound_values_t =
            __value_types_of_t<
              _Sender,
              _ScheduleFromReceiver,
              __bind_front_q<__decayed_tuple, set_value_t>,
              __q<__bind_tuples>>;

          using __variant_t =
            __error_types_of_t<
              _Sender,
              _ScheduleFromReceiver,
              __transform<
                __bind_front_q<__decayed_tuple, set_error_t>,
                __bound_values_t>>;

          template <receiver _Receiver>
            struct __f : private __variant_t {
              __f() = default;
              using __variant_t::emplace;

              void complete(_Receiver& __rcvr) noexcept try {
                std::visit([&](auto&& __tupl) -> void {
                  std::apply([&](auto __tag, auto&&... __args) -> void {
                    __tag((_Receiver&&) __rcvr, (decltype(__args)&&) __args...);
                  }, (decltype(__tupl)&&) __tupl);
                }, (__variant_t&&) *this);
              } catch(...) {
                set_error((_Receiver&&) __rcvr, current_exception());
              }
            };
        };

      template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
        struct __operation1;

      template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
        struct __receiver1;

      // This receiver is to be completed on the execution context
      // associated with the scheduler. When the source sender
      // completes, the completion information is saved off in the
      // operation state so that when this receiver completes, it can
      // read the completion out of the operation state and forward it
      // to the output receiver after transitioning to the scheduler's
      // context.
      template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
        struct __receiver2 {
          using _Receiver = __t<_ReceiverId>;
          __operation1<_SchedulerId, _CvrefSenderId, _ReceiverId>* __op_state_;

          // If the work is successfully scheduled on the new execution
          // context and is ready to run, forward the completion signal in
          // the operation state
          friend void tag_invoke(set_value_t, __receiver2&& __self) noexcept {
            __self.__op_state_->__data_.complete(__self.__op_state_->__rcvr_);
          }

          template <__one_of<set_error_t, set_done_t> _Tag, class... _Args>
            requires __callable<_Tag, _Receiver, _Args...>
          friend void tag_invoke(_Tag, __receiver2&& __self, _Args&&... __args) noexcept {
            _Tag{}((_Receiver&&) __self.__op_state_->__rcvr_, (_Args&&) __args...);
          }

          template <__receiver_query _Tag, class... _Args>
            requires __callable<_Tag, const _Receiver&, _Args...>
          friend decltype(auto) tag_invoke(_Tag __tag, const __receiver2& __self, _Args&&... __args)
            noexcept(__nothrow_callable<_Tag, const _Receiver&, _Args...>) {
            return ((_Tag&&) __tag)(as_const(__self.__op_state_->__rcvr_), (_Args&&) __args...);
          }
        };

      // This receiver is connected to the input sender. When that
      // sender completes (on whatever context it completes on), save
      // the completion information into the operation state. Then,
      // schedule a second operation to complete on the execution
      // context of the scheduler. That second receiver will read the
      // completion information out of the operation state and propagate
      // it to the output receiver from within the desired context.
      template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
        struct __receiver1 {
          using _CvrefSender = __t<_CvrefSenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver2_t =
            __receiver2<_SchedulerId, _CvrefSenderId, _ReceiverId>;
          __operation1<_SchedulerId, _CvrefSenderId, _ReceiverId>* __op_state_;

          template <__one_of<set_value_t, set_error_t, set_done_t> _Tag, class... _Args>
            requires __callable<_Tag, _Receiver, _Args...>
          friend void tag_invoke(_Tag, __receiver1&& __self, _Args&&... __args) noexcept try {
            // Write the tag and the args into the operation state so that
            // we can forward the completion from within the scheduler's
            // execution context.
            __self.__op_state_->__data_.template emplace<__decayed_tuple<_Tag, _Args...>>(_Tag{}, (_Args&&) __args...);
            // Schedule the completion to happen on the scheduler's
            // execution context.
            __self.__op_state_->__state2_.emplace(
                __conv{[__op_state = __self.__op_state_] {
                  return connect(schedule(__op_state->__sched_), __receiver2_t{__op_state});
                }});
            // Enqueue the scheduled operation:
            start(*__self.__op_state_->__state2_);
          } catch(...) {
            set_error((_Receiver&&) __self.__op_state_->__rcvr_, current_exception());
          }

          template <__receiver_query _Tag, class... _Args>
            requires __callable<_Tag, const _Receiver&, _Args...>
          friend decltype(auto) tag_invoke(_Tag __tag, const __receiver1& __self, _Args&&... __args)
            noexcept(__nothrow_callable<_Tag, const _Receiver&, _Args...>) {
            return ((_Tag&&) __tag)(as_const(__self.__op_state_->__rcvr_), (_Args&&) __args...);
          }
        };

      template <class _SchedulerId, class _CvrefSenderId, class _ReceiverId>
        struct __operation1 {
          using _Scheduler = __t<_SchedulerId>;
          using _CvrefSender = __t<_CvrefSenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver1_t =
            __receiver1<_SchedulerId, _CvrefSenderId, _ReceiverId>;
          using __receiver2_t =
            __receiver2<_SchedulerId, _CvrefSenderId, _ReceiverId>;

          _Scheduler __sched_;
          _Receiver __rcvr_;
          __minvoke<__completion_storage<_CvrefSender, __receiver1_t>, _Receiver> __data_;
          connect_result_t<_CvrefSender, __receiver1_t> __state1_;
          optional<connect_result_t<schedule_result_t<_Scheduler>, __receiver2_t>> __state2_;

          __operation1(_Scheduler __sched, _CvrefSender&& __sndr, __decays_to<_Receiver> auto&& __rcvr)
            : __sched_(__sched)
            , __rcvr_((decltype(__rcvr)&&) __rcvr)
            , __state1_(connect((_CvrefSender&&) __sndr, __receiver1_t{this})) {}

          friend void tag_invoke(start_t, __operation1& __op_state) noexcept {
            start(__op_state.__state1_);
          }
        };

      template <class _SchedulerId, class _SenderId>
        struct __sender {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          _Scheduler __sched_;
          _Sender __sndr_;

          template <__decays_to<__sender> _Self, receiver _Receiver>
            requires sender_to<__member_t<_Self, _Sender>, _Receiver>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation1<_SchedulerId, __x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>> {
            return {__self.__sched_, ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr};
          }

          template <__one_of<set_value_t, set_done_t> _Tag>
          friend _Scheduler tag_invoke(get_completion_scheduler_t<_Tag>, const __sender& __self) noexcept {
            return __self.__sched_;
          }

          template <__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>) {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

          template <class _Self, class _Receiver>
            using __receiver1_t =
              __receiver1<_SchedulerId, __x<__member_t<_Self, _Sender>>, __x<decay_t<_Receiver>>>;

          template <__decays_to<__sender> _Self, class _Receiver>
          friend constexpr auto tag_invoke(get_sender_traits_t, _Self&&, _Receiver&&) noexcept
              -> sender_traits<__member_t<_Self, _Sender>, __receiver1_t<_Self, _Receiver>> {
            return {};
          }
        };
    } // namespace __impl

    inline constexpr struct schedule_from_t {
      // NOT TO SPEC: permit non-typed senders:
      template <scheduler _Scheduler, sender _Sender>
        requires tag_invocable<schedule_from_t, _Scheduler, _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<schedule_from_t, _Scheduler, _Sender>)
        -> tag_invoke_result_t<schedule_from_t, _Scheduler, _Sender> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      // NOT TO SPEC: permit non-typed senders:
      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        -> __impl::__sender<__x<decay_t<_Scheduler>>, __x<decay_t<_Sender>>> {
        return {(_Scheduler&&) __sched, (_Sender&&) __sndr};
      }
    } schedule_from {};
  } // namespace __schedule_from

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.transfer]
  inline namespace __transfer {
    inline constexpr struct transfer_t {
      template <sender _Sender, scheduler _Scheduler>
        requires __tag_invocable_with_completion_scheduler<transfer_t, set_value_t, _Sender, _Scheduler>
      tag_invoke_result_t<transfer_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Scheduler>
      operator()(_Sender&& __sndr, _Scheduler&& __sched) const
        noexcept(nothrow_tag_invocable<transfer_t, __completion_scheduler_for<_Sender, set_value_t>, _Sender, _Scheduler>) {
        auto csch = get_completion_scheduler<set_value_t>(__sndr);
        return tag_invoke(transfer_t{}, std::move(csch), (_Sender&&) __sndr, (_Scheduler&&) __sched);
      }
      template <sender _Sender, scheduler _Scheduler>
        requires (!__tag_invocable_with_completion_scheduler<transfer_t, set_value_t, _Sender, _Scheduler>) &&
          tag_invocable<transfer_t, _Sender, _Scheduler>
      tag_invoke_result_t<transfer_t, _Sender, _Scheduler>
      operator()(_Sender&& __sndr, _Scheduler&& __sched) const noexcept(nothrow_tag_invocable<transfer_t, _Sender, _Scheduler>) {
        return tag_invoke(transfer_t{}, (_Sender&&) __sndr, (_Scheduler&&) __sched);
      }
      // NOT TO SPEC: permit non-typed senders:
      template <sender _Sender, scheduler _Scheduler>
        requires (!__tag_invocable_with_completion_scheduler<transfer_t, set_value_t, _Sender, _Scheduler>) &&
          (!tag_invocable<transfer_t, _Sender, _Scheduler>)
      auto operator()(_Sender&& __sndr, _Scheduler&& __sched) const {
        return schedule_from((_Scheduler&&) __sched, (_Sender&&) __sndr);
      }
      template <scheduler _Scheduler>
      __binder_back<transfer_t, decay_t<_Scheduler>> operator()(_Scheduler&& __sched) const {
        return {{}, {}, {(_Scheduler&&) __sched}};
      }
    } transfer {};
  } // namespace __transfer

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  inline namespace __on {
    namespace __impl {
      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __operation;

      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __receiver_ref
          : receiver_adaptor<__receiver_ref<_SchedulerId, _SenderId, _ReceiverId>> {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          __operation<_SchedulerId, _SenderId, _ReceiverId>* __op_state_;
          _Receiver&& base() && noexcept {
            return (_Receiver&&) __op_state_->__rcvr_;
          }
          const _Receiver& base() const & noexcept {
            return __op_state_->__rcvr_;
          }
          friend _Scheduler tag_invoke(get_scheduler_t,
                                      const __receiver_ref& __self) noexcept {
            return __self.__op_state_->__scheduler_;
          }
        };

      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __receiver
          : receiver_adaptor<__receiver<_SchedulerId, _SenderId, _ReceiverId>> {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_ref_t =
            __receiver_ref<_SchedulerId, _SenderId, _ReceiverId>;
          __operation<_SchedulerId, _SenderId, _ReceiverId>* __op_state_;
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
              start(__op_state->__data_.template emplace<1>(__conv{
                [__op_state] {
                  return connect((_Sender&&) __op_state->__sndr_,
                                  __receiver_ref_t{{}, __op_state});
                }
              }));
            } catch(...) {
              set_error((_Receiver&&) __op_state->__rcvr_,
                        current_exception());
            }
          }
        };

      template <class _SchedulerId, class _SenderId, class _ReceiverId>
        struct __operation {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          using __receiver_t = __receiver<_SchedulerId, _SenderId, _ReceiverId>;
          using __receiver_ref_t = __receiver_ref<_SchedulerId, _SenderId, _ReceiverId>;

          friend void tag_invoke(start_t, __operation& __self) noexcept {
            start(std::get<0>(__self.__data_));
          }

          template <class _Sender2, class _Receiver2>
          __operation(_Scheduler __sched, _Sender2&& __sndr, _Receiver2&& __rcvr)
            : __data_{in_place_index<0>, __conv{[&, this]{
                return connect(schedule(__sched),
                                __receiver_t{{}, this});
              }}}
            , __scheduler_((_Scheduler&&) __sched)
            , __sndr_((_Sender2&&) __sndr)
            , __rcvr_((_Receiver2&&) __rcvr) {}

          variant<
              connect_result_t<schedule_result_t<_Scheduler>, __receiver_t>,
              connect_result_t<_Sender, __receiver_ref_t>> __data_;
          _Scheduler __scheduler_;
          _Sender __sndr_;
          _Receiver __rcvr_;
        };

      template <class _SchedulerId, class _SenderId>
        struct __sender {
          using _Scheduler = __t<_SchedulerId>;
          using _Sender = __t<_SenderId>;
          template <class _ReceiverId>
            using __receiver_ref_t =
              __receiver_ref<_SchedulerId, _SenderId, _ReceiverId>;
          template <class _ReceiverId>
            using __receiver_t =
              __receiver<_SchedulerId, _SenderId, _ReceiverId>;
          template <class _ReceiverId>
            using __operation_t =
              __operation<_SchedulerId, _SenderId, _ReceiverId>;

          _Scheduler __scheduler_;
          _Sender __sndr_;

          template <__decays_to<__sender> _Self, receiver _Receiver>
            requires constructible_from<_Sender, __member_t<_Self, _Sender>> &&
              sender_to<schedule_result_t<_Scheduler>,
                        __receiver_t<__x<decay_t<_Receiver>>>> &&
              sender_to<_Sender, __receiver_ref_t<__x<decay_t<_Receiver>>>>
          friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
            -> __operation_t<__x<decay_t<_Receiver>>> {
            return {((_Self&&) __self).__scheduler_,
                    ((_Self&&) __self).__sndr_,
                    (_Receiver&&) __rcvr};
          }

          template <__sender_query _Tag, class... _As>
            requires __callable<_Tag, const _Sender&, _As...>
          friend decltype(auto) tag_invoke(_Tag __tag, const __sender& __self, _As&&... __as)
            noexcept(__nothrow_callable<_Tag, const _Sender&, _As...>) {
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          }

          template <__decays_to<__sender> _Self, class _Receiver>
          friend constexpr auto tag_invoke(get_sender_traits_t, _Self&&, _Receiver&&) noexcept
            -> sender_traits<__member_t<_Self, _Sender>, __receiver_ref_t<__x<decay_t<_Receiver>>>> {
            return {};
          }
        };
    } // namespace __impl

    inline constexpr struct on_t {
      template <scheduler _Scheduler, sender _Sender>
        requires tag_invocable<on_t, _Scheduler, _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        noexcept(nothrow_tag_invocable<on_t, _Scheduler, _Sender>)
        -> tag_invoke_result_t<on_t, _Scheduler, _Sender> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Sender&&) __sndr);
      }

      template <scheduler _Scheduler, sender _Sender>
      auto operator()(_Scheduler&& __sched, _Sender&& __sndr) const
        -> __impl::__sender<__x<decay_t<_Scheduler>>,
                            __x<decay_t<_Sender>>> {
        return {(_Scheduler&&) __sched, (_Sender&&) __sndr};
      }
    } on {};
  } // namespace __on

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  inline namespace __transfer_just {
    inline constexpr struct transfer_just_t {
      template <scheduler _Scheduler, __movable_value... _Values>
        requires tag_invocable<transfer_just_t, _Scheduler, _Values...> &&
          sender<tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...>>
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const
        noexcept(nothrow_tag_invocable<transfer_just_t, _Scheduler, _Values...>)
        -> tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...> {
        return tag_invoke(*this, (_Scheduler&&) __sched, (_Values&&) __vals...);
      }
      template <scheduler _Scheduler, __movable_value... _Values>
        requires (!tag_invocable<transfer_just_t, _Scheduler, _Values...> ||
          !sender<tag_invoke_result_t<transfer_just_t, _Scheduler, _Values...>>)
      auto operator()(_Scheduler&& __sched, _Values&&... __vals) const
        -> decltype(transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched)) {
        return transfer(just((_Values&&) __vals...), (_Scheduler&&) __sched);
      }
    } transfer_just {};
  } // namespace __transfer_just

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.into_variant]
  inline namespace __into_variant {
    namespace __impl {
      template <class _Sender, class _Receiver>
          requires typed_sender<_Sender, _Receiver>
        using __into_variant_result_t =
          value_types_of_t<_Sender, _Receiver>;

      template <class _SenderId, class _ReceiverId>
        class __receiver
          : receiver_adaptor<__receiver<_SenderId, _ReceiverId>, __t<_ReceiverId>> {
          using _Sender = __t<_SenderId>;
          using _Receiver = __t<_ReceiverId>;
          friend receiver_adaptor<__receiver, _Receiver>;
          template <class _Self>
          using __variant_t = __into_variant_result_t<_Sender, _Self>;

          // Customize set_value by invoking the invocable and passing the result
          // to the base class
          template <class _Self = __receiver, class... _As>
            requires constructible_from<__variant_t<_Self>, tuple<_As&&...>>
          void set_value(_As&&... __as) && noexcept try {
            using _Variant = __into_variant_result_t<_Sender, __variant_t<_Self>>;
            execution::set_value(
                ((__receiver&&) *this).base(),
                _Variant{tuple<_As&&...>{(_As&&) __as...}});
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }

         public:
          using receiver_adaptor<__receiver, _Receiver>::receiver_adaptor;
        };

      template <class, class>
        struct __traits {};

      template <class _SenderId, class _ReceiverId>
          requires typed_sender<__t<_SenderId>, __receiver<_SenderId, _ReceiverId>>
        struct __traits<_SenderId, _ReceiverId> {
          using _Sender = __t<_SenderId>;
          using __receiver_t = __receiver<_SenderId, _ReceiverId>;
          using __variant_t = __into_variant_result_t<_Sender, __receiver_t>;

          template <template <class...> class _Tuple, template <class...> class _Variant>
            using value_types = _Variant<_Tuple<__variant_t>>;

          template <template <class...> class _Variant>
            using error_types =
              __minvoke2<
                __push_back_unique<__q<_Variant>>,
                error_types_of_t<_Sender, __receiver_t, __types>,
                exception_ptr>;

          static constexpr bool sends_done =
            __sends_done<_Sender, __receiver_t>();
        };

      template <class _SenderId>
        struct __sender : sender_adaptor<__sender<_SenderId>, __t<_SenderId>> {
          using _Sender = __t<_SenderId>;
          friend sender_adaptor<__sender, _Sender>;
          template <class _Receiver>
            using __receiver_t = __receiver<_SenderId, __x<remove_cvref_t<_Receiver>>>;

          template <receiver _Receiver>
            requires __valid<__into_variant_result_t, _Sender, __receiver_t<_Receiver>> &&
              receiver_of<_Receiver, __into_variant_result_t<_Sender, __receiver_t<_Receiver>>> &&
              sender_to<_Sender, __receiver_t<_Receiver>>
          auto connect(_Receiver&& __rcvr) && noexcept(
            __has_nothrow_connect<_Sender, __receiver_t<_Receiver>>)
            -> connect_result_t<_Sender, __receiver_t<_Receiver>> {
            return execution::connect(
                ((__sender&&) *this).base(),
                __receiver_t<_Receiver>{(_Receiver&&) __rcvr});
          }

          template <class _Receiver>
          friend constexpr __traits<__x<_Sender>, __x<_Receiver>>
          tag_invoke(get_sender_traits_t, const __sender&, const _Receiver&) noexcept {
            return {};
          }

         public:
          using sender_adaptor<__sender, _Sender>::sender_adaptor;
        };
    } // namespace __impl

    inline constexpr struct __into_variant_t {
      template <sender _Sender>
        auto operator()(_Sender&& __sndr) const
          -> __impl::__sender<__x<remove_cvref_t<_Sender>>> {
          return __impl::__sender<__x<remove_cvref_t<_Sender>>>{(_Sender&&) __sndr};
        }
      auto operator()() const noexcept {
        return __binder_back<__into_variant_t>{};
      }
    } into_variant {};
  } // namespace __into_variant

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  // [execution.senders.adaptors.when_all_with_variant]
  inline namespace __when_all {
    namespace __impl {
      enum __state_t { __started, __error, __done };

      struct __on_stop_requested {
        in_place_stop_source& __stop_source_;
        void operator()() noexcept {
          __stop_source_.request_stop();
        }
      };

      template <class... _SenderIds>
        struct __sender {
          template <class... _Sndrs>
            explicit __sender(_Sndrs&&... __sndrs)
              : __sndrs_((_Sndrs&&) __sndrs...)
            {}

         private:
          template <class, class = index_sequence_for<_SenderIds...>>
            struct __traits {};
          template <class _CvrefReceiverId>
            struct __operation;

          template <class _CvrefReceiverId, size_t _Index>
            struct __receiver : receiver_adaptor<__receiver<_CvrefReceiverId, _Index>> {
              using _WhenAll = __member_t<_CvrefReceiverId, __sender>;
              using _Receiver = __t<decay_t<_CvrefReceiverId>>;
              using _Traits = __traits<_CvrefReceiverId>;

              _Receiver&& base() && noexcept {
                return (_Receiver&&) __op_state_->__recvr_;
              }
              const _Receiver& base() const & noexcept {
                return __op_state_->__recvr_;
              }
              template <class _Error>
                void __set_error(_Error&& __err, __state_t __expected) noexcept {
                  // TODO: _What memory orderings are actually needed here?
                  if (__op_state_->__state_.compare_exchange_strong(__expected, __error)) {
                    __op_state_->__stop_source_.request_stop();
                    // We won the race, free to write the error into the operation
                    // state without worry.
                    try {
                      __op_state_->__errors_.template emplace<decay_t<_Error>>((_Error&&) __err);
                    } catch(...) {
                      __op_state_->__errors_.template emplace<exception_ptr>(current_exception());
                    }
                  }
                  __op_state_->__arrive();
                }
              template <class... _Values>
                void set_value(_Values&&... __vals) && noexcept {
                  if constexpr (_Traits::__has_values) {
                    // We only need to bother recording the completion values
                    // if we're not already in the "error" or "done" state.
                    if (__op_state_->__state_ == __started) {
                      try {
                        std::get<_Index>(__op_state_->__values_).emplace(
                            (_Values&&) __vals...);
                      } catch(...) {
                        __set_error(current_exception(), __started);
                      }
                    }
                  }
                  __op_state_->__arrive();
                }
              template <class _Error>
                  requires receiver<_Receiver, _Error>
                void set_error(_Error&& __err) && noexcept {
                  __set_error((_Error&&) __err, __started);
                }
              void set_done() && noexcept {
                __state_t __expected = __started;
                // Transition to the "done" state if and only if we're in the
                // "started" state. (_If this fails, it's because we're in an
                // error state, which trumps cancellation.)
                if (__op_state_->__state_.compare_exchange_strong(__expected, __done)) {
                  __op_state_->__stop_source_.request_stop();
                }
                __op_state_->__arrive();
              }
              friend in_place_stop_token tag_invoke(
                  get_stop_token_t, const __receiver& __self) noexcept {
                return __self.__op_state_->__stop_source_.get_token();
              }
              __operation<_CvrefReceiverId>* __op_state_;
            };

          template <class _Variant, class _CvrefReceiverId, size_t... _Is>
            using __error_types =
              __minvoke<
                __concat<__munique<_Variant>>,
                __types<exception_ptr>,
                error_types_of_t<
                  __member_t<_CvrefReceiverId, __t<_SenderIds>>,
                  __receiver<_CvrefReceiverId, _Is>,
                  __types>...>;

          template <class _Sender, class _CvrefReceiverId, size_t _Is, class _SingleOrVoid>
            using __is_typed_sender = __bool<
              requires {
                typename __value_types_of_t<
                  __member_t<_CvrefReceiverId, _Sender>,
                  __receiver<_CvrefReceiverId, _Is>,
                  __q<__types>,
                  _SingleOrVoid>;
                typename __error_types_of_t<
                    __member_t<_CvrefReceiverId, _Sender>,
                    __receiver<_CvrefReceiverId, _Is>,
                    __q<__types>>;
              }>;

          template <class _CvrefReceiverId, size_t... _Is>
            struct __traits_base {
              static constexpr bool __has_values = false;
              template <class, class _Variant>
                using __value_types = __minvoke<_Variant>;
            };
          template <class _CvrefReceiverId, size_t... _Is>
              requires (__is_typed_sender<
                  __t<_SenderIds>, _CvrefReceiverId, _Is, __q<__single_t>>()() &&...)
            struct __traits_base<_CvrefReceiverId, _Is...> {
              static constexpr bool __has_values = true;
              template <class _Tuple, class _Variant>
                using __value_types =
                  __minvoke<
                    _Variant,
                    __minvoke<
                      _Tuple,
                      value_types_of_t<
                        __member_t<_CvrefReceiverId, __t<_SenderIds>>,
                        __receiver<_CvrefReceiverId, _Is>,
                        __types,
                        __single_t>...>>;
            };

          template <class _CvrefReceiverId, size_t... _Is>
              requires (__is_typed_sender<
                  __t<_SenderIds>, _CvrefReceiverId, _Is, __q<__single_or_void_t>>()() &&...)
            struct __traits<_CvrefReceiverId, index_sequence<_Is...>>
                : __traits_base<_CvrefReceiverId, _Is...> {
              template <template <class...> class _Tuple, template <class...> class _Variant>
                using value_types =
                  typename __traits::template __value_types<__concat<__q<_Tuple>>, __q<_Variant>>;

              template <template <class...> class _Variant>
                using error_types =
                  __error_types<__q<_Variant>, _CvrefReceiverId, _Is...>;

              static constexpr bool sends_done = true;
            };

          template <class _CvrefReceiverId>
            struct __operation {
              using _WhenAll = __member_t<_CvrefReceiverId, __sender>;
              using _Receiver = __t<decay_t<_CvrefReceiverId>>;

              template <class _Sender, size_t _Index>
                using __child_op_state =
                  connect_result_t<__member_t<_WhenAll, _Sender>, __receiver<_CvrefReceiverId, _Index>>;

              using _Indices = index_sequence_for<_SenderIds...>;

              template <size_t... _Is>
                static auto __connect_children(
                    __operation* __self, _WhenAll&& __when_all, index_sequence<_Is...>)
                    -> tuple<__child_op_state<__t<_SenderIds>, _Is>...> {
                  return tuple<__child_op_state<__t<_SenderIds>, _Is>...>{
                    __conv{[&__when_all, __self]() {
                      return execution::connect(
                          std::get<_Is>(((_WhenAll&&) __when_all).__sndrs_),
                          __receiver<_CvrefReceiverId, _Is>{{}, __self});
                    }}...
                  };
                }

              using __child_op_states_tuple_t =
                  decltype(__connect_children(nullptr, __declval<_WhenAll>(), _Indices{}));

              void __arrive() noexcept {
                if (0 == --__count_) {
                  __complete();
                }
              }

              void __complete() noexcept {
                // Stop callback is no longer needed. Destroy it.
                __on_stop_.reset();
                // All child operations have completed and arrived at the barrier.
                switch(__state_.load(memory_order_relaxed)) {
                case __started:
                  if constexpr (__traits<_CvrefReceiverId>::__has_values) {
                    // All child operations completed successfully:
                    std::apply(
                      [this](auto&... __opt_vals) -> void {
                        std::apply(
                          [this](auto&... __all_vals) -> void {
                            try {
                              execution::set_value(
                                  (_Receiver&&) __recvr_, std::move(__all_vals)...);
                            } catch(...) {
                              execution::set_error(
                                  (_Receiver&&) __recvr_, current_exception());
                            }
                          },
                          std::tuple_cat(
                            std::apply(
                              [](auto&... __vals) { return std::tie(__vals...); },
                              *__opt_vals
                            )...
                          )
                        );
                      },
                      __values_
                    );
                  }
                  break;
                case __error:
                  std::visit([this](auto& __err) noexcept {
                    execution::set_error((_Receiver&&) __recvr_, std::move(__err));
                  }, __errors_);
                  break;
                case __done:
                  execution::set_done((_Receiver&&) __recvr_);
                  break;
                default:
                  ;
                }
              }

              __operation(_WhenAll&& when_all, _Receiver __rcvr)
                : __child_states_{__connect_children(this, (_WhenAll&&) when_all, _Indices{})}
                , __recvr_((_Receiver&&) __rcvr)
              {}

              friend void tag_invoke(start_t, __operation& __self) noexcept {
                // register stop callback:
                __self.__on_stop_.emplace(
                    get_stop_token(__self.__recvr_),
                    __on_stop_requested{__self.__stop_source_});
                if (__self.__stop_source_.stop_requested()) {
                  // Stop has already been requested. Don't bother starting
                  // the child operations.
                  execution::set_done((_Receiver&&) __self.__recvr_);
                } else {
                  apply([](auto&&... __child_ops) noexcept -> void {
                    (execution::start(__child_ops), ...);
                  }, __self.__child_states_);
                }
              }

              // tuple<optional<tuple<Vs1...>>, optional<tuple<Vs2...>>, ...>
              using __child_values_tuple_t =
                typename __traits<_CvrefReceiverId>::template __value_types<
                  __transform<__uncurry<__compose<__q1<optional>, __q<__decayed_tuple>>>>,
                  __uncurry<__q<tuple>>>;

              __child_op_states_tuple_t __child_states_;
              _Receiver __recvr_;
              atomic<size_t> __count_{sizeof...(_SenderIds)};
              // Could be non-atomic here and atomic_ref everywhere except __completion_fn
              atomic<__state_t> __state_{__started};
              error_types_of_t<__sender, _Receiver, variant> __errors_{};
              [[no_unique_address]] __child_values_tuple_t __values_{};
              in_place_stop_source __stop_source_{};
              optional<typename stop_token_of_t<_Receiver&>::template
                  callback_type<__on_stop_requested>> __on_stop_{};
            };

          template <class _Receiver, class... _Values>
            using __receiver_of = __bool<receiver_of<_Receiver, _Values...>>;

          template <class _Self, class _Receiver>
            using __can_connect_to_t =
              __value_types_of_t<
                __traits<__member_t<_Self, __x<decay_t<_Receiver>>>>,
                _Receiver,
                __bind_front_q<__receiver_of, _Receiver>,
                __q<__single_or_void_t>>;

          template <__decays_to<__sender> _Self, receiver _Receiver>
              requires __is_true<__can_connect_to_t<_Self, _Receiver>>
            friend auto tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr)
              -> __operation<__member_t<_Self, __x<decay_t<_Receiver>>>> {
              return {(_Self&&) __self, (_Receiver&&) __rcvr};
            }

          template <__decays_to<__sender> _Self, receiver _Receiver>
            friend auto tag_invoke(get_sender_traits_t, _Self&& __self, _Receiver&& __rcvr)
              -> __traits<__member_t<_Self, __x<decay_t<_Receiver>>>> {
              return {};
            }

          tuple<__t<_SenderIds>...> __sndrs_;
        };

      template <class _Sender>
        using __into_variant_result_t =
          decltype(into_variant(__declval<_Sender>()));

      template <class _Sender>
        concept __sender_with_zero_or_one_sets_of_values =
          sender<_Sender> &&
          (!typed_sender<_Sender, __default_context> ||
            __valid<
              __value_types_of_t,
              _Sender,
              __default_context,
              __q<__types>,
              __q<__single_or_void_t>>);
    } // namespce __impl

    inline constexpr struct when_all_t {
      template <__impl::__sender_with_zero_or_one_sets_of_values... _Senders>
        requires tag_invocable<when_all_t, _Senders...> &&
          sender<tag_invoke_result_t<when_all_t, _Senders...>>
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<when_all_t, _Senders...>)
        -> tag_invoke_result_t<when_all_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <__impl::__sender_with_zero_or_one_sets_of_values... _Senders>
      auto operator()(_Senders&&... __sndrs) const
        -> __impl::__sender<__x<decay_t<_Senders>>...> {
        return __impl::__sender<__x<decay_t<_Senders>>...>{
            (_Senders&&) __sndrs...};
      }
    } when_all {};

    inline constexpr struct when_all_with_variant_t {
      template <sender... _Senders>
        requires tag_invocable<when_all_with_variant_t, _Senders...> &&
          sender<tag_invoke_result_t<when_all_with_variant_t, _Senders...>>
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<when_all_with_variant_t, _Senders...>)
        -> tag_invoke_result_t<when_all_with_variant_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <sender... _Senders>
        requires (!tag_invocable<when_all_with_variant_t, _Senders...>) &&
          (__callable<__into_variant_t, _Senders> &&...)
      auto operator()(_Senders&&... __sndrs) const
        -> __impl::__sender<__impl::__into_variant_result_t<_Senders>...> {
        return __impl::__sender<__impl::__into_variant_result_t<_Senders>...>{
            into_variant((_Senders&&) __sndrs)...};
      }
    } when_all_with_variant {};

    inline constexpr struct transfer_when_all_t {
      template <scheduler _Sched, __impl::__sender_with_zero_or_one_sets_of_values... _Senders>
        requires tag_invocable<transfer_when_all_t, _Sched, _Senders...> &&
          sender<tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...>>
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<transfer_when_all_t, _Sched, _Senders...>)
        -> tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...> {
        return tag_invoke(*this, (_Sched&&) __sched, (_Senders&&) __sndrs...);
      }

      template <scheduler _Sched, __impl::__sender_with_zero_or_one_sets_of_values... _Senders>
        requires ((!tag_invocable<transfer_when_all_t, _Sched, _Senders...>) ||
          (!sender<tag_invoke_result_t<transfer_when_all_t, _Sched, _Senders...>>))
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const {
        return transfer(when_all((_Senders&&) __sndrs...), (_Sched&&) __sched);
      }
    } transfer_when_all {};

    inline constexpr struct transfer_when_all_with_variant_t {
      template <scheduler _Sched, sender... _Senders>
        requires tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...> &&
          sender<tag_invoke_result_t<transfer_when_all_with_variant_t, _Sched, _Senders...>>
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>)
        -> tag_invoke_result_t<transfer_when_all_with_variant_t, _Sched, _Senders...> {
        return tag_invoke(*this, (_Sched&&) __sched, (_Senders&&) __sndrs...);
      }

      template <scheduler _Sched, sender... _Senders>
        requires (!tag_invocable<transfer_when_all_with_variant_t, _Sched, _Senders...>) &&
          (__callable<__into_variant_t, _Senders> &&...)
      auto operator()(_Sched&& __sched, _Senders&&... __sndrs) const {
        return transfer_when_all((_Sched&&) __sched, into_variant((_Senders&&) __sndrs)...);
      }
    } transfer_when_all_with_variant {};
  } // namespace __when_all

  inline namespace __read_ {
    namespace __impl {
      template <class _Tag, class _ReceiverId>
        struct __operation {
          __t<_ReceiverId> __rcvr_;
          friend void tag_invoke(start_t, __operation& __self) noexcept try {
            set_value(std::move(__self.__rcvr_), _Tag{}(std::as_const(__self.__rcvr_)));
          } catch(...) {
            set_error(std::move(__self.__rcvr_), current_exception());
          }
        };

      template <class _Tag>
        struct __sender {
          template <class _Receiver>
            requires invocable<_Tag, __cref_t<_Receiver>> &&
              receiver_of<_Receiver, invoke_result_t<_Tag, __cref_t<_Receiver>>>
          friend auto tag_invoke(connect_t, __sender, _Receiver&& __rcvr)
            noexcept(is_nothrow_constructible_v<decay_t<_Receiver>, _Receiver>)
            -> __operation<_Tag, __x<decay_t<_Receiver>>> {
            return {(_Receiver&&) __rcvr};
          }

          friend __ tag_invoke(get_sender_traits_t, __sender, auto&&) noexcept;

          template <class _Receiver>
            requires invocable<_Tag, __cref_t<_Receiver>>
          friend auto tag_invoke(get_sender_traits_t, __sender, _Receiver&&) noexcept
            -> completion_signatures<
                set_value_t(invoke_result_t<_Tag, __cref_t<_Receiver>>),
                set_error_t(exception_ptr)>;
        };
    } // namespace __impl

    inline constexpr struct __read_t {
      template <class _Tag>
      constexpr __impl::__sender<_Tag> operator()(_Tag) const noexcept {
        return {};
      }
    } __read {};
  } // namespace __read_

  namespace __general_queries::__impl {
    inline auto get_scheduler_t::operator()() const noexcept {
      return __read_::__impl::__sender<get_scheduler_t>{};
    }
    inline auto get_allocator_t::operator()() const noexcept {
      return __read_::__impl::__sender<get_allocator_t>{};
    }
    inline auto get_stop_token_t::operator()() const noexcept {
      return __read_::__impl::__sender<get_stop_token_t>{};
    }
  }
} // namespace std::execution

namespace std::this_thread {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  // [execution.senders.consumers.sync_wait_with_variant]
  inline namespace __sync_wait {
    namespace __impl {
      template <class _Sender>
        using __into_variant_result_t =
          decltype(execution::into_variant(__declval<_Sender>()));

      template <class _SenderId>
        struct __receiver;

      // What should sync_wait(just_done()) return?
      template <class _Sender>
          requires execution::typed_sender<_Sender, __receiver<__x<_Sender>>>
        using __sync_wait_result_t =
          execution::value_types_of_t<
            _Sender,
            __receiver<__x<_Sender>>,
            execution::__decayed_tuple,
            __single_t>;

      template <class _Sender>
        using __sync_wait_with_variant_result_t =
          __sync_wait_result_t<__into_variant_result_t<_Sender>>;

      template <class _SenderId>
        struct __state;

      template <class _SenderId>
        struct __receiver {
          using _Sender = __t<_SenderId>;
          __state<_SenderId>* __state_;
          execution::run_loop* __loop_;
          template <class _Sender2 = _Sender, class... _As>
            requires constructible_from<__sync_wait_result_t<_Sender2>, _As...>
          friend void tag_invoke(execution::set_value_t, __receiver&& __rcvr, _As&&... __as) {
            __rcvr.__state_->__data_.template emplace<1>((_As&&) __as...);
            __rcvr.__loop_->finish();
          }
          friend void tag_invoke(execution::set_error_t, __receiver&& __rcvr, exception_ptr __err) noexcept {
            __rcvr.__state_->__data_.template emplace<2>((exception_ptr&&) __err);
            __rcvr.__loop_->finish();
          }
          friend void tag_invoke(execution::set_done_t __d, __receiver&& __rcvr) noexcept {
            __rcvr.__state_->__data_.template emplace<3>(__d);
            __rcvr.__loop_->finish();
          }
          friend execution::run_loop::__scheduler
          tag_invoke(execution::get_scheduler_t, const __receiver& __rcvr) noexcept {
            return __rcvr.__loop_->get_scheduler();
          }
          friend execution::run_loop::__scheduler
          tag_invoke(execution::get_delegee_scheduler_t, const __receiver& __rcvr) noexcept {
            return __rcvr.__loop_->get_scheduler();
          }
        };

      template <class _SenderId>
        struct __state {
          using _Tuple = __sync_wait_result_t<__t<_SenderId>>;
          variant<monostate, _Tuple, exception_ptr, execution::set_done_t> __data_;
        };

      template <class _Sender>
        using __into_variant_result_t =
          decltype(execution::into_variant(__declval<_Sender>()));
    } // namespace __impl

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait]
    inline constexpr struct sync_wait_t {
      // TODO: constrain on return type
      template <execution::sender _Sender> // NOT TO SPEC
        requires
          execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>
      tag_invoke_result_t<
        sync_wait_t,
        execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
        _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<
          sync_wait_t,
          execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
          _Sender>) {
        auto __sched =
          execution::get_completion_scheduler<execution::set_value_t>(__sndr);
        return tag_invoke(sync_wait_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      // TODO: constrain on return type
      template <execution::sender _Sender> // NOT TO SPEC
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>) &&
          tag_invocable<sync_wait_t, _Sender>
      tag_invoke_result_t<sync_wait_t, _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<sync_wait_t, _Sender>) {
        return tag_invoke(sync_wait_t{}, (_Sender&&) __sndr);
      }
      template <execution::sender _Sender>
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_t, execution::set_value_t, _Sender>) &&
          (!tag_invocable<sync_wait_t, _Sender>) &&
          execution::typed_sender<_Sender, __impl::__receiver<__x<_Sender>>>
      optional<__impl::__sync_wait_result_t<_Sender>>
      operator()(_Sender&& __sndr) const {
        using state_t = __impl::__state<__x<_Sender>>;
        state_t __state {};
        execution::run_loop __loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto __op_state =
          execution::connect(
            (_Sender&&) __sndr,
            __impl::__receiver<__x<_Sender>>{&__state, &__loop});
        execution::start(__op_state);

        // Wait for the variant to be filled in.
        __loop.run();

        if (__state.__data_.index() == 2)
          rethrow_exception(std::get<2>(__state.__data_));

        if (__state.__data_.index() == 3)
          return nullopt;

        return std::move(std::get<1>(__state.__data_));
      }
    } sync_wait {};

    ////////////////////////////////////////////////////////////////////////////
    // [execution.senders.consumers.sync_wait_with_variant]
    inline constexpr struct sync_wait_with_variant_t {
      template <execution::sender _Sender> // NOT TO SPEC
        requires
          execution::__tag_invocable_with_completion_scheduler<
            sync_wait_with_variant_t, execution::set_value_t, _Sender>
      tag_invoke_result_t<
        sync_wait_with_variant_t,
        execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
        _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<
          sync_wait_with_variant_t,
          execution::__completion_scheduler_for<_Sender, execution::set_value_t>,
          _Sender>) {
        auto __sched =
          execution::get_completion_scheduler<execution::set_value_t>(__sndr);
        return tag_invoke(
          sync_wait_with_variant_t{}, std::move(__sched), (_Sender&&) __sndr);
      }
      template <execution::sender _Sender> // NOT TO SPEC
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_with_variant_t, execution::set_value_t, _Sender>) &&
          tag_invocable<sync_wait_with_variant_t, _Sender>
      tag_invoke_result_t<sync_wait_with_variant_t, _Sender>
      operator()(_Sender&& __sndr) const noexcept(
        nothrow_tag_invocable<sync_wait_with_variant_t, _Sender>) {
        return tag_invoke(sync_wait_with_variant_t{}, (_Sender&&) __sndr);
      }
      template <execution::sender _Sender>
        requires
          (!execution::__tag_invocable_with_completion_scheduler<
            sync_wait_with_variant_t, execution::set_value_t, _Sender>) &&
          (!tag_invocable<sync_wait_with_variant_t, _Sender>) &&
          invocable<sync_wait_t, __impl::__into_variant_result_t<_Sender>>
      optional<__impl::__sync_wait_with_variant_result_t<_Sender>>
      operator()(_Sender&& __sndr) const {
        return sync_wait(execution::into_variant((_Sender&&) __sndr));
      }
    } sync_wait_with_variant {};
  } // namespace __sync_wait
} // namespace std::this_thread

_PRAGMA_POP()
