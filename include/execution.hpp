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

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <tuple>
#include <optional>
#include <variant>
#include <mutex>
#include <condition_variable>

#include <__utility.hpp>
#include <concepts.hpp>
#include <functional.hpp>
#include <coroutine.hpp>
#include <stop_token.hpp>

namespace std::execution {
  template<template<template<class...> class, template<class...> class> class>
    struct __test_has_values;

  template<template<template<class...> class> class>
    struct __test_has_errors;

  template<class T>
    concept __has_sender_types = requires {
      typename __test_has_values<T::template value_types>;
      typename __test_has_errors<T::template error_types>;
      typename bool_constant<T::sends_done>;
    };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.traits]
  using sender_base = struct __sender_base {};

  struct __no_sender_traits {
    using __unspecialized = void;
  };

  template <bool SendsDone>
  struct __void_sender {
    template<template<class...> class Tuple, template<class...> class Variant>
      using value_types = Variant<Tuple<>>;
    template<template<class...> class Variant>
      using error_types = Variant<std::exception_ptr>;
    static constexpr bool sends_done = SendsDone;
  };

  template <bool SendsDone, class... Ts>
    struct __sender_of {
      template<template<class...> class Tuple, template<class...> class Variant>
        using value_types = Variant<Tuple<Ts...>>;
      template<template<class...> class Variant>
        using error_types = Variant<std::exception_ptr>;
      static constexpr bool sends_done = SendsDone;
    };

  template<class S>
    struct __typed_sender {
      template<template<class...> class Tuple, template<class...> class Variant>
        using value_types = typename S::template value_types<Tuple, Variant>;
      template<template<class...> class Variant>
        using error_types = typename S::template error_types<Variant>;
      static constexpr bool sends_done = S::sends_done;
    };

  template<class S>
  auto __sender_traits_base_fn() {
    if constexpr (__has_sender_types<S>) {
      return __typed_sender<S>{};
    } else if constexpr (derived_from<S, sender_base>) {
      return sender_base{};
    } else if constexpr (__awaitable<S>) { // NOT TO SPEC
      if constexpr (is_void_v<__await_result_t<S>>) {
        return __void_sender<false>{};
      } else {
        return __sender_of<false, __await_result_t<S>>{};
      }
    } else {
      return __no_sender_traits{};
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.traits]
  template<class S>
  struct sender_traits
    : decltype(__sender_traits_base_fn<S>()) {};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  inline namespace __receiver_cpo {
    inline constexpr struct set_value_t {
      template<class R, class... As>
        requires tag_invocable<set_value_t, R, As...>
      void operator()(R&& r, As&&... as) const
        noexcept(nothrow_tag_invocable<set_value_t, R, As...>) {
        (void) tag_invoke(set_value_t{}, (R&&) r, (As&&) as...);
      }
    } set_value{};

    inline constexpr struct set_error_t {
      template<class R, class E>
        requires tag_invocable<set_error_t, R, E>
      void operator()(R&& r, E&& e) const
        noexcept(nothrow_tag_invocable<set_error_t, R, E>) {
        (void) tag_invoke(set_error_t{}, (R&&) r, (E&&) e);
      }
    } set_error {};

    inline constexpr struct set_done_t {
      template<class R>
        requires tag_invocable<set_done_t, R>
      void operator()(R&& r) const
        noexcept(nothrow_tag_invocable<set_done_t, R>) {
        (void) tag_invoke(set_done_t{}, (R&&) r);
      }
    } set_done{};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.receivers]
  template<class R, class E = exception_ptr>
    concept receiver =
      move_constructible<remove_cvref_t<R>> &&
      constructible_from<remove_cvref_t<R>, R> &&
      requires(remove_cvref_t<R>&& r, E&& e) {
        { set_done(std::move(r)) } noexcept;
        { set_error(std::move(r), (E&&) e) } noexcept;
      };

  template<class R, class... An>
    concept receiver_of =
      receiver<R> &&
      requires(remove_cvref_t<R>&& r, An&&... an) {
        set_value((remove_cvref_t<R>&&) r, (An&&) an...);
      };

  // NOT TO SPEC
  template<class R, class...As>
    inline constexpr bool nothrow_receiver_of =
      receiver_of<R, As...> &&
      nothrow_tag_invocable<set_value_t, R, As...>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  template<class S>
    concept sender =
      move_constructible<remove_cvref_t<S>> &&
      !requires {
        typename sender_traits<remove_cvref_t<S>>::__unspecialized;
      };

  template<class S>
    concept typed_sender =
      sender<S> &&
      __has_sender_types<sender_traits<remove_cvref_t<S>>>;

  template <class... As>
    requires (sizeof...(As) != 0)
    struct __front;
  template <class A, class... As>
    struct __front<A, As...> {
      using type = A;
    };
  template <class... As>
    requires (sizeof...(As) == 1)
    using __single_t = __t<__front<As...>>;
  template <class... As>
    requires (sizeof...(As) <= 1)
    using __single_or_void_t = __t<__front<As..., void>>;

  template<class S>
    using __single_sender_value_t =
      typename sender_traits<remove_cvref_t<S>>
        ::template value_types<__single_or_void_t, __single_or_void_t>;

  template<class S>
    concept __single_typed_sender =
      typed_sender<S> &&
      requires { typename __single_sender_value_t<S>; };
<<<<<<< HEAD
=======

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.schedule]
  inline namespace __schedule {
    inline constexpr struct schedule_t {
      template<class S>
        requires tag_invocable<schedule_t, S> &&
          sender<tag_invoke_result_t<schedule_t, S>>
      auto operator()(S&& s) const
        noexcept(nothrow_tag_invocable<schedule_t, S>) {
        return tag_invoke(schedule_t{}, (S&&) s);
      }
    } schedule {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.schedulers]
  template<class S>
    concept scheduler =
      copy_constructible<remove_cvref_t<S>> &&
      equality_comparable<remove_cvref_t<S>> &&
      requires(S&& s) {
        schedule((S&&) s);
      };

  // NOT TO SPEC
  template <scheduler S>
    using schedule_result_t = decltype(schedule(std::declval<S>()));

  // [execution.receivers.queries], receiver queries
  inline namespace __receiver_queries {
    namespace __impl {
      template <class T>
        using __cref_t = const remove_reference_t<T>&;

      // TODO: implement allocator concept
      template <class A>
        concept __allocator = true;

      struct get_scheduler_t {
        template <receiver R>
          requires nothrow_tag_invocable<get_scheduler_t, __cref_t<R>> &&
            scheduler<tag_invoke_result_t<get_scheduler_t, __cref_t<R>>>
        tag_invoke_result_t<get_scheduler_t, __cref_t<R>> operator()(R&& r) const
          noexcept(nothrow_tag_invocable<get_scheduler_t, __cref_t<R>>) {
          return tag_invoke(get_scheduler_t{}, std::as_const(r));
        }
      };

      struct get_allocator_t {
        template <receiver R>
          requires nothrow_tag_invocable<get_allocator_t, __cref_t<R>> &&
            __allocator<tag_invoke_result_t<get_allocator_t, __cref_t<R>>>
        tag_invoke_result_t<get_allocator_t, __cref_t<R>> operator()(R&& r) const
          noexcept(nothrow_tag_invocable<get_allocator_t, __cref_t<R>>) {
          return tag_invoke(get_allocator_t{}, std::as_const(r));
        }
      };

      struct get_stop_token_t {
        template <receiver R>
          requires tag_invocable<get_stop_token_t, __cref_t<R>> &&
            stoppable_token<tag_invoke_result_t<get_stop_token_t, __cref_t<R>>>
        tag_invoke_result_t<get_stop_token_t, __cref_t<R>> operator()(R&& r) const
          noexcept(nothrow_tag_invocable<get_stop_token_t, __cref_t<R>>) {
          return tag_invoke(get_stop_token_t{}, std::as_const(r));
        }
        never_stop_token operator()(receiver auto&&) const noexcept {
          return {};
        }
      };
    }
    using __impl::get_allocator_t;
    using __impl::get_scheduler_t;
    using __impl::get_stop_token_t;
    inline constexpr get_scheduler_t get_scheduler{};
    inline constexpr get_allocator_t get_allocator{};
    inline constexpr get_stop_token_t get_stop_token{};
  }

  // NOT TO SPEC
  template <class R>
    using stop_token_type_t = remove_cvref_t<decltype(get_stop_token(std::declval<R>()))>;
>>>>>>> origin/main

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  inline namespace __start {
    inline constexpr struct start_t {
      template<class O>
        requires tag_invocable<start_t, O&>
      void operator()(O& o) const noexcept(nothrow_tag_invocable<start_t, O&>) {
        (void) tag_invoke(start_t{}, o);
      }
    } start {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.op_state]
  template<class O>
    concept operation_state =
      destructible<O> &&
      is_object_v<O> &&
      requires (O& o) {
        {start(o)} noexcept;
      };

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: __connect_awaitable_
  inline namespace __connect_awaitable_ {
    namespace __impl {
      struct __op_base {
        struct __promise_base {
          coro::suspend_always initial_suspend() noexcept {
            return {};
          }
          [[noreturn]] coro::suspend_always final_suspend() noexcept {
            terminate();
          }
          [[noreturn]] void unhandled_exception() noexcept {
            terminate();
          }
          [[noreturn]] void return_void() noexcept {
            terminate();
          }
          template <class Func>
          auto yield_value(Func&& func) noexcept {
            struct awaiter {
              Func&& func_;
              bool await_ready() noexcept {
                return false;
              }
              void await_suspend(coro::coroutine_handle<>)
                noexcept(is_nothrow_invocable_v<Func>) {
                // If this throws, the runtime catches the exception,
                // resumes the __connect_awaitable coroutine, and immediately
                // rethrows the exception. The end result is that an
                // exception_ptr to the exception gets passed to set_error.
                ((Func &&) func_)();
              }
              [[noreturn]] void await_resume() noexcept {
                terminate();
              }
            };
            return awaiter{(Func &&) func};
          }
        };

        coro::coroutine_handle<> coro_;

        explicit __op_base(coro::coroutine_handle<> coro) noexcept
          : coro_(coro) {}

        __op_base(__op_base&& other) noexcept
          : coro_(exchange(other.coro_, {})) {}

        ~__op_base() {
          if (coro_)
            coro_.destroy();
        }

        friend void tag_invoke(start_t, __op_base& self) noexcept {
          self.coro_.resume();
        }
      };
      template<class R_>
        class __op : public __op_base {
          using R = __t<R_>;
        public:
          struct promise_type : __promise_base {
            template <class A>
            explicit promise_type(A&, R& r) noexcept
              : r_(r)
            {}

            coro::coroutine_handle<> unhandled_done() noexcept {
              set_done(std::move(r_));
              // Returning noop_coroutine here causes the __connect_awaitable
              // coroutine to never resume past the point where it co_await's
              // the awaitable.
              return coro::noop_coroutine();
            }

            __op get_return_object() noexcept {
              return __op{
                coro::coroutine_handle<promise_type>::from_promise(*this)};
            }

            // Pass through receiver queries
            template<class... As, invocable<R&, As...> CPO>
              requires (!__one_of<CPO, set_value_t, set_error_t, set_done_t>)
            friend auto tag_invoke(CPO cpo, const promise_type& self, As&&... as)
              noexcept(is_nothrow_invocable_v<CPO, R&, As...>)
              -> invoke_result_t<CPO, R&, As...> {
              return ((CPO&&) cpo)(self.r_, (As&&) as...);
            }

            R& r_;
          };

          using __op_base::__op_base;
        };
    }

    inline constexpr struct __fn {
    private:
      template <class R, class... Args>
        using __nothrow_ = bool_constant<nothrow_receiver_of<R, Args...>>;

      template <class A, class R>
      static __impl::__op<__id_t<remove_cvref_t<R>>> __impl(A&& a, R&& r) {
        exception_ptr ex;
        try {
          // This is a bit mind bending control-flow wise.
          // We are first evaluating the co_await expression.
          // Then the result of that is passed into a lambda
          // that curries a reference to the result into another
          // lambda which is then returned to 'co_yield'.
          // The 'co_yield' expression then invokes this lambda
          // after the coroutine is suspended so that it is safe
          // for the receiver to destroy the coroutine.
          auto fn = [&]<bool Nothrow>(bool_constant<Nothrow>, auto&&... as) noexcept {
            return [&]() noexcept(Nothrow) -> void {
              set_value((R&&) r, (add_rvalue_reference_t<__await_result_t<A>>) as...);
            };
          };
          if constexpr (is_void_v<__await_result_t<A>>)
            co_yield (co_await (A &&) a, fn(__nothrow_<R>{}));
          else
            co_yield fn(__nothrow_<R, __await_result_t<A>>{}, co_await (A &&) a);
        } catch (...) {
          ex = current_exception();
        }
        co_yield [&]() noexcept -> void {
          set_error((R&&) r, (exception_ptr&&) ex);
        };
      }
    public:
      template <__awaitable A, receiver R>
        requires receiver_of<R, __await_result_t<A>>
      __impl::__op<__id_t<remove_cvref_t<R>>> operator()(A&& a, R&& r) const {
        return __impl((A&&) a, (R&&) r);
      }
      template <__awaitable A, receiver R>
        requires same_as<void, __await_result_t<A>> && receiver_of<R>
      __impl::__op<__id_t<remove_cvref_t<R>>> operator()(A&& a, R&& r) const {
        return __impl((A&&) a, (R&&) r);
      }
    } __connect_awaitable{};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.connect]
  inline namespace __connect {
    inline constexpr struct connect_t {
      template<sender S, receiver R>
        requires tag_invocable<connect_t, S, R> &&
          operation_state<tag_invoke_result_t<connect_t, S, R>>
      auto operator()(S&& s, R&& r) const
        noexcept(nothrow_tag_invocable<connect_t, S, R>) {
        return tag_invoke(connect_t{}, (S&&) s, (R&&) r);
      }
      // NOT TO SPEC:
      template<__awaitable A, receiver R>
        requires (!tag_invocable<connect_t, A, R>)
      auto operator()(A&& a, R&& r) const {
        return __connect_awaitable((A&&) a, (R&&) r);
      }
    } connect {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders]
  template<class S, class R>
    concept sender_to =
      sender<S> &&
      receiver<R> &&
      requires (S&& s, R&& r) {
        connect((S&&) s, (R&&) r);
      };

  template<class S, class R>
    using connect_result_t = tag_invoke_result_t<connect_t, S, R>;

  /////////////////////////////////////////////////////////////////////////////
  // NOT TO SPEC: as_awaitable and with_awaitable_senders
  inline namespace __with_awaitable_senders {
    namespace __impl {
      struct __void {};
      template <class Value>
        using __value_or_void_t =
          conditional_t<is_void_v<Value>, __void, Value>;
      template <class Value>
        using __expected_t =
          variant<monostate, __value_or_void_t<Value>, std::exception_ptr>;

      template <class Value>
        struct __rec_base {
          template <class... Us>
            requires constructible_from<Value, Us...> ||
              (is_void_v<Value> && sizeof...(Us) == 0)
          friend void tag_invoke(set_value_t, __rec_base&& self, Us&&... us)
              noexcept(is_nothrow_constructible_v<Value, Us...> ||
                  is_void_v<Value>) {
            self.result_->template emplace<1>((Us&&) us...);
            self.continuation_.resume();
          }

          friend void tag_invoke(set_error_t, __rec_base&& self, exception_ptr eptr) noexcept {
            self.result_->template emplace<2>((exception_ptr&&) eptr);
            self.continuation_.resume();
          }

          __expected_t<Value>* result_;
          coro::coroutine_handle<> continuation_;
        };

      template <typename P_, typename Value>
        struct __awaitable_base {
          using Promise = __t<P_>;
          struct __rec : __rec_base<Value> {
            friend void tag_invoke(set_done_t, __rec&& self) noexcept {
              auto continuation = coro::coroutine_handle<Promise>::from_address(
                self.continuation_.address());
              continuation.promise().unhandled_done().resume();
            }

            // Forward other tag_invoke overloads to the promise
            template <class... As, invocable<Promise&, As...> CPO>
            friend auto tag_invoke(CPO cpo, const __rec& self, As&&... as)
                noexcept(is_nothrow_invocable_v<CPO, Promise&, As...>)
                -> invoke_result_t<CPO, Promise&, As...> {
              auto continuation = coro::coroutine_handle<Promise>::from_address(
                self.continuation_.address());
              return ((CPO&&) cpo)(continuation.promise(), (As&&) as...);
            }
          };

        bool await_ready() const noexcept {
          return false;
        }

        Value await_resume() {
          switch (result_.index()) {
          case 0: // receiver contract not satisfied
            assert(!"Should never get here");
            break;
          case 1: // set_value
            if constexpr (!is_void_v<Value>)
              return (Value&&) std::get<1>(result_);
            else
              return;
          case 2: // set_error
            std::rethrow_exception(std::get<2>(result_));
          }
          terminate();
        }

      protected:
        __expected_t<Value> result_;
      };

      template <typename P_, typename S_>
      struct __awaitable
        : __awaitable_base<P_, __single_sender_value_t<__t<S_>>> {
      private:
        using Promise = __t<P_>;
        using Sender = __t<S_>;
        using Base = __awaitable_base<P_, __single_sender_value_t<Sender>>;
        using __rec = typename Base::__rec;
        connect_result_t<Sender, __rec> op_;
      public:
        __awaitable(Sender&& sender, coro::coroutine_handle<> h)
          noexcept(/* TODO: is_nothrow_connectable_v<Sender, __rec>*/ false)
          : op_(connect((Sender&&)sender, __rec{{&this->result_, h}}))
        {}

        void await_suspend(coro::coroutine_handle<>) noexcept {
          start(op_);
        }
      };
    }

    inline constexpr struct as_awaitable_t {
      template <__single_typed_sender S, class Promise>
      auto operator()(S&& s, Promise& promise) const
        noexcept(/*TODO*/ false)
        -> __impl::__awaitable<__id_t<Promise>, __id_t<remove_cvref_t<S>>> {
        auto h = coro::coroutine_handle<Promise>::from_promise(promise);
        return {(S&&) s, h};
      }
    } as_awaitable{};

    template <class Promise>
    struct with_awaitable_senders;

    struct with_awaitable_senders_base {
      template <class OtherPromise>
      void set_continuation(coro::coroutine_handle<OtherPromise> h) noexcept {
        static_assert(!is_void_v<OtherPromise>);
        continuation_ = h;
        if constexpr (requires(OtherPromise& other) { other.unhandled_done(); }) {
          done_callback_ = [](void* address) noexcept -> coro::coroutine_handle<> {
            // This causes the rest of the coroutine (the part after the co_await
            // of the sender) to be skipped and invokes the calling coroutine's
            // done handler.
            return coro::coroutine_handle<OtherPromise>::from_address(address)
                .promise().unhandled_done();
          };
        }
        // If OtherPromise doesn't implement unhandled_done(), then if a "done" unwind
        // reaches this point, it's considered an unhandled exception and terminate()
        // is called.
      }

      coro::coroutine_handle<> continuation() const noexcept {
        return continuation_;
      }

      coro::coroutine_handle<> unhandled_done() noexcept {
        return (*done_callback_)(continuation_.address());
      }

    private:
      coro::coroutine_handle<> continuation_{};
      coro::coroutine_handle<> (*done_callback_)(void*) noexcept =
        [](void*) noexcept -> coro::coroutine_handle<> {
          std::terminate();
        };
    };

    template <class Promise>
    struct with_awaitable_senders : with_awaitable_senders_base {
      template <class Value>
      decltype(auto) await_transform(Value&& value) {
        static_assert(derived_from<Promise, with_awaitable_senders>);
        if constexpr (__awaitable<Value>)
          return (Value&&) value;
        else if constexpr (sender<Value>)
          return as_awaitable((Value&&) value, static_cast<Promise&>(*this));
        else
          return (Value&&) value;
      }
    };
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  // TODO: turn this into start_detached
  inline namespace __submit {
    namespace __impl {
      template<class S, class R_>
        struct __rec {
          using R = remove_cvref_t<R_>;
          struct __wrap {
            __rec* __this;
            // Forward all tag_invoke calls, including the receiver ops.
            template<__same_<__wrap> Self, class... As, invocable<__member_t<Self, R>, As...> Tag>
            friend auto tag_invoke(Tag tag, Self&& self, As&&... as)
                noexcept(is_nothrow_invocable_v<Tag, __member_t<Self, R>, As...>)
                -> invoke_result_t<Tag, __member_t<Self, R>, As...> {
              // If we are about to complete the receiver contract, delete the state as cleanup:
              struct _g_t {
                __rec* r_;
                ~_g_t() { delete r_; }
              } _g{__one_of<Tag, set_value_t, set_error_t, set_done_t> ? self.__this : nullptr};
              return ((Tag&&) tag)((__member_t<Self, R>&&) self.__this->__r, (As&&) as...);
            }
          };
          R __r;
          connect_result_t<S, __wrap> __state;
          __rec(S&& s, R_&& r)
            : __r((R_&&) r)
            , __state(connect((S&&) s, __wrap{this}))
          {}
        };
    }

    inline constexpr struct submit_t {
      template<receiver R, sender_to<R> S>
      void operator()(S&& s, R&& r) const noexcept(false) {
        start((new __impl::__rec<S, R>{(S&&) s, (R&&) r})->__state);
      }
      template<receiver R, sender_to<R> S>
        requires tag_invocable<submit_t, S, R>
      void operator()(S&& s, R&& r) const
        noexcept(nothrow_tag_invocable<submit_t, S, R>) {
        (void) tag_invoke(submit_t{}, (S&&) s, (R&&) r);
      }
    } submit {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  inline namespace __just {
    namespace __impl {
      template <class CPO, class... Ts>
        struct __traits {
          template<template<class...> class Tuple,
                   template<class...> class Variant>
          using value_types = Variant<Tuple<Ts...>>;

          template<template<class...> class Variant>
          using error_types = Variant<exception_ptr>;

          static const constexpr auto sends_done = false;
        };
      template<class Error>
        struct __traits<set_error_t, Error> {
          template<template<class...> class,
                   template<class...> class Variant>
          using value_types = Variant<>;

          template<template<class...> class Variant>
          using error_types = Variant<Error>;

          static const constexpr auto sends_done = false;
        };
      template<>
        struct __traits<set_done_t> {
          template<template<class...> class,
                   template<class...> class Variant>
          using value_types = Variant<>;

          template<template<class...> class Variant>
          using error_types = Variant<>;

          static const constexpr auto sends_done = true;
        };
      template <class CPO, class... Ts>
        struct __sender : __traits<CPO, Ts...> {
          tuple<Ts...> vs_;

          template<class R_>
          struct __op {
            using R = __t<R_>;
            std::tuple<Ts...> vs_;
            R r_;

            friend void tag_invoke(start_t, __op& op) noexcept try {
              std::apply([&op](Ts&... ts) {
                CPO{}((R&&) op.r_, (Ts&&) ts...);
              }, op.vs_);
            } catch(...) {
              set_error((R&&) op.r_, current_exception());
            }
          };

          // NOT TO SPEC: copy_constructible
          template<receiver_of<Ts...> R>
            requires (copy_constructible<Ts> &&...)
          friend auto tag_invoke(connect_t, const __sender& s, R&& r)
            noexcept((is_nothrow_copy_constructible_v<Ts> &&...))
            -> __op<__id_t<remove_cvref_t<R>>> {
            return {s.vs_, (R&&) r};
          }

          template<receiver_of<Ts...> R>
          friend auto tag_invoke(connect_t, __sender&& s, R&& r)
            noexcept((is_nothrow_move_constructible_v<Ts> &&...))
            -> __op<__id_t<remove_cvref_t<R>>> {
            return {((__sender&&) s).vs_, (R&&) r};
          }
        };
    }

    inline constexpr struct __just_t {
      template <class... Ts>
        requires (constructible_from<decay_t<Ts>, Ts> &&...)
      __impl::__sender<set_value_t, decay_t<Ts>...> operator()(Ts&&... ts) const
        noexcept((is_nothrow_constructible_v<decay_t<Ts>, Ts> &&...)) {
        return {{}, {(Ts&&) ts...}};
      }
    } just {};

    inline constexpr struct __just_error_t {
      template <class Error>
        requires constructible_from<decay_t<Error>, Error>
      __impl::__sender<set_error_t, Error> operator()(Error&& err) const
        noexcept(is_nothrow_constructible_v<decay_t<Error>, Error>) {
        return {{}, {(Error&&) err}};
      }
    } just_error {};

    inline constexpr struct __just_done_t {
      __impl::__sender<set_done_t> operator()() const noexcept {
        return {{}, {}};
      }
    } just_done {};
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.execute]
  inline namespace __execute {
    namespace __impl {
      template<class F>
        struct __as_receiver {
          F f_;
          friend void tag_invoke(set_value_t, __as_receiver&& r) noexcept(is_nothrow_invocable_v<F&>) {
            r.f_();
          }
          [[noreturn]]
          friend void tag_invoke(set_error_t, __as_receiver&&, std::exception_ptr) noexcept {
            terminate();
          }
          friend void tag_invoke(set_done_t, __as_receiver&&) noexcept {}
        };
    }

    inline constexpr struct execute_t {
      template<scheduler Sch, class F>
        requires invocable<F&> && move_constructible<F>
      void operator()(Sch&& sch, F f) const
        noexcept(noexcept(
          submit(schedule((Sch&&) sch), __impl::__as_receiver<F>{(F&&) f}))) {
        (void) submit(schedule((Sch&&) sch), __impl::__as_receiver<F>{(F&&) f});
      }
      template<scheduler Sch, class F>
        requires invocable<F&> &&
          move_constructible<F> &&
          tag_invocable<execute_t, Sch, F>
      void operator()(Sch&& sch, F f) const
        noexcept(nothrow_tag_invocable<execute_t, Sch, F>) {
        (void) tag_invoke(execute_t{}, (Sch&&) sch, (F&&) f);
      }
    } execute {};
  }

  // NOT TO SPEC:
  namespace __closure {
    template <class D>
      requires is_class_v<D> && same_as<D, remove_cvref_t<D>>
      struct sender_adaptor_closure;
  }
  using __closure::sender_adaptor_closure;

  template <class T>
    concept __sender_adaptor_closure =
      derived_from<remove_cvref_t<T>, sender_adaptor_closure<remove_cvref_t<T>>> &&
      move_constructible<remove_cvref_t<T>> &&
      constructible_from<remove_cvref_t<T>, T>;

  namespace __closure {
    template <class A, class B>
    struct __compose : sender_adaptor_closure<__compose<A, B>> {
      [[no_unique_address]] A a_;
      [[no_unique_address]] B b_;

      template <sender S>
        requires invocable<A, S> && invocable<B, invoke_result_t<A, S>>
      invoke_result_t<B, invoke_result_t<A, S>> operator()(S&& s) && {
        return ((B&&) b_)(((A&&) a_)((S&&) s));
      }

      template <sender S>
        requires invocable<const A&, S> && invocable<const B&, invoke_result_t<const A&, S>>
      invoke_result_t<B, invoke_result_t<A, S>> operator()(S&& s) const & {
        return b_(a_((S&&) s));
      }
    };

    template <class D>
      requires is_class_v<D> && same_as<D, remove_cvref_t<D>>
      struct sender_adaptor_closure
      {};

    template <__sender_adaptor_closure A, __sender_adaptor_closure B>
    __compose<remove_cvref_t<A>, remove_cvref_t<B>> operator|(A&& a, B&& b) {
      return {(A&&) a, (B&&) b};
    }

    template <sender S, __sender_adaptor_closure C>
      requires invocable<C, S>
    invoke_result_t<C, S> operator|(S&& s, C&& c) {
      return ((C&&) c)((S&&) s);
    }

    template <class Fn, class... As>
    struct __binder_back : sender_adaptor_closure<__binder_back<Fn, As...>> {
      [[no_unique_address]] Fn fn;
      tuple<As...> as;

      template <sender S>
        requires invocable<Fn, S, As...>
      invoke_result_t<Fn, S, As...> operator()(S&& s) &&
        noexcept(is_nothrow_invocable_v<Fn, S, As...>) {
        return std::apply([&s, this](As&... as) {
            return ((Fn&&) fn)((S&&) s, (As&&) as...);
          }, as);
      }

      template <sender S>
        requires invocable<Fn, S, As...>
      invoke_result_t<const Fn&, S, const As&...> operator()(S&& s) const &
        noexcept(is_nothrow_invocable_v<const Fn&, S, const As&...>) {
        return std::apply([&s, this](const As&... as) {
            return fn((S&&) s, as...);
          }, as);
      }
    };
  }

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  inline namespace __then {
    namespace __impl {
      template<class R_, class F>
        struct __receiver {
          using R = __t<R_>;
          [[no_unique_address]] R r_;
          [[no_unique_address]] F f_;

          // Customize set_value by invoking the callable and passing the result to the base class
          template<class... As>
            requires invocable<F, As...> &&
              receiver_of<R, invoke_result_t<F, As...>>
          friend void tag_invoke(set_value_t, __receiver&& r, As&&... as) noexcept(/*...*/ true) {
            set_value((R&&) r.r_, invoke((F&&) r.f_, (As&&) as...));
          }
          // Handle the case when the callable returns void
          template<class... As>
            requires invocable<F, As...> &&
              same_as<void, invoke_result_t<F, As...>> &&
              receiver_of<R>
          friend void tag_invoke(set_value_t, __receiver&& r, As&&... as) noexcept(/*...*/ true) {
            invoke((F&&) r.f_, (As&&) as...);
            set_value((R&&) r.r_);
          }
          // Forward all other tag_invoke CPOs.
          template <__same_<__receiver> Self, class... As, invocable<__member_t<Self, R>, As...> Tag>
          friend auto tag_invoke(Tag tag, Self&& r, As&&... as)
            noexcept(is_nothrow_invocable_v<Tag, __member_t<Self, R>, As...>)
            -> invoke_result_t<Tag, __member_t<Self, R>, As...> {
            return ((Tag&&) tag)(((Self&&) r).r_, (As&&) as...);
          }
        };

      template<class S_, class F>
        struct __sender {
          using S = __t<S_>;
          [[no_unique_address]] S s_;
          [[no_unique_address]] F f_;

          template<receiver R, class R2 = __id_t<remove_cvref_t<R>>>
            requires sender_to<S, __receiver<R2, F>>
          friend auto tag_invoke(connect_t, __sender&& self, R&& r)
            noexcept(/*todo*/ false)
            -> connect_result_t<S, __receiver<R2, F>> {
            return connect(
              (S&&) self.s_,
              __receiver<R2, F>{(R&&) r, (F&&) self.f_});
          }
        };
    }

    inline constexpr struct lazy_then_t {
      template<sender S, class F>
        requires tag_invocable<lazy_then_t, S, F>
      sender auto operator()(S&& s, F f) const
        noexcept(nothrow_tag_invocable<lazy_then_t, S, F>) {
        return tag_invoke(lazy_then_t{}, (S&&) s, (F&&) f);
      }
      template<sender S, class F>
      sender auto operator()(S&& s, F f) const {
        return __impl::__sender<__id_t<remove_cvref_t<S>>, F>{(S&&)s, (F&&)f};
      }
      template <class F>
      __closure::__binder_back<lazy_then_t, F> operator()(F f) const {
        return {{}, {}, {(F&&) f}};
      }
    } lazy_then {};

    inline constexpr struct then_t {
      template<sender S, class F>
        requires tag_invocable<then_t, S, F>
      sender auto operator()(S&& s, F f) const
        noexcept(nothrow_tag_invocable<then_t, S, F>) {
        return tag_invoke(then_t{}, (S&&) s, (F&&) f);
      }
      template<sender S, class F>
      sender auto operator()(S&& s, F f) const {
        return lazy_then((S&&) s, (F&&) f);
      }
      template <class F>
      __closure::__binder_back<then_t, F> operator()(F f) const {
        return {{}, {}, {(F&&) f}};
      }
    } then {};
  }

  // Make the lazy_then sender typed if the input sender is also typed.
  template <class S_, class F>
    requires typed_sender<__t<S_>>
  struct sender_traits<__then::__impl::__sender<S_, F>> {
    using S = __t<S_>;
    template <template<class...> class Tuple, template<class...> class Variant>
      using value_types =
        typename sender_traits<S>::template value_types<
          __bind_front<invoke_result_t, F>::template __f,
          __transform<
            Variant,
            __eval2<
              __if<is_void, __empty<Tuple>, __q<Tuple>>::template __f
            >::template __f
          >::template __f
        >;

    template <template<class...> class Variant>
      using error_types = typename sender_traits<S>::template error_types<Variant>;

    static constexpr bool sends_done = sender_traits<S>::sends_done;
  };
} // namespace std::execution

namespace std::this_thread {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  inline namespace __sync_wait {
    namespace __impl {
      // What should sync_wait(just_done()) return?
      template <class S>
        using __sync_wait_result_t =
            typename execution::sender_traits<
              remove_cvref_t<S>
            >::template value_types<tuple, execution::__single_t>;

      template <class Tuple>
        struct __state {
          mutex mtx;
          condition_variable cv;
          variant<monostate, Tuple, exception_ptr, execution::set_done_t> data;
          struct __receiver {
            __state* state_;
            template <class... As>
              requires constructible_from<Tuple, As...>
            friend void tag_invoke(execution::set_value_t, __receiver&& r, As&&... as) {
              r.state_->data.template emplace<1>((As&&) as...);
              r.state_->cv.notify_one();
            }
            friend void tag_invoke(execution::set_error_t, __receiver&& r, exception_ptr e) noexcept {
              r.state_->data.template emplace<2>((exception_ptr&&) e);
              r.state_->cv.notify_one();
            }
            friend void tag_invoke(execution::set_done_t d, __receiver&& r) noexcept {
              r.state_->data.template emplace<3>(d);
              r.state_->cv.notify_one();
            }
          };
        };
    }

    inline constexpr struct sync_wait_t {
      template <execution::typed_sender S>
      optional<__impl::__sync_wait_result_t<S>> operator()(S&& s) const {
        using state_t = __impl::__state<__impl::__sync_wait_result_t<S>>;
        state_t state {};

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto op = execution::connect((S&&) s, typename state_t::__receiver{&state});
        execution::start(op);

        // Wait for the variant to be filled in.
        {
          unique_lock g{state.mtx};
          state.cv.wait(g, [&]{return state.data.index() != 0;});
        }

        if (state.data.index() == 2)
          rethrow_exception(std::get<2>(state.data));
        if (state.data.index() == 3)
          return nullopt;

        return std::move(std::get<1>(state.data));
      }
      template <execution::sender S>
        requires tag_invocable<sync_wait_t, S>
      decltype(auto) operator()(S&& s) const
        noexcept(nothrow_tag_invocable<sync_wait_t, S>) {
        return tag_invoke(sync_wait_t{}, (S&&) s);
      }
    } sync_wait {};
  }
}
