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
    } else if constexpr (__awaitable<S>) {
      if constexpr (is_void_v<__await_result_t<S>>) {
        return __sender_of<false>{};
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

  template <bool>
    struct __variant_ {
      template <class... Ts>
        using __f = variant<Ts...>;
    };
  template <>
    struct __variant_<false> {
      struct __not_a_variant {
        __not_a_variant() = delete;
      };
      template <class...>
        using __f = __not_a_variant;
    };
  template <class... Ts>
    using __variant = __minvoke<__variant_<sizeof...(Ts) != 0>, Ts...>;

  template <typed_sender Sender,
            template <class...> class Tuple = tuple,
            template <class...> class Variant = __variant>
    using value_types_of_t =
      typename sender_traits<decay_t<Sender>>::template
        value_types<Tuple, Variant>;

  template <typed_sender Sender, template <class...> class Variant = __variant>
    using error_types_of_t =
      typename sender_traits<decay_t<Sender>>::template error_types<Variant>;

  template <typed_sender Sender, class Tuple, class Variant>
    using __value_types_of_t =
      value_types_of_t<Sender, Tuple::template __f, Variant::template __f>;

  template <typed_sender Sender, class Variant>
    using __error_types_of_t =
      error_types_of_t<Sender, Variant::template __f>;

  template<class S, class... Ts>
    concept sender_of =
      typed_sender<S> &&
      same_as<__types<Ts...>, value_types_of_t<S, __types, __single_t>>;

  template<class S>
    using __single_sender_value_t =
      value_types_of_t<S, __single_or_void_t, __single_t>;

  template<class S>
    concept __single_typed_sender =
      typed_sender<S> &&
      requires { typename __single_sender_value_t<S>; };

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

  template<__one_of<set_value_t, set_error_t, set_done_t> CPO>
    struct get_completion_scheduler_t;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.schedulers]
  template<class S>
    concept scheduler =
      copy_constructible<remove_cvref_t<S>> &&
      equality_comparable<remove_cvref_t<S>> &&
      requires(S&& s, const get_completion_scheduler_t<set_value_t> tag) {
        { schedule((S&&) s) } -> sender_of;
        { tag_invoke(tag, schedule((S&&) s)) } -> same_as<remove_cvref_t<S>>;
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

  template <class R>
    using stop_token_of_t =
      remove_cvref_t<decltype(get_stop_token(std::declval<R>()))>;

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

  inline namespace __as_awaitable {
    struct as_awaitable_t;
    extern const as_awaitable_t as_awaitable;
  }

  /////////////////////////////////////////////////////////////////////////////
  // __connect_awaitable_
  inline namespace __connect_awaitable_ {
    namespace __impl {
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

      struct __op_base {
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

      template <class R_>
        struct __promise;

      template<class R_>
        struct __op : __op_base {
          using promise_type = __promise<R_>;
          using __op_base::__op_base;
        };

      template <class R_>
        struct __promise : __promise_base {
          using R = __t<R_>;

          template <class A>
          explicit __promise(A&, R& r) noexcept
            : r_(r)
          {}

          coro::coroutine_handle<> unhandled_done() noexcept {
            set_done(std::move(r_));
            // Returning noop_coroutine here causes the __connect_awaitable
            // coroutine to never resume past the point where it co_await's
            // the awaitable.
            return coro::noop_coroutine();
          }

          __op<R_> get_return_object() noexcept {
            return __op<R_>{
              coro::coroutine_handle<__promise>::from_promise(*this)};
          }

          template <class A>
          auto await_transform(A&& a)
              noexcept(is_nothrow_invocable_v<as_awaitable_t, A, __promise&>)
              -> invoke_result_t<as_awaitable_t, A, __promise&> {
            return as_awaitable((A&&) a, *this);
          }

          // Pass through receiver queries
          template<__none_of<set_value_t, set_error_t, set_done_t> CPO, class... As>
            requires invocable<CPO, const R&, As...>
          friend auto tag_invoke(CPO cpo, const __promise& self, As&&... as)
            noexcept(is_nothrow_invocable_v<CPO, const R&, As...>)
            -> invoke_result_t<CPO, const R&, As...> {
            return ((CPO&&) cpo)(as_const(self.r_), (As&&) as...);
          }
          // Must look like a receiver or else we can't issue receiver queries
          // against the promise:
          friend void tag_invoke(std::execution::set_error_t, __promise&&, const std::exception_ptr&) noexcept;
          friend void tag_invoke(std::execution::set_done_t, __promise&&) noexcept;

          R& r_;
        };
    }

    template <class R>
      using __promise_t = __impl::__promise<__x<remove_cvref_t<R>>>;

    inline constexpr struct __fn {
     private:
      template <class R, class... Args>
        using __nothrow_ = bool_constant<nothrow_receiver_of<R, Args...>>;

      template <class R>
        using __op_t = __impl::__op<__x<remove_cvref_t<R>>>;

      template <class A, class R>
      static __op_t<R> __impl(A a, R r) {
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
              set_value((R&&) r, (add_rvalue_reference_t<__await_result_t<A, __promise_t<R>>>) as...);
            };
          };
          if constexpr (is_void_v<__await_result_t<A, __promise_t<R>>>)
            co_yield (co_await (A &&) a, fn(__nothrow_<R>{}));
          else
            co_yield fn(__nothrow_<R, __await_result_t<A, __promise_t<R>>>{}, co_await (A &&) a);
        } catch (...) {
          ex = current_exception();
        }
        co_yield [&]() noexcept -> void {
          set_error((R&&) r, (exception_ptr&&) ex);
        };
      }
     public:
      template <receiver R, __awaitable<__promise_t<R>> A>
        requires receiver_of<R, __await_result_t<A, __promise_t<R>>>
      __op_t<R> operator()(A&& a, R&& r) const {
        return __impl((A&&) a, (R&&) r);
      }
      template <receiver R, __awaitable<__promise_t<R>> A>
        requires same_as<void, __await_result_t<A, __promise_t<R>>> && receiver_of<R>
      __op_t<R> operator()(A&& a, R&& r) const {
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
      template<class A, receiver R>
        requires (!tag_invocable<connect_t, A, R>) &&
          __awaitable<A, __connect_awaitable_::__promise_t<R>>
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

  template<class S, class R>
    concept __has_nothrow_connect =
      noexcept(connect(__declval<S>(), __declval<R>()));

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.queries], sender queries
  template<__one_of<set_value_t, set_error_t, set_done_t> CPO>
    struct get_completion_scheduler_t {
      template <sender S>
        requires tag_invocable<get_completion_scheduler_t, const S&> &&
          scheduler<tag_invoke_result_t<get_completion_scheduler_t, const S&>>
      auto operator()(const S& s) const noexcept
          -> tag_invoke_result_t<get_completion_scheduler_t, const S&> {
        // NOT TO SPEC:
        static_assert(
          nothrow_tag_invocable<get_completion_scheduler_t, const S&>,
          "get_completion_scheduler<CPO> should be noexcept");
        return tag_invoke(*this, s);
      }
    };

  template<__one_of<set_value_t, set_error_t, set_done_t> CPO>
    inline constexpr get_completion_scheduler_t<CPO> get_completion_scheduler{};

  template <class S, class CPO>
    concept __has_completion_scheduler =
      invocable<get_completion_scheduler_t<CPO>, S>;

  template <class S, class CPO>
    using __completion_scheduler_for =
      invoke_result_t<get_completion_scheduler_t<CPO>, S>;

  template <class Fn, class CPO, class S, class... As>
    concept __tag_invocable_with_completion_scheduler =
      __has_completion_scheduler<S, CPO> &&
      tag_invocable<Fn, __completion_scheduler_for<S, CPO>, S, As...>;

  /////////////////////////////////////////////////////////////////////////////
  // execution::as_awaitable [execution.coro_utils.as_awaitable]
  inline namespace __as_awaitable {
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

          template <class Error>
          friend void tag_invoke(set_error_t, __rec_base&& self, Error&& err) noexcept {
            if constexpr (__decays_to<Error, exception_ptr>)
              self.result_->template emplace<2>((Error&&) err);
            else if constexpr (__decays_to<Error, error_code>)
              self.result_->template emplace<2>(make_exception_ptr(system_error(err)));
            else
              self.result_->template emplace<2>(make_exception_ptr((Error&&) err));
            self.continuation_.resume();
          }

          __expected_t<Value>* result_;
          coro::coroutine_handle<> continuation_;
        };

      template <typename P_, typename Value>
        struct __sender_awaitable_base {
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
      struct __sender_awaitable
        : __sender_awaitable_base<P_, __single_sender_value_t<__t<S_>>> {
      private:
        using Promise = __t<P_>;
        using Sender = __t<S_>;
        using Base = __sender_awaitable_base<P_, __single_sender_value_t<Sender>>;
        using __rec = typename Base::__rec;
        connect_result_t<Sender, __rec> op_;
      public:
        __sender_awaitable(Sender&& sender, coro::coroutine_handle<Promise> h)
            noexcept(__has_nothrow_connect<Sender, __rec>)
          : op_(connect((Sender&&)sender, __rec{{&this->result_, h}}))
        {}

        void await_suspend(coro::coroutine_handle<Promise>) noexcept {
          start(op_);
        }
      };
      template <class Promise, class Sender>
        using __sender_awaitable_t =
          __sender_awaitable<__x<Promise>, __x<remove_cvref_t<Sender>>>;

      template <class T, class Promise>
        concept __custom_tag_invoke_awaiter =
          tag_invocable<as_awaitable_t, T, Promise&> &&
          __awaitable<tag_invoke_result_t<as_awaitable_t, T, Promise&>, Promise>;

      template <class Sender, class Promise>
        using __rec =
          typename __sender_awaitable_base<
            __x<Promise>,
            __single_sender_value_t<Sender>
          >::__rec;

      template <class Sender, class Promise>
        concept __awaitable_sender =
          __single_typed_sender<Sender> &&
          sender_to<Sender, __rec<Sender, Promise>> &&
          requires (Promise& promise) {
            { promise.unhandled_done() } -> convertible_to<coro::coroutine_handle<>>;
          };
    } // namespace __impl

    inline constexpr struct as_awaitable_t {
      template <class T, class Promise>
      static constexpr bool __is_noexcept() noexcept {
        if constexpr (__impl::__custom_tag_invoke_awaiter<T, Promise>) {
          return nothrow_tag_invocable<as_awaitable_t, T, Promise&>;
        } else if constexpr (__awaitable<T>) {
          return true;
        } else if constexpr (__impl::__awaitable_sender<T, Promise>) {
          using S = __impl::__sender_awaitable_t<Promise, T>;
          return is_nothrow_constructible_v<S, T, coro::coroutine_handle<Promise>>;
        } else {
          return true;
        }
      }
      template <class T, class Promise>
      decltype(auto) operator()(T&& t, Promise& promise) const
          noexcept(__is_noexcept<T, Promise>()) {
        if constexpr (__impl::__custom_tag_invoke_awaiter<T, Promise>) {
          return tag_invoke(*this, (T&&) t, promise);
        } else if constexpr (__awaitable<T>) {
          return (T&&) t;
        } else if constexpr (__impl::__awaitable_sender<T, Promise>) {
          auto h = coro::coroutine_handle<Promise>::from_promise(promise);
          return __impl::__sender_awaitable_t<Promise, T>{(T&&) t, h};
        } else {
          return (T&&) t;
        }
      }
    } as_awaitable{};
  } // namespace __as_awaitable

  inline namespace __with_awaitable_senders {
    namespace __impl {
      struct __with_awaitable_senders_base {
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
    } // namespace __impl

    template <class Promise>
    struct with_awaitable_senders : __impl::__with_awaitable_senders_base {
      template <class Value>
      decltype(auto) await_transform(Value&& value) {
        static_assert(derived_from<Promise, with_awaitable_senders>);
        return as_awaitable((Value&&) value, static_cast<Promise&>(*this));
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
            template<__decays_to<__wrap> Self, class... As, invocable<__member_t<Self, R>, As...> Tag>
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

          template<receiver_of<Ts...> R>
            requires (copy_constructible<Ts> &&...)
          friend auto tag_invoke(connect_t, const __sender& s, R&& r)
            noexcept((is_nothrow_copy_constructible_v<Ts> &&...))
            -> __op<__x<remove_cvref_t<R>>> {
            return {s.vs_, (R&&) r};
          }

          template<receiver_of<Ts...> R>
          friend auto tag_invoke(connect_t, __sender&& s, R&& r)
            noexcept((is_nothrow_move_constructible_v<Ts> &&...))
            -> __op<__x<remove_cvref_t<R>>> {
            return {((__sender&&) s).vs_, (R&&) r};
          }
        };
    }

    inline constexpr struct __just_t {
      template <__movable_value... Ts>
      __impl::__sender<set_value_t, decay_t<Ts>...> operator()(Ts&&... ts) const
        noexcept((is_nothrow_constructible_v<decay_t<Ts>, Ts> &&...)) {
        return {{}, {(Ts&&) ts...}};
      }
    } just {};

    inline constexpr struct __just_error_t {
      template <__movable_value Error>
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
    template <__class D>
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

    template <__class D>
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

    inline constexpr struct __bind_back_fn {
      template <class Fn, class... As>
        __binder_back<Fn, decay_t<As>...> operator()(Fn fn, As&&... as) const
          noexcept(noexcept(__binder_back<Fn, decay_t<As>...>{
              {}, (Fn&&) fn, tuple<decay_t<As>...>{(As&&) as...}})) {
            return {{}, (Fn&&) fn, tuple<decay_t<As>...>{(As&&) as...}};
          }
    } __bind_back {};
  } // __closure
  using __closure::__binder_back;
  using __closure::__bind_back;

  namespace __tag_invoke_adaptors {
    // A derived-to-base cast that works even when the base is not
    // accessible from derived.
    template <class T, class U>
      __member_t<U, T> __c_cast(U&& u) noexcept requires __decays_to<T, T> {
        static_assert(is_reference_v<__member_t<U, T>>);
        static_assert(is_base_of_v<T, remove_reference_t<U>>);
        return (__member_t<U, T>) (U&&) u;
      }
    namespace __no {
      struct __nope {};
      struct __receiver : __nope {};
      void tag_invoke(set_error_t, __receiver, std::exception_ptr) noexcept;
      void tag_invoke(set_done_t, __receiver) noexcept;
    }
    using __not_a_receiver = __no::__receiver;

    template <class Base>
      struct __adaptor {
        struct __t {
          template <class B>
            requires constructible_from<Base, B>
          explicit __t(B&& base) : base_((B&&) base) {}

        private:
          [[no_unique_address]] Base base_;

        protected:
          Base& base() & noexcept { return base_; }
          const Base& base() const & noexcept { return base_; }
          Base&& base() && noexcept { return (Base&&) base_; }
        };
      };
    template <derived_from<__no::__nope> Base>
      struct __adaptor<Base> {
        struct __t : __no::__nope { };
      };
    template <class Base>
      using __adaptor_base = typename __adaptor<Base>::__t;

    template <class Sender, class Receiver>
      concept __has_connect =
        requires(Sender&& s, Receiver&& r) {
          ((Sender&&) s).connect((Receiver&&) r);
        };

    template <__class Derived, sender Base>
      struct __sender_adaptor {
        class __t : __adaptor_base<Base> {
          template <__decays_to<Derived> Self, receiver R>
            requires __has_connect<Self, R>
          friend auto tag_invoke(connect_t, Self&& self, R&& r)
            noexcept(noexcept(((Self&&) self).connect((R&&) r)))
            -> decltype(((Self&&) self).connect((R&&) r)) {
            return ((Self&&) self).connect((R&&) r);
          }

        protected:
          using __adaptor_base<Base>::base;

        public:
          __t() = default;
          using __adaptor_base<Base>::__adaptor_base;
        };
      };

    template <class Receiver, class... As>
      concept __has_set_value =
        requires(Receiver&& r, As&&... as) {
          ((Receiver&&) r).set_value((As&&) as...);
        };

    template <class Receiver, class A>
      concept __has_set_error =
        requires(Receiver&& r, A&& a) {
          ((Receiver&&) r).set_error((A&&) a);
        };

    template <class Receiver>
      concept __has_set_done =
        requires(Receiver&& r) {
          ((Receiver&&) r).set_done();
        };

    template <__class Derived, receiver Base>
      struct __receiver_adaptor {
        class __t : __adaptor_base<Base> {
          friend Derived;

          template <class D>
            static decltype(auto) __get_base(D&& self) noexcept {
              if constexpr (is_base_of_v<__no::__nope, Base>) {
                return ((D&&) self).base();
              } else {
                return __c_cast<__t>((D&&) self).base();
              }
            }
          template <class D>
            using __base_t = decltype(__get_base(__declval<D>()));

          template <class... As>
            requires __has_set_value<Derived, As...>
          friend void tag_invoke(set_value_t, Derived&& self, As&&... as)
            noexcept(noexcept(((Derived&&) self).set_value((As&&) as...))) {
            ((Derived&&) self).set_value((As&&) as...);
          }

          template <class D = Derived, class... As>
            requires (!__has_set_value<Derived, As...>) &&
              receiver_of<__base_t<D>, As...>
          friend void tag_invoke(set_value_t, Derived&& self, As&&... as)
            noexcept(nothrow_receiver_of<__base_t<D>, As...>) {
            set_value(__get_base((Derived&&) self), (As&&) as...);
          }

          template <class E>
            requires __has_set_error<Derived, E>
          friend void tag_invoke(set_error_t, Derived&& self, E&& e) noexcept {
            static_assert(noexcept(((Derived&&) self).set_error((E&&) e)));
            ((Derived&&) self).set_error((E&&) e);
          }

          template <class E, class D = Derived>
            requires (!__has_set_error<Derived, E>) && receiver<__base_t<D>, E>
          friend void tag_invoke(set_error_t, Derived&& self, E&& e) noexcept {
            set_error(__get_base((Derived&&) self), (E&&) e);
          }

          friend void tag_invoke(set_done_t, Derived&& self) noexcept {
            if constexpr (__has_set_done<Derived>) {
              static_assert(noexcept(((Derived&&) self).set_done()));
              ((Derived&&) self).set_done();
            } else {
              set_done(__get_base((Derived&&) self));
            }
          }

          template <__none_of<set_value_t, set_error_t, set_done_t> Tag, class D = Derived, class... As>
            requires invocable<Tag, __base_t<const D&>, As...>
          friend auto tag_invoke(Tag tag, const Derived& self, As&&... as)
            noexcept(is_nothrow_invocable_v<Tag, __base_t<const D&>, As...>)
            -> invoke_result_t<Tag, __base_t<const D&>, As...> {
            return ((Tag&&) tag)(__get_base(self), (As&&) as...);
          }

         public:
          __t() = default;
          using __adaptor_base<Base>::__adaptor_base;
        };
      };

    template <class OpState>
      concept __has_start =
        requires(OpState& op) {
          op.start();
        };

    template <__class Derived, operation_state Base>
      struct __operation_state_adaptor {
        class __t : __adaptor_base<Base> {
          friend void tag_invoke(start_t, Derived& self) noexcept
            requires __has_start<Derived> {
            static_assert(noexcept(self.start()));
            self.start();
          }

          friend void tag_invoke(start_t, Derived& self) noexcept {
            execution::start(__c_cast<__t>(self).base());
          }

          template <__none_of<start_t> Tag, class... As>
            requires invocable<Tag, const Base&, As...>
          friend auto tag_invoke(Tag tag, const Derived& self, As&&... as)
            noexcept(is_nothrow_invocable_v<Tag, const Base&, As...>)
            -> invoke_result_t<Tag, const Base&, As...> {
            return ((Tag&&) tag)(__c_cast<__t>(self).base(), (As&&) as...);
          }

        protected:
          using __adaptor_base<Base>::base;

        public:
          __t() = default;
          using __adaptor_base<Base>::__adaptor_base;
        };
      };

    template <class Sched>
      concept __has_schedule =
        requires(Sched&& sched) {
          ((Sched&&) sched).schedule();
        };

    template <__class Derived, scheduler Base>
      struct __scheduler_adaptor {
        class __t : __adaptor_base<Base> {
          template <__decays_to<Derived> Self>
            requires __has_schedule<Self>
          friend auto tag_invoke(schedule_t, Self&& self)
            noexcept(noexcept(((Self&&) self).schedule()))
            -> decltype(((Self&&) self).schedule()) {
            return ((Self&&) self).schedule();
          }

          template <__decays_to<Derived> Self>
            requires (!__has_schedule<Self>) && scheduler<__member_t<Self, Base>>
          friend auto tag_invoke(schedule_t, Self&& self)
            noexcept(noexcept(execution::schedule(__declval<__member_t<Self, Base>>())))
            -> schedule_result_t<Self> {
            return execution::schedule(__c_cast<__t>((Self&&) self).base());
          }

          template <__none_of<schedule_t> Tag, same_as<Derived> Self, class... As>
            requires invocable<Tag, const Base&, As...>
          friend auto tag_invoke(Tag tag, const Self& self, As&&... as)
            noexcept(is_nothrow_invocable_v<Tag, const Base&, As...>)
            -> invoke_result_t<Tag, const Base&, As...> {
            return ((Tag&&) tag)(__c_cast<__t>(self).base(), (As&&) as...);
          }

        protected:
          using __adaptor_base<Base>::base;

        public:
          __t() = default;
          using __adaptor_base<Base>::__adaptor_base;
        };
      };
  } // namespace __tag_invoke_adaptors

  // NOT TO SPEC
  template <__class Derived, sender Base>
    using sender_adaptor =
      typename __tag_invoke_adaptors::__sender_adaptor<Derived, Base>::__t;

  template <__class Derived, receiver Base = __tag_invoke_adaptors::__not_a_receiver>
    using receiver_adaptor =
      typename __tag_invoke_adaptors::__receiver_adaptor<Derived, Base>::__t;

  // NOT TO SPEC
  template <__class Derived, operation_state Base>
    using operation_state_adaptor =
      typename __tag_invoke_adaptors::__operation_state_adaptor<Derived, Base>::__t;

  // NOT TO SPEC
  template <__class Derived, scheduler Base>
    using scheduler_adaptor =
      typename __tag_invoke_adaptors::__scheduler_adaptor<Derived, Base>::__t;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  inline namespace __then {
    namespace __impl {
      template<class R_, class F>
        class __receiver : receiver_adaptor<__receiver<R_, F>, __t<R_>> {
          using R = __t<R_>;
          friend receiver_adaptor<__receiver, R>;
          [[no_unique_address]] F __f_;

          // Customize set_value by invoking the invocable and passing the result
          // to the base class
          template<class... As>
            requires invocable<F, As...> &&
              receiver_of<R, invoke_result_t<F, As...>>
          void set_value(As&&... as) && noexcept try {
            execution::set_value(
                ((__receiver&&) *this).base(),
                std::invoke((F&&) __f_, (As&&) as...));
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }
          // Handle the case when the invocable returns void
          template<class R2 = R, class... As>
            requires invocable<F, As...> &&
              same_as<void, invoke_result_t<F, As...>> &&
              receiver_of<R2>
          void set_value(As&&... as) && noexcept try {
            invoke((F&&) __f_, (As&&) as...);
            execution::set_value(((__receiver&&) *this).base());
          } catch(...) {
            execution::set_error(
                ((__receiver&&) *this).base(),
                current_exception());
          }

         public:
          explicit __receiver(R r, F f)
            : receiver_adaptor<__receiver, R>((R&&) r)
            , __f_((F&&) f)
          {}
        };

      template<class S_, class F>
        class __sender : sender_adaptor<__sender<S_, F>, __t<S_>> {
          using S = __t<S_>;
          friend sender_adaptor<__sender, S>;
          template <class R>
            using __rec = __receiver<__x<remove_cvref_t<R>>, F>;

          [[no_unique_address]] F f_;

          template<receiver R>
            requires sender_to<S, __rec<R>>
          auto connect(R&& r) && noexcept(__has_nothrow_connect<S, __rec<R>>)
            -> connect_result_t<S, __rec<R>> {
            return execution::connect(
                ((__sender&&) *this).base(),
                __rec<R>{(R&&) r, (F&&) f_});
          }

        public:
          explicit __sender(S s, F f)
            : sender_adaptor<__sender, S>{(S&&) s}
            , f_((F&&) f)
          {}
        };
    }

    inline constexpr struct then_t {
      template <class S, class F>
        using __sender = __impl::__sender<__x<remove_cvref_t<S>>, F>;

      template<sender S, class F>
        requires __tag_invocable_with_completion_scheduler<then_t, set_value_t, S, F>
      sender auto operator()(S&& s, F f) const
        noexcept(nothrow_tag_invocable<then_t, __completion_scheduler_for<S, set_value_t>, S, F>) {
        auto sch = get_completion_scheduler<set_value_t>(s);
        return tag_invoke(then_t{}, std::move(sch), (S&&) s, (F&&) f);
      }
      template<sender S, class F>
        requires (!__tag_invocable_with_completion_scheduler<then_t, set_value_t, S, F>) &&
          tag_invocable<then_t, S, F>
      sender auto operator()(S&& s, F f) const
        noexcept(nothrow_tag_invocable<then_t, S, F>) {
        return tag_invoke(then_t{}, (S&&) s, (F&&) f);
      }
      template<sender S, class F>
        requires (!__tag_invocable_with_completion_scheduler<then_t, set_value_t, S, F>) &&
          (!tag_invocable<then_t, S, F>) &&
          sender<__sender<S, F>>
      __sender<S, F> operator()(S&& s, F f) const {
        return __sender<S, F>{(S&&) s, (F&&) f};
      }
      template <class F>
      __binder_back<then_t, F> operator()(F f) const {
        return {{}, {}, {(F&&) f}};
      }
    } then {};
  }

  // Make the then sender typed if the input sender is also typed.
  template <class S_, class F>
    requires typed_sender<__t<S_>> &&
      requires {
        // Can the function F be called with each set of value types?
        typename __value_types_of_t<__t<S_>,
          __bind_front_q<invoke_result_t, F>,
          __q<__types>>;
      }
  struct sender_traits<__then::__impl::__sender<S_, F>> {
    using S = __t<S_>;
    template <template<class...> class Tuple, template<class...> class Variant>
      using value_types =
        __value_types_of_t<
          S,
          __bind_front_q<invoke_result_t, F>,
          __transform<
            __q<__types>,
            __replace<
              __types<void>,
              __types<>,
              __transform<__uncurry<__q<Tuple>>, __q<Variant>>>>>;

    template <template<class...> class Variant>
      using error_types =
        __mapply<
          __q<Variant>,
          __minvoke2<
            __push_back_unique,
            error_types_of_t<S, __types>,
            exception_ptr>>;

    static constexpr bool sends_done = sender_traits<S>::sends_done;
  };

  /////////////////////////////////////////////////////////////////////////////
  // run_loop
  inline namespace __loop {
    class run_loop;

    namespace __impl {
      struct __task {
        virtual void __execute() noexcept = 0;
        __task* __next_ = nullptr;
      };

      template <typename Receiver_>
        class __operation final : __task {
          using Receiver = __t<Receiver_>;
          using stop_token_type = stop_token_of_t<Receiver&>;

          friend void tag_invoke(start_t, __operation& op) noexcept {
            op.__start_();
          }

          void __execute() noexcept override try {
            if (get_stop_token(__receiver_).stop_requested()) {
              set_done((Receiver&&) __receiver_);
            } else {
              set_value((Receiver&&) __receiver_);
            }
          } catch(...) {
            set_error((Receiver&&) __receiver_, current_exception());
          }

          void __start_() noexcept;

          [[no_unique_address]] Receiver __receiver_;
          run_loop* const __loop_;

        public:
          template <typename Receiver2>
          explicit __operation(Receiver2&& receiver, run_loop* loop)
            : __receiver_((Receiver2 &&) receiver)
            , __loop_(loop) {}
        };
    } // namespace __impl

    class run_loop {
      template <class>
        friend struct __impl::__operation;
     public:
      class __scheduler {
        struct __schedule_task {
          template <
              template <typename...> class Tuple,
              template <typename...> class Variant>
          using value_types = Variant<Tuple<>>;

          template <template <typename...> class Variant>
          using error_types = Variant<std::exception_ptr>;

          static constexpr bool sends_done = true;

        private:
          friend __scheduler;

          template <typename Receiver>
          friend __impl::__operation<__x<Receiver>>
          tag_invoke(connect_t, const __schedule_task& self, Receiver&& receiver) {
            return __impl::__operation<__x<Receiver>>{(Receiver &&) receiver, self.__loop_};
          }

          template <class CPO>
          friend __scheduler
          tag_invoke(get_completion_scheduler_t<CPO>, const __schedule_task& self) noexcept {
            return __scheduler{self.__loop_};
          }

          explicit __schedule_task(run_loop* loop) noexcept
            : __loop_(loop)
          {}

          run_loop* const __loop_;
        };

        friend run_loop;

        explicit __scheduler(run_loop* loop) noexcept : __loop_(loop) {}

       public:
        friend __schedule_task tag_invoke(schedule_t, const __scheduler& self) noexcept {
          return __schedule_task{self.__loop_};
        }

        bool operator==(const __scheduler&) const noexcept = default;

       private:
        run_loop* __loop_;
      };

      __scheduler get_scheduler() {
        return __scheduler{this};
      }

      void run();

      void finish();

     private:
      void push_back(__impl::__task* task);
      __impl::__task* pop_front();

      mutex __mutex_;
      condition_variable __cv_;
      __impl::__task* __head_ = nullptr;
      __impl::__task* __tail_ = nullptr;
      bool __stop_ = false;
    };

    namespace __impl {
      template <typename Receiver_>
      inline void __operation<Receiver_>::__start_() noexcept try {
        __loop_->push_back(this);
      } catch(...) {
        set_error((Receiver&&) __receiver_, current_exception());
      }
    }

    inline void run_loop::run() {
      while (auto* task = pop_front()) {
        task->__execute();
      }
    }

    inline void run_loop::finish() {
      unique_lock lock{__mutex_};
      __stop_ = true;
      __cv_.notify_all();
    }

    inline void run_loop::push_back(__impl::__task* task) {
      unique_lock lock{__mutex_};
      if (__head_ == nullptr) {
        __head_ = task;
      } else {
        __tail_->__next_ = task;
      }
      __tail_ = task;
      task->__next_ = nullptr;
      __cv_.notify_one();
    }

    inline __impl::__task* run_loop::pop_front() {
      unique_lock __lock{__mutex_};
      while (__head_ == nullptr) {
        if (__stop_)
          return nullptr;
        __cv_.wait(__lock);
      }
      auto* task = __head_;
      __head_ = task->__next_;
      if (__head_ == nullptr)
        __tail_ = nullptr;
      return task;
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
        template <receiver Receiver>
          struct __f {
            std::any __tuple_;
            void (*__complete_)(Receiver& rcvr, any& tup) noexcept;

            template <class Tuple, class... Args>
              void emplace(Args&&... args) {
                __tuple_.emplace<Tuple>((Args&&) args...);
                __complete_ = [](Receiver& rcvr, any& tup) noexcept {
                  try {
                    std::apply([&](auto tag, auto&&... args) -> void {
                      tag((Receiver&&) rcvr, (decltype(args)&&) args...);
                    }, any_cast<Tuple>(tup));
                  } catch(...) {
                    set_error((Receiver&&) rcvr, current_exception());
                  }
                };
              }

            void complete(Receiver& rcvr) noexcept {
              __complete_(rcvr, __tuple_);
            }
          };
        };

      template <sender Sender>
        struct __completion_storage : __completion_storage_non_typed {};

      // This specialization is for typed senders, where the completion
      // information can be stored in situ within a variant in the operation
      // state 
      template <typed_sender Sender>
        struct __completion_storage<Sender> {
          // Compute a variant type that is capable of storing the results of the
          // input sender when it completes. The variant has type:
          //   variant<
          //     tuple<set_done_t>,
          //     tuple<set_value_t, decay_t<Values1>...>,
          //     tuple<set_value_t, decay_t<Values2>...>,
          //        ...
          //     tuple<set_error_t, decay_t<Error1>>,
          //     tuple<set_error_t, decay_t<Error2>>,
          //        ...
          //   >
          template <class... Ts>
            using __bind_tuples =
              __bind_front_q<variant, tuple<set_done_t>, Ts...>;

          using __bound_values_t = 
            __value_types_of_t<
              Sender,
              __transform<__q1<decay_t>, __bind_front_q<tuple, set_value_t>>,
              __q<__bind_tuples>>;

          using __variant_t =
            __error_types_of_t<
              Sender,
              __transform<
                __transform<__q1<decay_t>, __bind_front_q<tuple, set_error_t>>,
                __bound_values_t>>;

          template <receiver Receiver>
            struct __f : private __variant_t {
              __f() = default;
              using __variant_t::emplace;

              void complete(Receiver& rcvr) noexcept try {
                std::visit([&](auto&& tup) -> void {
                  std::apply([&](auto tag, auto&&... args) -> void {
                    tag((Receiver&&) rcvr, (decltype(args)&&) args...);
                  }, (decltype(tup)&&) tup);
                }, (__variant_t&&) *this);
              } catch(...) {
                set_error((Receiver&&) rcvr, current_exception());
              }
            };
        };

      template <class Scheduler_, class Sender_>
        struct __sender : sender_base {
          using Scheduler = __t<Scheduler_>;
          using Sender = __t<Sender_>;
          Scheduler __sch_;
          Sender __snd_;

          template <class CvrefReceiver_>
            struct __op1 {
              // Bit of a hack here: the cvref qualification of the Sender
              // is encoded in the type of the Receiver:
              using CvrefReceiver = __t<CvrefReceiver_>;
              using CvrefSender = __member_t<CvrefReceiver, Sender>;
              using Receiver = decay_t<CvrefReceiver>;

              // This receiver is to be completed on the execution context
              // associated with the scheduler. When the source sender
              // completes, the completion information is saved off in the
              // operation state so that when this receiver completes, it can
              // read the completion out of the operation state and forward it
              // to the output receiver after transitioning to the scheduler's
              // context.
              struct __receiver2 {
                __op1* __op_;

                // If the work is successfully scheduled on the new execution
                // context and is ready to run, forward the completion signal in
                // the operation state
                friend void tag_invoke(set_value_t, __receiver2&& self) noexcept {
                  self.__op_->__data_.complete(self.__op_->__rec_);
                }

                template <__one_of<set_error_t, set_done_t> Tag, class... Args>
                  requires invocable<Tag, Receiver, Args...>
                friend void tag_invoke(Tag, __receiver2&& self, Args&&... args) noexcept {
                  Tag{}((Receiver&&) self.__op_->__rec_, (Args&&) args...);
                }

                template <__none_of<set_value_t, set_error_t, set_done_t> Tag, class... Args>
                friend auto tag_invoke(Tag tag, const __receiver2& self, Args&&... args)
                  noexcept(is_nothrow_invocable_v<Tag, const Receiver&, Args...>)
                  -> invoke_result_t<Tag, const Receiver&, Args...> {
                  return ((Tag&&) tag)(as_const(self.__op_->__rec_), (Args&&) args...);
                }
              };

              // This receiver is connected to the input sender. When that
              // sender completes (on whatever context it completes on), save
              // the completion information into the operation state. Then,
              // schedule a second operation to complete on the execution
              // context of the scheduler. That second receiver will read the
              // completion information out of the operation state and propagate
              // it to the output receiver from within the desired context.
              struct __receiver1 {
                __op1* __op_;

                template <__one_of<set_value_t, set_error_t, set_done_t> Tag, class... Args>
                  requires invocable<Tag, Receiver, Args...>
                friend void tag_invoke(Tag, __receiver1&& self, Args&&... args) noexcept try {
                  // Write the tag and the args into the operation state so that
                  // we can forward the completion from within the scheduler's
                  // execution context.
                  self.__op_->__data_.template emplace<tuple<Tag, decay_t<Args>...>>(Tag{}, (Args&&) args...);
                  // Schedule the completion to happen on the scheduler's
                  // execution context.
                  self.__op_->__state2_.emplace(
                      __conv{[op = self.__op_] {
                        return connect(schedule(op->__sch_), __receiver2{op});
                      }});
                  // Enqueue the scheduled operation:
                  start(*self.__op_->__state2_);
                } catch(...) {
                  set_error((Receiver&&) self.__op_->__rec_, current_exception());
                }

                template <__none_of<set_value_t, set_error_t, set_done_t> Tag, class... Args>
                friend auto tag_invoke(Tag tag, const __receiver1& self, Args&&... args)
                  noexcept(is_nothrow_invocable_v<Tag, const Receiver&, Args...>)
                  -> invoke_result_t<Tag, const Receiver&, Args...> {
                  return ((Tag&&) tag)(as_const(self.__op_->__rec_), (Args&&) args...);
                }
              };

              Scheduler __sch_;
              Receiver __rec_;
              __minvoke<__completion_storage<Sender>, Receiver> __data_;
              connect_result_t<CvrefSender, __receiver1> __state1_;
              optional<connect_result_t<schedule_result_t<Scheduler>, __receiver2>> __state2_;

              __op1(Scheduler sch, CvrefSender&& snd, __decays_to<Receiver> auto&& rec)
                : __sch_(sch)
                , __rec_((decltype(rec)&&) rec)
                , __state1_(connect((CvrefSender&&) snd, __receiver1{this})) {}

              friend void tag_invoke(start_t, __op1& op) noexcept {
                start(op.__state1_);
              }
            };

          template <__decays_to<__sender> Self, receiver R>
            requires sender_to<__member_t<Self, Sender>, R>
          friend auto tag_invoke(connect_t, Self&& self, R&& rec)
              -> __op1<__x<__member_t<Self, decay_t<R>>>> {
            return {self.__sch_, ((Self&&) self).__snd_, (R&&) rec};
          }

          template <__one_of<set_value_t, set_error_t, set_done_t> Tag>
          friend Scheduler tag_invoke(get_completion_scheduler_t<Tag>, const __sender& self) noexcept {
            return self.__sch_;
          }
        };
    } // namespace __impl

    inline constexpr struct schedule_from_t {
      // NOT TO SPEC: permit non-typed senders:
      template <scheduler Sch, sender S>
        requires tag_invocable<schedule_from_t, Sch, S>
      auto operator()(Sch&& sch, S&& s) const
        noexcept(nothrow_tag_invocable<schedule_from_t, Sch, S>)
        -> tag_invoke_result_t<schedule_from_t, Sch, S> {
        return tag_invoke(*this, (Sch&&) sch, (S&&) s);
      }

      // NOT TO SPEC: permit non-typed senders:
      template <scheduler Sch, sender S>
      auto operator()(Sch&& sch, S&& s) const
        -> __impl::__sender<__x<decay_t<Sch>>, __x<decay_t<S>>> {
        return {{}, (Sch&&) sch, (S&&) s};
      }
    } schedule_from {};
  } // namespace __schedule_from

  template <class Scheduler_, class Sender_>
    struct sender_traits<__schedule_from::__impl::__sender<Scheduler_, Sender_>>
      : sender_traits<__t<Sender_>> { };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.transfer]
  inline namespace __transfer {
    inline constexpr struct transfer_t {
      template <sender S, scheduler Sch>
        requires __tag_invocable_with_completion_scheduler<transfer_t, set_value_t, S, Sch>
      tag_invoke_result_t<transfer_t, __completion_scheduler_for<S, set_value_t>, S, Sch>
      operator()(S&& s, Sch&& sch) const
        noexcept(nothrow_tag_invocable<transfer_t, __completion_scheduler_for<S, set_value_t>, S, Sch>) {
        auto csch = get_completion_scheduler<set_value_t>(s);
        return tag_invoke(transfer_t{}, std::move(csch), (S&&) s, (Sch&&) sch);
      }
      template <sender S, scheduler Sch>
        requires (!__tag_invocable_with_completion_scheduler<transfer_t, set_value_t, S, Sch>) &&
          tag_invocable<transfer_t, S, Sch>
      tag_invoke_result_t<transfer_t, S, Sch>
      operator()(S&& s, Sch&& sch) const noexcept(nothrow_tag_invocable<transfer_t, S, Sch>) {
        return tag_invoke(transfer_t{}, (S&&) s, (Sch&&) sch);
      }
      // NOT TO SPEC: permit non-typed senders:
      template <sender S, scheduler Sch>
        requires (!__tag_invocable_with_completion_scheduler<transfer_t, set_value_t, S, Sch>) &&
          (!tag_invocable<transfer_t, S, Sch>)
      auto operator()(S&& s, Sch&& sch) const {
        return schedule_from((Sch&&) sch, (S&&) s);
      }
      template <scheduler Sch>
      __binder_back<transfer_t, decay_t<Sch>> operator()(Sch&& sch) const {
        return {{}, {}, {(Sch&&) sch}};
      }
    } transfer {};
  } // namespace __transfer

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.on]
  inline namespace __on {
    namespace __impl {
      template <class SchedulerId, class SenderId>
        struct __sender : sender_base {
          using Scheduler = __t<SchedulerId>;
          using Sender = __t<SenderId>;

          Scheduler __scheduler_;
          Sender __sender_;

          template <class ReceiverId>
            struct __operation;

          template <class ReceiverId>
            struct __receiver_ref
              : receiver_adaptor<__receiver_ref<ReceiverId>> {
              using Receiver = __t<ReceiverId>;
              __operation<ReceiverId>* __op_state_;
              Receiver&& base() && noexcept {
                return (Receiver&&) __op_state_->__receiver_;
              }
              const Receiver& base() const & noexcept {
                return __op_state_->__receiver_;
              }
              friend Scheduler tag_invoke(get_scheduler_t,
                                          const __receiver_ref& self) noexcept {
                return self.__op_state_->__scheduler_;
              }
            };

          template <class ReceiverId>
            struct __receiver : receiver_adaptor<__receiver<ReceiverId>> {
              using Receiver = __t<ReceiverId>;
              __operation<ReceiverId>* __op_state_;
              Receiver&& base() && noexcept {
                return (Receiver&&) __op_state_->__receiver_;
              }
              const Receiver& base() const & noexcept {
                return __op_state_->__receiver_;
              }

              void set_value() && noexcept {
                // cache this locally since *this is going bye-bye.
                auto* op_state = __op_state_;
                try {
                  // This line will invalidate *this:
                  start(op_state->__data_.template emplace<1>(__conv{
                    [op_state] {
                      return connect((Sender&&) op_state->__sender_,
                                     __receiver_ref<ReceiverId>{{}, op_state});
                    }
                  }));
                } catch(...) {
                  set_error((Receiver&&) op_state->__receiver_,
                            current_exception());
                }
              }
            };

          template <class ReceiverId>
            struct __operation {
              using Receiver = __t<ReceiverId>;

              friend void tag_invoke(start_t, __operation& self) noexcept {
                start(std::get<0>(self.__data_));
              }

              template <class Sender2, class Receiver2>
              __operation(Scheduler sched, Sender2&& sndr, Receiver2&& recvr)
                : __data_{in_place_index<0>, __conv{[&, this]{
                    return connect(schedule(sched),
                                   __receiver<ReceiverId>{{}, this});
                  }}}
                , __scheduler_((Scheduler&&) sched)
                , __sender_((Sender2&&) sndr)
                , __receiver_((Receiver2&&) recvr) {}

              std::variant<
                  connect_result_t<schedule_result_t<Scheduler>,
                                   __receiver<ReceiverId>>,
                  connect_result_t<Sender,
                                   __receiver_ref<ReceiverId>>> __data_;
              Scheduler __scheduler_;
              Sender __sender_;
              Receiver __receiver_;
            };

          template <__decays_to<__sender> Self, receiver Receiver>
            requires constructible_from<Sender, __member_t<Self, Sender>> &&
              sender_to<Sender, __receiver_ref<__x<decay_t<Receiver>>>> &&
              sender_to<schedule_result_t<Scheduler>,
                        __receiver<__x<decay_t<Receiver>>>>
          friend auto tag_invoke(connect_t, Self&& self, Receiver&& recvr)
            -> __operation<__x<decay_t<Receiver>>> {
            return {((Self&&) self).__scheduler_,
                    ((Self&&) self).__sender_,
                    (Receiver&&) recvr};
          }
        };
    } // namespace __impl

    inline constexpr struct on_t {
      template <scheduler Scheduler, sender Sender>
        requires tag_invocable<on_t, Scheduler, Sender>
      auto operator()(Scheduler&& sched, Sender&& sndr) const 
        noexcept(nothrow_tag_invocable<on_t, Scheduler, Sender>)
        -> tag_invoke_result_t<on_t, Scheduler, Sender> {
        return tag_invoke(*this, (Scheduler&&) sched, (Sender&&) sndr);
      }

      template <scheduler Scheduler, sender Sender>
      auto operator()(Scheduler&& sched, Sender&& sndr) const 
        -> __impl::__sender<__x<decay_t<Scheduler>>,
                            __x<decay_t<Sender>>> {
        return {{}, (Scheduler&&) sched, (Sender&&) sndr};
      }
    } on {};
  } // namespace __on

  template <class SchedulerId, class SenderId>
    requires typed_sender<__t<SenderId>>
  struct sender_traits<__on::__impl::__sender<SchedulerId, SenderId>>
    : sender_traits<__t<SenderId>> {};

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  inline namespace __transfer_just {
    inline constexpr struct transfer_just_t {
      template <scheduler Scheduler, __movable_value... Values>
        requires tag_invocable<transfer_just_t, Scheduler, Values...> &&
          typed_sender<tag_invoke_result_t<transfer_just_t, Scheduler, Values...>>
      auto operator()(Scheduler&& sched, Values&&... values) const
        noexcept(nothrow_tag_invocable<transfer_just_t, Scheduler, Values...>)
        -> tag_invoke_result_t<transfer_just_t, Scheduler, Values...> {
        return tag_invoke(*this, (Scheduler&&) sched, (Values&&) values...);
      }
      template <scheduler Scheduler, __movable_value... Values>
      auto operator()(Scheduler&& sched, Values&&... values) const
        -> decltype(transfer(just((Values&&) values...), (Scheduler&&) sched)) {
        return transfer(just((Values&&) values...), (Scheduler&&) sched);
      }
    } transfer_just {};
  } // namespace __transfer_just

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.when_all]
  inline namespace __when_all {
    namespace __impl {
      enum __state_t { __started, __error, __done };

      struct __on_stop_requested {
        in_place_stop_source& __stop_source_;
        void operator()() noexcept {
          __stop_source_.request_stop();
        }
      };

      template <class... Ts>
          requires (sizeof...(Ts) == 0)
        using __none_t = void;

      template <class Sender>
        concept __zero_alternatives =
          requires {
            typename value_types_of_t<Sender, __types, __none_t>;
          };

      template <class... SenderIds>
        struct __sender {
         private:
          template <template <class...> class Tuple,
                    template <class...> class Variant>
            struct __value_types {
              using type = Variant<__minvoke<__concat<__q<Tuple>>,
                value_types_of_t<__t<SenderIds>, __types, __single_t>...>>;
            };

          template <template <class...> class Tuple,
                    template <class...> class Variant>
              requires (__zero_alternatives<__t<SenderIds>> ||...)
            struct __value_types<Tuple, Variant> {
              using type = Variant<>;
            };

         public:
          template <template <class...> class Tuple,
                    template <class...> class Variant>
            using value_types = __t<__value_types<Tuple, Variant>>;

          template <template <class...> class Variant>
            using error_types =
              __minvoke<
                __concat<__unique<__q<Variant>>>,
                __types<exception_ptr>,
                error_types_of_t<__t<SenderIds>, __types>...>;

          static constexpr bool sends_done = true;

          template <class... Sndrs>
            explicit __sender(Sndrs&&... sndrs)
              : __sndrs_((Sndrs&&) sndrs...)
            {}

         private:
          template <class CvrefReceiverId>
            struct __operation {
              using WhenAll = __member_t<CvrefReceiverId, __sender>;
              using Receiver = __t<decay_t<CvrefReceiverId>>;
              template <size_t Index>
                struct __receiver : receiver_adaptor<__receiver<Index>> {
                  Receiver&& base() && noexcept {
                    return (Receiver&&) __op_->__recvr_;
                  }
                  const Receiver& base() const & noexcept {
                    return __op_->__recvr_;
                  }
                  template <class Error>
                    void __set_error(Error&& err, __state_t expected) noexcept {
                      // TODO: What memory orderings are actually needed here?
                      if (__op_->__state_.compare_exchange_strong(expected, __error)) {
                        __op_->__stop_source_.request_stop();
                        // We won the race, free to write the error into the operation
                        // state without worry.
                        try {
                          __op_->__errors_.template emplace<decay_t<Error>>((Error&&) err);
                        } catch(...) {
                          __op_->__errors_.template emplace<exception_ptr>(current_exception());
                        }
                      }
                      __op_->__arrive();
                    }
                  template <class... Values>
                    void set_value(Values&&... vals) && noexcept {
                      // We only need to bother recording the completion values
                      // if we're not already in the "error" or "done" state.
                      if (__op_->__state_ == __started) {
                        try {
                          std::get<Index>(__op_->__values_).emplace(
                              (Values&&) vals...);
                        } catch(...) {
                          __set_error(current_exception(), __started);
                        }
                      }
                      __op_->__arrive();
                    }
                  template <class Error>
                      requires receiver<Receiver, Error>
                    void set_error(Error&& err) && noexcept {
                      __set_error((Error&&) err, __started);
                    }
                  void set_done() && noexcept {
                    __state_t expected = __started;
                    // Transition to the "done" state if and only if we're in the
                    // "started" state. (If this fails, it's because we're in an
                    // error state, which trumps cancellation.)
                    if (__op_->__state_.compare_exchange_strong(expected, __done)) {
                      __op_->__stop_source_.request_stop();
                    }
                    __op_->__arrive();
                  }
                  friend in_place_stop_token tag_invoke(
                      get_stop_token_t, const __receiver& self) noexcept {
                    return self.__op_->__stop_source_.get_token();
                  }
                  __operation* __op_;
                };

              template <class Sender, size_t Index>
                using __child_opstate =
                  connect_result_t<__member_t<WhenAll, Sender>, __receiver<Index>>;

              using Indices = index_sequence_for<SenderIds...>;

              template <size_t... Is>
                static auto connect_children(
                    __operation* self, WhenAll&& when_all, index_sequence<Is...>)
                    -> tuple<__child_opstate<__t<SenderIds>, Is>...> {
                  return tuple<__child_opstate<__t<SenderIds>, Is>...>{
                    __conv{[&when_all, self]() {
                      return execution::connect(
                          std::get<Is>(((WhenAll&&) when_all).__sndrs_),
                          __receiver<Is>{{}, self});
                    }}...
                  };
                }

              using child_optstates_tuple_t =
                  decltype(connect_children(nullptr, __declval<WhenAll>(), Indices{}));

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
                  // All child operations completed successfully:
                  std::apply(
                    [this](auto&... opt_values) -> void {
                      std::apply(
                        [this](auto&... all_values) -> void {
                          try {
                            execution::set_value(
                                (Receiver&&) __recvr_, std::move(all_values)...);
                          } catch(...) {
                            execution::set_error(
                                (Receiver&&) __recvr_, current_exception());
                          }
                        },
                        std::tuple_cat(
                          std::apply(
                            [](auto&... vals) { return std::tie(vals...); },
                            *opt_values
                          )...
                        )
                      );
                    },
                    __values_
                  );
                  break;
                case __error:
                  std::visit([this](auto& error) noexcept {
                    execution::set_error((Receiver&&) __recvr_, std::move(error));
                  }, __errors_);
                  break;
                case __done:
                  execution::set_done((Receiver&&) __recvr_);
                  break;
                default:
                  ;
                }
              }

              __operation(WhenAll&& when_all, Receiver recvr)
                : __child_states_{connect_children(this, (WhenAll&&) when_all, Indices{})}
                , __recvr_((Receiver&&) recvr)
              {}

              friend void tag_invoke(start_t, __operation& self) noexcept {
                // register stop callback:
                self.__on_stop_.emplace(
                    get_stop_token(self.__recvr_),
                    __on_stop_requested{self.__stop_source_});
                if (self.__stop_source_.stop_requested()) {
                  // Stop has already been requested. Don't bother starting
                  // the child operations.
                  execution::set_done((Receiver&&) self.__recvr_);
                } else {
                  apply([](auto&&... child_ops) noexcept -> void {
                    (execution::start(child_ops), ...);
                  }, self.__child_states_);
                }
              }

              child_optstates_tuple_t __child_states_;
              Receiver __recvr_;
              atomic<size_t> __count_{sizeof...(SenderIds)};
              // Could be non-atomic here and atomic_ref everywhere except __completion_fn
              atomic<__state_t> __state_{__started};
              error_types_of_t<__sender, variant> __errors_{};
              tuple<value_types_of_t<__t<SenderIds>, tuple, optional>...> __values_{};
              in_place_stop_source __stop_source_{};
              optional<typename stop_token_of_t<Receiver&>::template
                  callback_type<__on_stop_requested>> __on_stop_{};
            };

          template <class Receiver>
            struct __receiver_of {
              template <class... Values>
                using __f = bool_constant<receiver_of<Receiver, Values...>>;
            };
          template <class Receiver>
            using __can_connect_to_t =
              __value_types_of_t<
                __sender,
                __receiver_of<Receiver>,
                __q<__single_or_void_t>>;

          template <__decays_to<__sender> Self, receiver Receiver>
              requires __is_true<__can_connect_to_t<Receiver>>
            friend auto tag_invoke(connect_t, Self&& self, Receiver&& recvr)
              -> __operation<__x<decay_t<Receiver>>> {
              return {(Self&&) self, (Receiver&&) recvr};
            }

          tuple<__t<SenderIds>...> __sndrs_;
        };

      template <class Sender>
        concept __zero_or_one_alternative =
          requires {
            typename value_types_of_t<Sender, __types, __single_or_void_t>;
          };
    } // namespce __impl

    inline constexpr struct when_all_t {
      template <typed_sender... Senders>
        requires tag_invocable<when_all_t, Senders...> &&
          sender<tag_invoke_result_t<when_all_t, Senders...>>
      auto operator()(Senders&&... sndrs) const
        noexcept(nothrow_tag_invocable<when_all_t, Senders...>)
        -> tag_invoke_result_t<when_all_t, Senders...> {
        return tag_invoke(*this, (Senders&&) sndrs...);
      }

      template <typed_sender... Senders>
        requires (__impl::__zero_or_one_alternative<Senders> &&...)
      auto operator()(Senders&&... sndrs) const
        noexcept(nothrow_tag_invocable<when_all_t, Senders...>)
        -> __impl::__sender<__x<decay_t<Senders>>...> {
        return __impl::__sender<__x<decay_t<Senders>>...>{
            (Senders&&) sndrs...};
      }
    } when_all {};
  } // namespace __when_all
} // namespace std::execution

namespace std::this_thread {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumers.sync_wait]
  inline namespace __sync_wait {
    namespace __impl {
      // What should sync_wait(just_done()) return?
      template <class S>
        using __sync_wait_result_t =
            execution::value_types_of_t<S, tuple, __single_t>;

      template <class Tuple>
        struct __state {
          variant<monostate, Tuple, exception_ptr, execution::set_done_t> data;
          struct __receiver {
            __state* __state_;
            execution::run_loop* __loop_;
            template <class... As>
              requires constructible_from<Tuple, As...>
            friend void tag_invoke(execution::set_value_t, __receiver&& r, As&&... as) {
              r.__state_->data.template emplace<1>((As&&) as...);
              r.__loop_->finish();
            }
            friend void tag_invoke(execution::set_error_t, __receiver&& r, exception_ptr e) noexcept {
              r.__state_->data.template emplace<2>((exception_ptr&&) e);
              r.__loop_->finish();
            }
            friend void tag_invoke(execution::set_done_t d, __receiver&& r) noexcept {
              r.__state_->data.template emplace<3>(d);
              r.__loop_->finish();
            }
            friend execution::run_loop::__scheduler
            tag_invoke(execution::get_scheduler_t, const __receiver& r) noexcept {
              return r.__loop_.get_scheduler();
            }
          };
        };
    } // namespace __impl

    inline constexpr struct sync_wait_t {
      // TODO: constrain on return type
      template <execution::sender S> // NOT TO SPEC
        requires execution::__tag_invocable_with_completion_scheduler<sync_wait_t, execution::set_value_t, S>
      tag_invoke_result_t<sync_wait_t, execution::__completion_scheduler_for<S, execution::set_value_t>, S>
      operator()(S&& s) const
        noexcept(nothrow_tag_invocable<sync_wait_t, execution::__completion_scheduler_for<S, execution::set_value_t>, S>) {
        auto sch = execution::get_completion_scheduler<execution::set_value_t>(s);
        return tag_invoke(sync_wait_t{}, std::move(sch), (S&&) s);
      }
      // TODO: constrain on return type
      template <execution::sender S> // NOT TO SPEC
        requires (!execution::__tag_invocable_with_completion_scheduler<sync_wait_t, execution::set_value_t, S>) &&
          tag_invocable<sync_wait_t, S>
      tag_invoke_result_t<sync_wait_t, S>
      operator()(S&& s) const noexcept(nothrow_tag_invocable<sync_wait_t, S>) {
        return tag_invoke(sync_wait_t{}, (S&&) s);
      }
      template <execution::typed_sender S>
        requires (!execution::__tag_invocable_with_completion_scheduler<sync_wait_t, execution::set_value_t, S>) &&
          (!tag_invocable<sync_wait_t, S>)
      optional<__impl::__sync_wait_result_t<S>> operator()(S&& s) const {
        using state_t = __impl::__state<__impl::__sync_wait_result_t<S>>;
        state_t state {};
        execution::run_loop loop;

        // Launch the sender with a continuation that will fill in a variant
        // and notify a condition variable.
        auto op = execution::connect((S&&) s, typename state_t::__receiver{&state, &loop});
        execution::start(op);

        // Wait for the variant to be filled in.
        loop.run();

        if (state.data.index() == 2)
          rethrow_exception(std::get<2>(state.data));

        if (state.data.index() == 3)
          return nullopt;

        return std::move(std::get<1>(state.data));
      }
    } sync_wait {};
  }
}
