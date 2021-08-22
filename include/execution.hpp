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

  struct __void_sender {
    template<template<class...> class Tuple, template<class...> class Variant>
      using value_types = Variant<Tuple<>>;
    template<template<class...> class Variant>
      using error_types = Variant<std::exception_ptr>;
    static constexpr bool sends_done = true;
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
        return __void_sender{};
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
        set_value(std::move(r), (An&&) an...);
      };

  template<class R, class...As>
    inline constexpr bool is_nothrow_receiver_of_v =
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
      template<class R_>
        class __op {
          using R = __t<R_>;
        public:
          struct promise_type {
            template <class A>
            explicit promise_type(A&, R& r) noexcept
              : r_(r)
            {}

            __op get_return_object() noexcept {
              return __op{
                  coro::coroutine_handle<promise_type>::from_promise(
                      *this)};
            }
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
                void await_suspend(coro::coroutine_handle<promise_type>) {
                  ((Func &&) func_)();
                }
                [[noreturn]] void await_resume() noexcept {
                  terminate();
                }
              };
              return awaiter{(Func &&) func};
            }

            // Pass through receiver queries
            template<__same_<promise_type> Self, invocable<__member_t<Self, R>> CPO>
            friend auto tag_invoke(CPO cpo, Self&& self)
              noexcept(is_nothrow_invocable_v<CPO, __member_t<Self, R>>)
              -> invoke_result_t<CPO, R> {
              return ((CPO&&) cpo)(((Self&&) self).r_);
            }

            R& r_;
          };

          coro::coroutine_handle<promise_type> coro_;

          explicit __op(coro::coroutine_handle<promise_type> coro) noexcept
            : coro_(coro) {}

          __op(__op&& other) noexcept
            : coro_(exchange(other.coro_, {})) {}

          ~__op() {
            if (coro_)
              coro_.destroy();
          }

          friend void tag_invoke(start_t, __op& self) noexcept {
            self.coro_.resume();
          }
        };
    }

    inline constexpr struct __fn {
    private:
      template <__awaitable A, receiver R>
      static __impl::__op<__id_t<remove_cvref_t<R>>> __impl(A&& a, R&& r) {
        exception_ptr ex;
        try {
          // This is a bit mind bending control-flow wise.
          // We are first evaluating the co_await expression.
          // Then the result of that is passed into invoke
          // which curries a reference to the result into another
          // lambda which is then returned to 'co_yield'.
          // The 'co_yield' expression then invokes this lambda
          // after the coroutine is suspended so that it is safe
          // for the receiver to destroy the coroutine.
          auto fn = [&](auto&&... result) {
            return [&] {
              set_value((R&&) r, (__await_result_t<A>&&) result...);
            };
          };
          if constexpr (is_void_v<__await_result_t<A>>)
            co_yield (co_await (A &&) a, fn());
          else
            co_yield fn(co_await (A &&) a);
        } catch (...) {
          ex = current_exception();
        }
        co_yield [&] {
          set_error((R&&) r, (exception_ptr&&) ex);
        };
      }
    public:
      template <__awaitable A, receiver R>
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

  template <scheduler S>
    using __schedule_result_t = decltype(schedule(std::declval<S>()));

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.factories]
  inline namespace __just {
    namespace __impl {
      template <class... Ts>
        struct __sender {
          tuple<Ts...> vs_;

          template<template<class...> class Tuple,
                   template<class...> class Variant>
          using value_types = Variant<Tuple<Ts...>>;

          template<template<class...> class Variant>
          using error_types = Variant<exception_ptr>;

          static const constexpr auto sends_done = false;

          template<class R_>
          struct __op {
            using R = __t<R_>;
            std::tuple<Ts...> vs_;
            R r_;

            friend void tag_invoke(start_t, __op& op) noexcept try {
              std::apply([&op](Ts&... ts) {
                set_value((R&&) op.r_, (Ts&&) ts...);
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
      __impl::__sender<decay_t<Ts>...> operator()(Ts&&... ts) const
        noexcept((is_nothrow_constructible_v<decay_t<Ts>, Ts> &&...)) {
        return {{(Ts&&) ts...}};
      }
    } just {};
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

  namespace __pipe {
    template <class Fn, class... As>
    struct __binder {
      [[no_unique_address]] Fn fn;
      tuple<As...> as;

      template <sender S>
      friend decltype(auto) operator|(S&& s, __binder b) {
        return std::apply([&](As&... as) {
            return ((Fn&&) b.fn)(s, (As&&) as...);
          }, b.as);
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
      __pipe::__binder<lazy_then_t, F> operator()(F f) const {
        return {{}, {(F&&) f}};
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
      __pipe::__binder<then_t, F> operator()(F f) const {
        return {{}, {(F&&) f}};
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
      template <class... As>
        using __back_t = decltype((static_cast<As(*)()>(0),...)());
      template <class... As>
        requires (sizeof...(As) == 1)
        using __single_t = __back_t<As...>;
      template <class S>
        using __sync_wait_result_t =
            typename execution::sender_traits<
              remove_cvref_t<S>
            >::template value_types<tuple, __impl::__single_t>;

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
