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
        { execution::set_done(std::move(r)) } noexcept;
        { execution::set_error(std::move(r), (E&&) e) } noexcept;
      };

  template<class R, class... An>
    concept receiver_of =
      receiver<R> &&
      requires(remove_cvref_t<R>&& r, An&&... an) {
        execution::set_value(std::move(r), (An&&) an...);
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
      template<class S, class R>
        struct __rec {
          struct __wrap {
            __rec* __this;
            // Forward all tag_invoke calls, including the receiver ops.
            template<__same_<__wrap> Self, class... As, invocable<__member_t<Self, R>, As...> Tag>
            friend decltype(auto) tag_invoke(Tag tag, Self&& self, As&&... as)
                noexcept(is_nothrow_invocable_v<Tag, __member_t<Self, R>, As...>) {
              ((Tag&&) tag)((__member_t<Self, R>&&) self.__this->__r, (As&&) as...);
              // If we just completed the receiver contract, delete the state:
              if constexpr (__one_of<Tag, set_value_t, set_error_t, set_done_t>)
                delete self.__this;
            }
          };
          remove_cvref_t<R> __r;
          connect_result_t<S, __wrap> __state;
          __rec(S&& s, R&& r)
            : __r((R&&) r)
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
        requires invocable<F&> &&
          move_constructible<F>
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

  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.then]
  inline namespace __then {
    namespace __impl {
      template<receiver R, class F>
        struct __receiver {
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
          friend decltype(auto) tag_invoke(Tag tag, Self&& r, As&&... as)
            noexcept(is_nothrow_invocable_v<Tag, __member_t<Self, R>, As...>) {
            return ((Tag&&) tag)((R&&) r.r_, (As&&) as...);
          }
        };
      template<sender S, class F>
        struct __sender {    
          [[no_unique_address]] S s_;
          [[no_unique_address]] F f_;
    
          template<receiver R, class R2 = remove_cvref_t<R>>
            requires sender_to<S, __receiver<R2, F>>
          friend auto tag_invoke(connect_t, __sender&& self, R&& r)
            noexcept(/*todo*/ false)
            -> connect_result_t<S, __receiver<R2, F>> {
            return execution::connect(
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
        return __impl::__sender<remove_cvref_t<S>, F>{(S&&)s, (F&&)f};
      }
    } lazy_then {};
  }

  // Make the lazy_then sender typed if the input sender is also typed.
  template <typed_sender S, class F>
  struct sender_traits<__then::__impl::__sender<S, F>> {
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
