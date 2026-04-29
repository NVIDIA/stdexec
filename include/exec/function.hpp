/* Copyright (c) 2026 Ian Petersen
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "../stdexec/__detail/__completion_signatures.hpp"
#include "../stdexec/__detail/__concepts.hpp"
#include "../stdexec/__detail/__env.hpp"
#include "../stdexec/__detail/__read_env.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__scope.hpp"
#include "../stdexec/__detail/__sender_concepts.hpp"

// TODO: split this header into pieces
#include "any_sender_of.hpp"

#include <exception>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

// This file defines function<ReturnType(Arguments...)>, which is a
// type-erased sender that can complete with
//  - set_value(ReturnType&&)
//  - set_error(std::exception_ptr)
//  - set_stopped()
//
// The type-erased operation state is allocated in connect; to accomplish
// this deferred allocation, the sender holds a tuple of arguments that
// are passed into a sender-factory in connect, which is why the template
// type parameter is a function type rather than just a return type.
//
// The intended use case is an ABI-stable API boundary, assuming that a
// std::tuple<Arguments...> qualifies as "ABI-stable". The hope is that
// this is a "better task" in that it represents an async function from
// arguments to value, just like a task coroutine, but, by deferring the
// allocation to connect, we can use receiver environment queries to pick
// the frame allocator from the environment without relying on TLS.
namespace experimental::execution
{
  // A forwarding query for a "frame allocator", to be used for dynamically allocating
  // the operation states of senders type-erased by exec::function.
  struct get_frame_allocator_t : STDEXEC::__query<get_frame_allocator_t>
  {
    using STDEXEC::__query<get_frame_allocator_t>::operator();

    constexpr auto operator()() const noexcept
    {
      return STDEXEC::read_env(get_frame_allocator_t{});
    }

    template <class Env>
    static constexpr void __validate() noexcept
    {
      static_assert(STDEXEC::__nothrow_callable<get_frame_allocator_t, Env const &>);
      using __alloc_t = STDEXEC::__call_result_t<get_frame_allocator_t, Env const &>;
      static_assert(STDEXEC::__simple_allocator<STDEXEC::__decay_t<__alloc_t>>);
    }

    static consteval auto query(STDEXEC::forwarding_query_t) noexcept -> bool
    {
      return true;
    }
  };

  inline constexpr get_frame_allocator_t get_frame_allocator{};

#if 0
  namespace _qry_detail
  {
    template <class Sig>
    inline constexpr bool is_query_function_v = false;

    template <class Return, class Query, class... Args>
    inline constexpr bool is_query_function_v<Return(Query, Args...)> = true;

    template <class Return, class Query, class... Args>
    inline constexpr bool is_query_function_v<Return(Query, Args...) noexcept> = true;
  }  // namespace _qry_detail
#endif

  namespace _func
  {
    using namespace STDEXEC;

    // the type-erased operation state type that supports starting and destruction
    struct _base_op
    {
      constexpr _base_op() = default;

      _base_op(_base_op &&) = delete;

      constexpr virtual ~_base_op() = default;

      constexpr virtual void start() & noexcept = 0;
    };

    // the operation state resulting from connecting a sender being erased by a function<...>
    // with an _any::_any_receiver_ref<...>; inherits from _base_op, and provides a
    // class-specific override of operator delete that invokes the allocator deallocation protocol
    template <class Sender, class Receiver, class Allocator>
    struct _derived_op : _base_op
    {
      constexpr explicit _derived_op(Sender &&sndr, Receiver rcvr, Allocator const &alloc)
        noexcept(std::is_nothrow_invocable_v<connect_t, Sender, Receiver>)
        : op_(connect(std::forward<Sender>(sndr), std::move(rcvr)))
        , alloc_(alloc)
      {}

      _derived_op(_derived_op &&) = delete;

      constexpr ~_derived_op() final = default;

      constexpr void start() & noexcept final
      {
        ::STDEXEC::start(op_);
      }

      // objects of this type are allocated with an allocator of type Allocator so they need
      // to be deallocated using the same allocator; providing a class-specific overload of
      // a destroying operator delete allows us to store the relevant allocator inside the
      // to-be-destroyed object and retrieve it before running the destructor
      static constexpr void operator delete(_derived_op *p, std::destroying_delete_t)
      {
        using traits = std::allocator_traits<Allocator>::template rebind_traits<_derived_op>;

        typename traits::allocator_type alloc = std::move(p->alloc_);
        traits::destroy(alloc, p);
        traits::deallocate(alloc, p, 1);
      }

     private:
      connect_result_t<Sender, Receiver> op_;
      STDEXEC_ATTRIBUTE(no_unique_address)
      Allocator alloc_;
    };

    template <class Receiver, class Sigs, class... Queries>
    class _func_op;

    // The concrete operation state resulting from connecting a function<...> to a concrete
    // receiver of type Receiver. This type manages a dynamically-allocated _derived_op instance,
    // which is the type-erased operation state resulting from connecting the type-erased sender
    // to an _any::_any_receiver_ref with the given completion signatures and queries.
    template <class Receiver, class... Sigs, class... Queries>
    class _func_op<Receiver, completion_signatures<Sigs...>, Queries...>
    {
      // rcvr_ has to be initialized before op_ because our implementation of get_env
      // is empirically accessed during our constructor and depends on rcvr_ being initialized
      STDEXEC_ATTRIBUTE(no_unique_address)
      Receiver rcvr_;
      // the default deleter is OK because we've virtualized operator delete to invoke
      // the allocator-based deallocation logic that's necessary to properly support
      // a user-provided frame allocator
      std::unique_ptr<_base_op> op_;

      using _receiver_t =
        ::exec::_any::_any_receiver_ref<completion_signatures<Sigs...>, queries<Queries...>>;

     public:
      using operation_state_concept = operation_state_tag;

      template <class Factory>
      constexpr _func_op(Receiver rcvr, Factory factory)
        : rcvr_(std::move(rcvr))
        , op_(factory(_receiver_t(rcvr_)))
      {}

      _func_op(_func_op &&) = delete;

      constexpr ~_func_op() = default;

      constexpr void start() & noexcept
      {
        op_->start();
      }
    };

    // given the concrete receiver's environment, choose the frame allocator; first choice
    // is the result of get_frame_allocator(env), second choice is get_allocator(env), and
    // the default is std::allocator
    template <class Env>
    constexpr auto choose_frame_allocator(Env const &env) noexcept
    {
      if constexpr (requires { get_frame_allocator(env); })
      {
        return get_frame_allocator(env);
      }
      else if constexpr (requires { get_allocator(env); })
      {
        return get_allocator(env);
      }
      else
      {
        return std::allocator<std::byte>();
      }
    }

    template <class Args, class Sigs, class Queries>
    class _func_impl;

    // the main implementation of the type-erasing sender function<...>
    //
    // SndrCncpt should be std::execution::sender_concept
    // Args... is the argument types used to construct the erased sender
    // Sigs... is the supported completion signatures
    // Queries... is the list of environment queries that must be supported by the eventual
    //            receiver; it's a pack of function type like Return(Query, Args...) or
    //            Return(Query, Args...) noexcept. The named query, when given the specified
    //            arguments, must return a value convertible to Return, and it must be noexcept,
    //            or not, as appropriate
    template <class SndrCncpt, class... Args, class... Sigs, class... Queries>
    class _func_impl<SndrCncpt(Args...), completion_signatures<Sigs...>, queries<Queries...>>
    {
      using _receiver_t =
        ::exec::_any::_any_receiver_ref<completion_signatures<Sigs...>, queries<Queries...>>;

      // the type-erased sender factory that, when called, constructs the erased sender from
      // args_ and connects the resulting sender to the provided receiver
      std::unique_ptr<_base_op> (*factory_)(_receiver_t, Args &&...);
      STDEXEC_ATTRIBUTE(no_unique_address)
      std::tuple<Args...> args_;

     public:
      using sender_concept = SndrCncpt;

      // TODO: I only know this works for empty lambdas; figure out whether function pointers
      //       and/or pointer-to-member functions can be made to work
      template <STDEXEC::__callable<Args...> Factory>
        requires STDEXEC::__not_decays_to<Factory, _func_impl>  //
              && STDEXEC::__std::constructible_from<Factory>    //
              && STDEXEC::__callable<Factory, Args...>
              && STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>, _receiver_t>
      constexpr explicit _func_impl(Args &&...args, Factory &&factory)
        noexcept(STDEXEC::__nothrow_move_constructible<Args...>)
        : args_(std::forward<Args>(args)...)
      {
        using sender_t = std::invoke_result_t<Factory, Args...>;

        factory_ = [](_receiver_t rcvr, Args &&...args) -> std::unique_ptr<_base_op>
        {
          // the type of the allocator provided by the receiver's environment
          using alloc_t = decltype(choose_frame_allocator(get_env(rcvr)));
          // the traits for that allocator, but normalized to std::byte to minimize
          // template instantiations
          using traits_t = std::allocator_traits<alloc_t>::template rebind_traits<std::byte>;

          // the type of operation we'll ultimately allocate, which depends on the type of
          // the allocator we're using
          using op_t = _derived_op<sender_t, _receiver_t, typename traits_t::allocator_type>;

          // finally, the allocator traits for an allocator that can allocate an op_t
          using traits = traits_t::template rebind_traits<op_t>;

          // ...and the allocator itself
          typename traits::allocator_type alloc(choose_frame_allocator(get_env(rcvr)));

          auto *op = traits::allocate(alloc, 1);

          __scope_guard guard{[&]() noexcept { traits::deallocate(alloc, op, 1); }};

          // TODO: as mentioned above, Factory must be a stateless lambda, which makes it
          //       default-constructible like this; this obviously doesn't work if Factory
          //       is a pointer type
          Factory factory;

          traits::construct(alloc,
                            op,
                            factory(std::forward<Args>(args)...),
                            std::move(rcvr),
                            alloc);

          guard.__dismiss();

          return std::unique_ptr<_base_op>(op);
        };
      }

      template <class Sender, class Env>
      static consteval auto get_completion_signatures() noexcept
      {
        static_assert(STDEXEC_IS_BASE_OF(_func_impl, __decay_t<Sender>));

        // TODO: validate that Env supports all the required queries
        //
        //if constexpr (std::constructible_from<Env, RcvrEnv const &>)
        {
          return completion_signatures<Sigs...>{};
        }
        //else
        //{
        // TODO: make this error accurate
        //return __throw_compile_time_error(__unrecognized_sender_error_t<Sender, RcvrEnv>());
        //}
      }

      // TODO: this assumes rvalue connection; lvalue connection requires thought and tests
      template <class Receiver>
      constexpr _func_op<Receiver, completion_signatures<Sigs...>, Queries...>
      connect(Receiver rcvr)
      {
        return {std::move(rcvr),
                [this](auto rcvr)
                {
                  return std::apply(
                    [&rcvr, this](Args &&...args)
                    { return factory_(std::move(rcvr), std::forward<Args>(args)...); },
                    std::move(args_));
                }};
      }
    };

    // given a possibly-noexcept function type like Return(Args...), compute the appropriate
    // completion_signatures. The result is a set_value overload taking either Return&& or
    // no args when Return is void, set_stopped, and, when the function type is not noexcept,
    // set_error(std::exception_ptr)
    template <class Sig>
    struct _sigs_from;

    template <class Return, class... Args>
    struct _sigs_from<Return(Args...)>
    {
      using type = STDEXEC::completion_signatures<STDEXEC::set_error_t(std::exception_ptr),
                                                  STDEXEC::set_stopped_t(),
                                                  STDEXEC::set_value_t(Return)>;
    };

    template <class... Args>
    struct _sigs_from<void(Args...)>
    {
      using type = STDEXEC::completion_signatures<STDEXEC::set_error_t(std::exception_ptr),
                                                  STDEXEC::set_stopped_t(),
                                                  STDEXEC::set_value_t()>;
    };

    template <class Return, class... Args>
    struct _sigs_from<Return(Args...) noexcept>
    {
      using type =
        STDEXEC::completion_signatures<STDEXEC::set_stopped_t(), STDEXEC::set_value_t(Return)>;
    };

    template <class... Args>
    struct _sigs_from<void(Args...) noexcept>
    {
      using type = STDEXEC::completion_signatures<STDEXEC::set_stopped_t(), STDEXEC::set_value_t()>;
    };

    template <class Sig>
    using _sigs_from_t = _sigs_from<Sig>::type;
  }  // namespace _func

  // the user-facing interface to exec::function that supports several different declaration
  // styles, including:
  //  - function<int(bar, baz)>: a fallible function from (bar, baz) to int
  //  - function<int(bar, baz) noexcept>: an infallible function from (bar, baz) to int
  //  - function<sender_tag(bar, baz), completion_signatures<...>>: a function from (bar, baz)
  //    that completes in the ways specified by the given specialization of completion_signatures
  //  - function<int(bar, baz), queries<Return(Query, Args...), ...>: a function from (bar, baz)
  //    to int that requires the final receiver to have an environment that supports the
  //    Query query, taking arguments Args..., and returning an object convertible to Return; queries
  //    may be required to be no-throw by delcaring the function type noexcept
  //  - function<
  //        sender_tag(bar, baz),
  //        completion_signatures<...>,
  //        queries<Return(Query, Args...)>>: a fully-specified async function that maps (bar, baz)
  //    to the specified completions, requiring the specified queries in the ultimate receiver's
  //    environment
  //
  // Future: support C-style ellipsis arguments in the function signature to permit type-erased
  //         arguments as well, like function<int(bar, baz, ...)> (a fallible function from
  //         (bar, baz) plus unspecified, erased additional arguments to int)
  template <class...>
  struct function;

  template <class Return, class... Args>
  // should this require STDEXEC::__not_same_as<Return, STDEXEC::sender_tag>?
  //
  // you *could* write STDEXEC::just(STDEXEC::sender_tag{}), but it seems more likely
  // that invoking this specialization with Return set to sender_tag is a bug...
  //
  // the same question applies to all the specializations below that take explicit
  // completion signatures
  struct function<Return(Args...)>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...)>,
                        queries<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...)>,
                                   queries<>>;

    using base::base;
  };

  template <class Return, class... Args>
  struct function<Return(Args...) noexcept>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...) noexcept>,
                        queries<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...) noexcept>,
                                   queries<>>;

    using base::base;
  };

  template <class... Args, class Sigs>
    requires STDEXEC::__is_instance_of<Sigs, STDEXEC::completion_signatures>
  struct function<STDEXEC::sender_tag(Args...), Sigs>
    : _func::_func_impl<STDEXEC::sender_tag(Args...), Sigs, queries<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...), Sigs, queries<>>;

    using base::base;
  };

  template <class Return, class... Args, class... Queries>
  struct function<Return(Args...), queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...)>,
                        queries<Queries...>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...)>,
                                   queries<Queries...>>;

    using base::base;
  };

  template <class Return, class... Args, class... Queries>
  struct function<Return(Args...) noexcept, queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...) noexcept>,
                        queries<Queries...>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...) noexcept>,
                                   queries<Queries...>>;

    using base::base;
  };

  template <class... Args, class... Sigs, class... Queries>
  struct function<STDEXEC::sender_tag(Args...),
                  STDEXEC::completion_signatures<Sigs...>,
                  queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        STDEXEC::completion_signatures<Sigs...>,
                        queries<Queries...>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   STDEXEC::completion_signatures<Sigs...>,
                                   queries<Queries...>>;

    using base::base;
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;
