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

  namespace _qry_detail
  {
    template <class Sig>
    inline constexpr bool is_query_function_v = false;

    template <class Return, class Query, class... Args, bool NoThrow>
    inline constexpr bool is_query_function_v<Return(Query, Args...) noexcept(NoThrow)> = true;
  }  // namespace _qry_detail

  // a "type list" for bundling together function type representing queries to support in
  // a type-erased environment. All of the types in Queries... must be (possibly noexcept)
  // function types. For example:
  //
  //   queries<
  //     std::execution::inline_stop_token(std::execution::get_stop_token_t) noexcept,
  //     std::pmr::polymorphic_allocator<std::byte>(std::execution::get_allocator_t)
  //   >
  template <class... Queries>
    requires(_qry_detail::is_query_function_v<Queries> && ...)
  struct queries
  {};

  namespace _func
  {
    using namespace STDEXEC;

    // a recursively-defined type with a vtable containing one virtual function for
    // each query in Queries...
    //
    // the base template is an empty class, representing the empty set of queries.
    template <class... Queries>
    struct _env_of_queries
    {};

    // a special case in the recursion: when there is only one query in the pack, there's
    // no base implementation of query to put in the using statement
    template <class Return, class Query, class... Args, bool NoThrow>
    struct _env_of_queries<Return(Query, Args...) noexcept(NoThrow)>
    {
      virtual Return query(Query query, Args &&...args) const noexcept(NoThrow) = 0;
    };

    // the recursive case that declares the named query as a pure virtual member function
    // and inherits the rest of the required queries through inheritance
    template <class Return, class Query, class... Args, bool NoThrow, class... Queries>
    struct _env_of_queries<Return(Query, Args...) noexcept(NoThrow), Queries...>
      : private _env_of_queries<Queries...>
    {
      _env_of_queries() = default;

      _env_of_queries(_env_of_queries &&) = delete;

      using _env_of_queries<Queries...>::query;

      virtual Return query(Query query, Args &&...args) const noexcept(NoThrow) = 0;

     protected:
      ~_env_of_queries() = default;
    };

    // an environment type that delegates query to an _env_of_queries<Queries...> so that the
    // environment type that we traffic in is cheaply copyable
    template <class... Queries>
    struct _delegate_env
    {
      using delegate_t = _env_of_queries<Queries...>;

      explicit _delegate_env(delegate_t const &delegate) noexcept
        : delegate_(std::addressof(delegate))
      {}

      template <class Query, class... Args>
        requires __queryable_with<delegate_t, Query, Args...>
      constexpr auto query(Query, Args &&...args) const
        noexcept(__nothrow_queryable_with<delegate_t, Query, Args...>)
          -> __query_result_t<delegate_t, Query, Args...>
      {
        return __query<Query>()(*delegate_, std::forward<Args>(args)...);
      }

     private:
      delegate_t const *delegate_;
    };

    // in the base case, there's no need to store a pointer
    template <>
    struct _delegate_env<>
    {
      using delegate_t = _env_of_queries<>;

      explicit _delegate_env(delegate_t const &) noexcept {}
    };

    template <class Sig>
    struct _virt_completion;

    // a vtable entry representing a receiver completion function; CPO should be a completion
    // function (e.g. set_Value_t), and Args... is the expected argument list.
    template <class CPO, class... Args>
    struct _virt_completion<CPO(Args...)>
    {
      constexpr _virt_completion() = default;

      _virt_completion(_virt_completion &&) = delete;

      constexpr virtual void complete(CPO, Args &&...) noexcept = 0;

     protected:
      constexpr ~_virt_completion() = default;
    };

    template <class Sigs, class... Queries>
    struct _virt_completions;

    // a class template that bundles together a pure virtual completion function for each
    // of the specified completion functions, and provides an implementation of get_env
    template <class... Sigs, class... Queries>
    struct _virt_completions<completion_signatures<Sigs...>, Queries...>
      : _virt_completion<Sigs>...
      , _env_of_queries<Queries...>
    {
      constexpr _virt_completions() = default;

      _virt_completions(_virt_completions &&) = delete;

      // this will complain if sizeof...(Sigs) == 0, but a sender with no completions
      // isn't super useful...
      using _virt_completion<Sigs>::complete...;

      constexpr _delegate_env<Queries...> get_env() const noexcept
      {
        return _delegate_env<Queries...>(*this);
      }

     protected:
      constexpr ~_virt_completions() = default;
    };

    template <class Sigs, class... Queries>
    class _func_rcvr;

    // a type-erased receiver expecting to be completed by one of the completions specified
    // in Sigs..., and providing an environment that supports the queries specified in
    // Queries...
    //
    // this is the receiver type that is passed into the sender being type-erased by a
    // function<...>, and it forwards completions to the concrete receiver through the
    // internal completer_ pointer
    template <class... Sigs, class... Queries>
    class _func_rcvr<completion_signatures<Sigs...>, Queries...>
    {
      using completer_t = _virt_completions<completion_signatures<Sigs...>, Queries...>;

      completer_t *completer_;

     public:
      using receiver_concept = receiver_tag;

      constexpr explicit _func_rcvr(completer_t &completer) noexcept
        : completer_(std::addressof(completer))
      {}

      template <class Error>
      constexpr void set_error(Error &&err) && noexcept
        requires requires { completer_->complete(set_error_t{}, std::forward<Error>(err)); }
      {
        completer_->complete(set_error_t{}, std::forward<Error>(err));
      }

      constexpr void set_stopped() && noexcept
        requires requires { completer_->complete(set_stopped_t{}); }
      {
        completer_->complete(set_stopped_t{});
      }

      template <class... Values>
      constexpr void set_value(Values &&...values) && noexcept
        requires requires { completer_->complete(set_value_t{}, std::forward<Values>(values)...); }
      {
        completer_->complete(set_value_t{}, std::forward<Values>(values)...);
      }

      constexpr auto get_env() const noexcept -> _delegate_env<Queries...>
      {
        return STDEXEC::get_env(*completer_);
      }
    };

    // the type-erased operation state type that supports starting and destruction
    struct _base_op
    {
      constexpr _base_op() = default;

      _base_op(_base_op &&) = delete;

      constexpr virtual ~_base_op() = default;

      constexpr virtual void start() & noexcept = 0;
    };

    // the operation state resulting from connecting a sender being erased by a function<...>
    // with a _func_rcvr<...>; inherits from _base_op, and provides a class-specific override
    // of operator delete that invokes the allocator deallocation protocol
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

    // a recursive implementation of Base, which is expected to inherit from
    // _virt_completions
    template <class Base, class Derived, class... Sigs>
    struct _func_op_completion;

    // the base case of the recursive implementation; all subclasses of this type have,
    // together, overridden all the virtual functions in Base so now we just need to
    // inherit from Base to ensure those virtual functions exist to be overridden
    template <class Base, class Derived>
    struct _func_op_completion<Base, Derived> : Base
    {};

    // the recursive case, which implements a single overload of complete and delegates
    // the implementation of all remaining overloads to the base class
    template <class Base, class Derived, class CPO, class... Args, class... Sigs>
    struct _func_op_completion<Base, Derived, CPO(Args...), Sigs...>
      : _func_op_completion<Base, Derived, Sigs...>
    {
      void complete(CPO, Args &&...args) noexcept final
      {
        // This seems like it ought to be true, but it fails...
        //
        // Some testing shows it's being evaluated when Derived is incomplete
        // during constraint satisfaction testing.
        //
        // static_assert(std::derived_from<_func_op_completion, Derived>);
        //
        // Consider: what if _func_op_completion<Base, Derived> (i.e. the base case of
        //           this recursive class hierarchy) owned the receiver? We could avoid
        //           CRTP and just use this->rcvr_, maybe.
        auto &rcvr = static_cast<Derived *>(this)->rcvr_;
        CPO{}(std::move(rcvr), std::forward<Args>(args)...);
      }
    };

    // a recursive implementation of all the queries in Queries...
    template <class Base, class Derived, class... Queries>
    struct _func_op_queries;

    // the base case of the recursive implementation; there are no more queries to
    // implement so just inherit from Base
    template <class Base, class Derived>
    struct _func_op_queries<Base, Derived> : Base
    {};

    // the recursive case, which implements a single query overload and delegates the
    // implementation of the remaining overloads to the base class
    template <class Base,
              class Derived,
              class Return,
              class Query,
              class... Args,
              bool NoThrow,
              class... Queries>
    struct _func_op_queries<Base, Derived, Return(Query, Args...) noexcept(NoThrow), Queries...>
      : _func_op_queries<Base, Derived, Queries...>
    {
      Return query(Query, Args &&...args) const noexcept(NoThrow) final
      {
        // the idea of storing the receiver in the base class could help here, too, but
        // we'd need to be careful about which class template is actually the base class
        auto const &rcvr = static_cast<Derived const *>(this)->rcvr_;
        return __query<Query>()(STDEXEC::get_env(rcvr), std::forward<Args>(args)...);
      }
    };

    template <class Receiver, class Sigs, class... Queries>
    class _func_op;

    // the concrete operation state resulting from connecting a function<...> to a concrete
    // receiver of type Receiver. this type manages a dynamically-allocated _derived_op instance,
    // which is the type-erased operation state resulting from connecting the type-erased sender
    // to a _func_rcvr
    template <class Receiver, class... Sigs, class... Queries>
    class _func_op<Receiver, completion_signatures<Sigs...>, Queries...>
      : _func_op_completion<
          _func_op_queries<_virt_completions<completion_signatures<Sigs...>, Queries...>,
                           _func_op<Receiver, completion_signatures<Sigs...>, Queries...>,
                           Queries...>,
          _func_op<Receiver, completion_signatures<Sigs...>, Queries...>,
          Sigs...>
    {
      // rcvr_ has to be initialized before op_ because our implementation of get_env
      // is empirically accessed during our constructor and depends on rcvr_ being initialized
      STDEXEC_ATTRIBUTE(no_unique_address)
      Receiver rcvr_;
      // the default deleter is OK because we've virtualized operator delete to invoke
      // the allocator-based deallocation logic that's necessary to properly support
      // a user-provided frame allocator
      std::unique_ptr<_base_op> op_;

      // these friend declaratiosn allow our CRTP base classes to access rcvr_; they could
      // disappear if we moved ownership of rcvr_ into the base class object
      template <class, class, class...>
      friend struct _func_op_completion;

      template <class, class, class...>
      friend struct _func_op_queries;

     public:
      using operation_state_concept = operation_state_tag;

      template <class Factory>
      constexpr _func_op(Receiver rcvr, Factory factory)
        : rcvr_(std::move(rcvr))
        , op_(factory(_func_rcvr<completion_signatures<Sigs...>, Queries...>(*this)))
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
      // the type-erased sender factory that, when called, constructs the erased sender from
      // args_ and connects the resulting sender to the provided receiver
      std::unique_ptr<_base_op> (*factory_)(_func_rcvr<completion_signatures<Sigs...>, Queries...>,
                                            Args &&...);
      STDEXEC_ATTRIBUTE(no_unique_address)
      std::tuple<Args...> args_;

     public:
      using sender_concept = SndrCncpt;

      // TODO: I only know this works for empty lambdas; figure out whether function pointers
      //       and/or pointer-to-member functions can be made to work
      template <STDEXEC::__callable<Args...> Factory>
        requires STDEXEC::__not_decays_to<Factory, _func_impl>  //
              && std::constructible_from<Factory>               //
              && STDEXEC::__callable<Factory, Args...>
              && STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>,
                                    _func_rcvr<completion_signatures<Sigs...>, Queries...>>
      constexpr explicit(sizeof...(Args) == 0) _func_impl(Args &&...args, Factory &&factory)
        noexcept((std::is_nothrow_constructible_v<Args, Args> && ...))
        : args_(std::forward<Args>(args)...)
      {
        using sender_t   = std::invoke_result_t<Factory, Args...>;
        using receiver_t = _func_rcvr<completion_signatures<Sigs...>, Queries...>;

        factory_ = [](receiver_t rcvr, Args &&...args) -> std::unique_ptr<_base_op>
        {
          // the type of the allocator provided by the receiver's environment
          using alloc_t = decltype(choose_frame_allocator(get_env(rcvr)));
          // the traits for that allocator, but normalized to std::byte to minimize
          // template instantiations
          using traits_t = std::allocator_traits<alloc_t>::template rebind_traits<std::byte>;

          // the type of operation we'll ultimately allocate, which depends on the type of
          // the allocator we're using
          using op_t = _derived_op<sender_t, receiver_t, typename traits_t::allocator_type>;

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

  template <class Return, class... Args, bool NoThrow>
  // should this require STDEXEC::__not_same_as<Return, STDEXEC::sender_tag>?
  //
  // you *could* write STDEXEC::just(STDEXEC::sender_tag{}), but it seems more likely
  // that invokign this specialization with Return set to sender_tag is a bug...
  //
  // the same question applies to all the specializations below that take explicit
  // completion signatures
  struct function<Return(Args...) noexcept(NoThrow)>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
                        queries<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
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

  template <class Return, class... Args, bool NoThrow, class... Queries>
  struct function<Return(Args...) noexcept(NoThrow), queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
                        queries<Queries...>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
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
