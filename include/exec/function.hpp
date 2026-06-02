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
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__read_env.hpp"
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__sender_concepts.hpp"
#include "../stdexec/__detail/__static_vector.hpp"
#include "../stdexec/__detail/__tuple.hpp"
#include "../stdexec/__detail/__typeinfo.hpp"
#include "../stdexec/__detail/__utility.hpp"
#include "../stdexec/functional.hpp"

#include "__frame_allocator.hpp"
#include "__memory_resource_adaptor.hpp"
// TODO: split this header into pieces
#include "any_sender_of.hpp"
#include "get_frame_allocator.hpp"

#include <cstddef>
#include <cstring>
#include <memory>

// This file defines function<ReturnType(Arguments...)>, which is a type-erased sender
// that can complete with:
//
//  - set_value(ReturnType)
//  - set_error(std::exception_ptr)
//  - set_stopped()
//
// The type-erased operation state is allocated in connect; to accomplish this deferred
// allocation, the sender holds a tuple of arguments that are passed into a sender-factory
// in connect, which is why the template type parameter is a function type rather than
// just a return type.
//
// The intended use case is an ABI-stable API boundary. The hope is that this is a "better
// task" in that it represents an async function from arguments to value, just like a task
// coroutine, but, by deferring the allocation to connect, we can use receiver environment
// queries to pick the frame allocator from the environment without relying on TLS.
namespace experimental::execution
{
  namespace __func
  {
    using namespace STDEXEC;

    //! given the concrete receiver's environment, choose the frame allocator; first
    //! choice is the result of get_frame_allocator(env), second choice is
    //! get_allocator(env), and the default is std::allocator
    inline constexpr auto __choose_frame_allocator =
      __first_callable{get_frame_allocator, get_allocator, __always{std::allocator<std::byte>()}};

    //! Wrap _Receiver, which is a type-erased receiver, in a type that can extract the
    //! concrete, to-be-erased receiver from the operation state that contains it.
    //!
    //! This wrapper exists primarily as a hook for injecting a defaulted frame allocator
    //! when _Receiver *doesn't* have get_frame_allocator in its environment. That
    //! injection happens in the partial specialization, below.
    template <class _Receiver>
    struct __receiver_wrapper : public _Receiver
    {
      template <class _Opstate>
      constexpr explicit __receiver_wrapper(_Opstate *__opstate)
        : _Receiver(__opstate->__rcvr_)
      {}
    };

    //! Wrap _Receiver, which is a type-erased receiver, in a type that can extract the
    //! concrete, to-be-erased receiver from the operation state that contains it, and
    //! inject an environment that contains a defaulted frame allocator.
    //!
    //! This partial specialization handles the case that _Receiver doesn't have a frame
    //! allocator in its environment, in which case we need to provide a type-erasing
    //! frame allocator in the injected environment because we won't know the concrete
    //! type of the allocator that's actually used as our frame allocator until we're
    //! connected to a concrete receiver. We could provide either
    //! std::pmr::memory_resource*, or std::pmr::polymorphic_allocator<> with basically
    //! the same tradeoffs so we provide an allocator rather than a memory resource to
    //! better match the name of the injected query.
    template <class _Receiver>
      requires(!__queryable_with<env_of_t<_Receiver>, get_frame_allocator_t>)
    struct __receiver_wrapper<_Receiver> : public _Receiver
    {
      //! the injected query response for the frame allocator
      using __prop_t = prop<get_frame_allocator_t, std::pmr::polymorphic_allocator<std::byte>>;

      template <class _Opstate>
      constexpr explicit __receiver_wrapper(_Opstate *__opstate)
        : _Receiver(__opstate->__rcvr_)
        , __env_(&__opstate->__env_)
      {}

      constexpr auto get_env() const noexcept  //
        -> __join_env_t<__prop_t, env_of_t<_Receiver>>
      {
        return __env::__join(*__env_, STDEXEC::get_env(*static_cast<_Receiver const *>(this)));
      }

     private:
      __prop_t *__env_;
    };

    template <class _Sigs, class _Queries>
    using __any_receiver_ref = ::exec::_any::_any_receiver_ref<_Sigs, _Queries>;

    template <class _Receiver, class _Sigs, class _Queries>
    struct __opstate_base
    {
      using __receiver_t   = __receiver_wrapper<__any_receiver_ref<_Sigs, _Queries>>;
      using __stop_token_t = stop_token_of_t<env_of_t<__receiver_t>>;

      _any::_state<_Receiver, __stop_token_t> __rcvr_;
    };

    template <class _Receiver, class _Sigs, class _Queries>
      requires(
        !__queryable_with<env_of_t<__any_receiver_ref<_Sigs, _Queries>>, get_frame_allocator_t>)
    struct __opstate_base<_Receiver, _Sigs, _Queries>
    {
      using __receiver_t   = __receiver_wrapper<__any_receiver_ref<_Sigs, _Queries>>;
      using __prop_t       = __receiver_t::__prop_t;
      using __stop_token_t = stop_token_of_t<env_of_t<__receiver_t>>;
      using __adaptee_t    = decltype(__choose_frame_allocator(__declval<_Receiver const &>()));

      __memory_resource_adaptor_t<__adaptee_t> __resource_;
      __prop_t                                 __env_;
      _any::_state<_Receiver, __stop_token_t>  __rcvr_;

      explicit __opstate_base(_Receiver __rcvr)
        : __resource_(__choose_frame_allocator(std::as_const(__rcvr)))
        , __env_(__make_env())
        , __rcvr_(static_cast<_Receiver &&>(__rcvr))
      {}

     private:
      //! the indirection through __make_env and __make_alloc is to work around what
      //! appears to be miscompilation with Clang 16; initializing __env_ inline
      //! rather than delegating to these helpers results in passing an invalid
      //! address to the polymorphic_allocator constructor instead of the address of
      //! __resource_, leading to segfaults
      __prop_t __make_env()
      {
        return __prop_t(get_frame_allocator, __make_alloc());
      }

      std::pmr::polymorphic_allocator<> __make_alloc()
      {
        return std::pmr::polymorphic_allocator<>(&__resource_);
      }
    };

    //! The concrete operation state resulting from connecting a function<...> to a
    //! concrete receiver of type Receiver. This type manages an _any::_any_opstate_base
    //! instance, which is the type-erased operation state resulting from connecting the
    //! type-erased sender to an _any::_any_receiver_ref with the given completion
    //! signatures and queries.
    template <class _Receiver, class _Sigs, class _Queries>
    class __opstate : public __opstate_base<_Receiver, _Sigs, _Queries>
    {
      using __base = __opstate_base<_Receiver, _Sigs, _Queries>;
      using typename __base::__receiver_t;

      _any::_any_opstate_base __op_;

     public:
      using operation_state_concept = operation_state_tag;

      template <class _Factory>
      explicit constexpr __opstate(_Receiver __rcvr, _Factory __factory)
        : __base(static_cast<_Receiver &&>(__rcvr))
        , __op_(__factory(__receiver_t(this)))
      {}

      constexpr void start() & noexcept
      {
        __op_.start();
      }
    };

    template <class _Sigs, class _Queries, class... _Args>
    class __function;

    //! the main implementation of the type-erasing sender function<...>
    //
    //! \tparam _Sigs The supported completion signatures
    //!
    //! \tparam _Queries The list of environment queries that must be supported by
    //! the eventual receiver; it's a pack of function type like Return(Query, Args...) or
    //! Return(Query, Args...) noexcept. The named query, when given the specified
    //! arguments, must return a value convertible to Return, and it must be noexcept, or
    //! not, as appropriate
    //!
    //! \tparam _Args The argument types used to construct the erased sender
    template <class _Sigs, class... _Queries, class... _Args>
    class __function<_Sigs, queries<_Queries...>, _Args...>
    {
      using __receiver_t = __receiver_wrapper<__any_receiver_ref<_Sigs, queries<_Queries...>>>;

      template <class _Receiver>
      using __opstate_t = __opstate<_Receiver, _Sigs, queries<_Queries...>>;

      template <class _Factory>
      static constexpr auto
      __mk_opstate(void *__storage, __receiver_t __rcvr, _Args &&...__args)  //
        -> _any::_any_opstate_base
      {
        auto &__make_sender = *__std::start_lifetime_as<_Factory>(__storage);
        using __alloc_t     = decltype(__choose_frame_allocator(get_env(__rcvr)));
        auto __alloc = __frame_allocator_t<__alloc_t>(__choose_frame_allocator(get_env(__rcvr)));
        return _any::_any_opstate_base(__in_place_from,
                                       std::allocator_arg,
                                       __alloc,
                                       STDEXEC::connect,
                                       __invoke(__make_sender, static_cast<_Args &&>(__args)...),
                                       static_cast<__receiver_t &&>(__rcvr));
      }

      //! The curried arguments that will be passed to __make_sender_ from inside
      //! __make_opstate_.
      STDEXEC_ATTRIBUTE(no_unique_address)
      __tuple<_Args...> __args_;
      //! The type-erased operation state factory; it points to a function that knows the
      //! concrete type of the sender factory stored in __make_sender_ so that it can
      //! construct the desired sender on demand and connect it to the given receiver. The
      //! expected arguments are the address of __make_sender_, the __any_receiver_ref to
      //! connect the sender to, and the arguments to pass to __make_sender_ to construct
      //! the sender.
      _any::_any_opstate_base (*__make_opstate_)(void *, __receiver_t, _Args &&...);
      //! Storage for the sender factory passed to our constructor template;
      //! __make_opstate_ will reconstitute the actual factory from this bag-of-bytes with
      //! start_lifetime_as because it internally knows the concrete type of the
      //! user-provided sender factory. We're reserving 2 * sizeof(void *) bytes to permit
      //! the factory to be a pointer to member function, which usually requires two
      //! pointers.
      std::byte __make_sender_[2 * sizeof(void *)]{};

     public:
      using sender_concept = sender_tag;

      template <__invocable<_Args...> _Factory>
        requires __not_decays_to<_Factory, __function>           //
                && (STDEXEC_IS_TRIVIALLY_COPYABLE(_Factory))     //
                && (sizeof(_Factory) <= sizeof(__make_sender_))  //
                && sender_to<__invoke_result_t<_Factory, _Args...>, __receiver_t>
      constexpr explicit __function(_Args &&...__args, _Factory __factory)
        noexcept(__nothrow_move_constructible<_Args...>)
        : __args_(static_cast<_Args &&>(__args)...)
        , __make_opstate_(&__mk_opstate<_Factory>)
      {
        std::memcpy(__make_sender_, std::addressof(__factory), sizeof(_Factory));
      }

      //! this implementation of get_completion_signatures is taken directly from the
      //! equivalent function on any_sender_of
      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures()
      {
        static_assert(__decays_to_derived_from<_Self, __function>);
        //! throw if _Env does not contain the queries needed to type-erase the receiver:
        using __check_queries_t = __mfind_error<_any::_check_query_t<_Queries, _Env...>...>;
        if constexpr (__merror<__check_queries_t>)
          return __throw_compile_time_error(__check_queries_t());
        else
          return _Sigs();
      }

      template <class _Receiver>
      constexpr auto connect(_Receiver __rcvr) &&  //
        -> __opstate_t<_Receiver>
      {
        auto __factory = [this]<class _RcvrRef>(_RcvrRef __rcvr)
        {
          return __apply(__make_opstate_,
                         static_cast<__tuple<_Args...> &&>(__args_),
                         __make_sender_,
                         static_cast<_RcvrRef &&>(__rcvr));
        };
        return __opstate_t<_Receiver>{static_cast<_Receiver &&>(__rcvr), __factory};
      }

      template <class _Receiver>
        requires __std::copy_constructible<__function>
      constexpr auto connect(_Receiver __rcvr) const &  //
        -> __opstate_t<_Receiver>
      {
        return __function(*this).connect(static_cast<_Receiver &&>(__rcvr));
      }
    };

    template <auto _Types, template <class...> class _Template, std::size_t... _Is>
    consteval auto __canonicalize_splice(__indices<_Is...>) noexcept
    {
      return _Template<__msplice<_Types[_Is]>...>();
    }

    template <std::size_t _Size>
    consteval auto __canonicalize_impl(__static_vector<__type_index, _Size> __types) noexcept
    {
      std::ranges::sort(__types);
      auto const __rest = std::ranges::unique(__types);
      __types.erase(__rest.begin(), __types.end());
      return __types;
    }

    template <template <class...> class _List, class... _Types>
    consteval auto __canonicalize(_List<_Types...> *) noexcept
    {
      using __types_t        = __static_vector<__type_index, sizeof...(_Types)>;
      constexpr auto __types = __func::__canonicalize_impl(__types_t{__mtypeid<_Types>...});
      return __func::__canonicalize_splice<__types, _List>(__make_indices<__types.size()>());
    }

    //! Map the type-list _Sigs to a canonical form, which sorts and uniques the contained
    //! elements to ensure user-specified type-lists are not order-dependent.
    //!
    //! \tparam _Sigs a type-list of types to be sorted and uniqued; expected to be a
    //!         specialization of completion_signatures or queries.
    template <class _Sigs>
    using __canonical_t = decltype(__func::__canonicalize(static_cast<_Sigs *>(nullptr)));

    //! Given a return type and a bool indicating whether the function is noexcept,
    //! compute the appropriate completion_signatures. The result is a set_value overload
    //! taking either Return&& or no args when Return is void, set_stopped, and, when the
    //! function type is not noexcept, set_error(std::exception_ptr)
    template <class _Return, bool _NoExcept>
    using __sigs_from_t = __canonical_t<__concat_completion_signatures_t<
      completion_signatures<__single_value_sig_t<_Return>, set_stopped_t()>,
      __eptr_completion_unless_t<__mbool<_NoExcept>>>>;

    //! Map a variety of function<...> specifications into the canonical type-erased
    //! contract represented by the user-provided specification.
    //!
    //! The canonical specification looks like this:
    //!
    //!   function<
    //!       sender_tag(Args...),
    //!       completion_signatures<Sigs...>,
    //!       queries<Queries...>>
    //!
    //! where:
    //!  - Args... is the type-erased sender factory's parameter list
    //!  - Sigs... is the set of completion signatures that the erased sender is allowed
    //!            to advertise
    //!  - Queries... is the set of queries that the eventual receiver's environment must
    //!               support
    //!
    //! The order of Args... is obviously important, but Sigs... and Queries... are both
    //! canonicalized into a sorted and uniqued list to ensure order is irrelevant.
    template <class...>
    struct __make_function;

    template <class _Return, class... _Args>
    struct __make_function<_Return(_Args...)>
    {
      using type = __function<__sigs_from_t<_Return, false>, queries<>, _Args...>;
    };

    template <class _Return, class... _Args>
    struct __make_function<_Return(_Args...) noexcept>
    {
      using type = __function<__sigs_from_t<_Return, true>, queries<>, _Args...>;
    };

    template <class... _Args, class... _Sigs>
    struct __make_function<sender_tag(_Args...), completion_signatures<_Sigs...>>
    {
      using type = __function<__canonical_t<completion_signatures<_Sigs...>>, queries<>, _Args...>;
    };

    template <class _Return, class... _Args, class... _Queries>
    struct __make_function<_Return(_Args...), queries<_Queries...>>
    {
      using type =
        __function<__sigs_from_t<_Return, false>, __canonical_t<queries<_Queries...>>, _Args...>;
    };

    template <class _Return, class... _Args, class... _Queries>
    struct __make_function<_Return(_Args...) noexcept, queries<_Queries...>>
    {
      using type =
        __function<__sigs_from_t<_Return, true>, __canonical_t<queries<_Queries...>>, _Args...>;
    };

    template <class... _Args, class... _Sigs, class... _Queries>
    struct __make_function<sender_tag(_Args...),
                           completion_signatures<_Sigs...>,
                           queries<_Queries...>>
    {
      using type = __function<__canonical_t<completion_signatures<_Sigs...>>,
                              __canonical_t<queries<_Queries...>>,
                              _Args...>;
    };
  }  // namespace __func

  //! the user-facing interface to exec::function that supports several different
  //! declaration styles, including:
  //!
  //! - function<int(bar, baz)>: a fallible function from (bar, baz) to int
  //! - function<int(bar, baz) noexcept>: an infallible function from (bar, baz) to int
  //! - function<sender_tag(bar, baz), completion_signatures<...>>: a function from (bar,
  //!   baz) that completes in the ways specified by the given specialization of
  //!   completion_signatures
  //! - function<int(bar, baz), queries<Return(Query, Args...), ...>: a function from
  //!   (bar, baz) to int that requires the final receiver to have an environment that
  //!   supports the Query query, taking arguments Args..., and returning an object
  //!   convertible to Return; queries may be required to be no-throw by declaring the
  //!   function type noexcept
  //! - function<sender_tag(bar, baz), completion_signatures<...>, queries<Return(Query,
  //!   Args...)>>: a fully-specified async function that maps (bar, baz) to the specified
  //!   completions, requiring the specified queries in the ultimate receiver's
  //!   environment
  //!
  //! Future: support C-style ellipsis arguments in the function signature to permit
  //! type-erased arguments as well, like function<int(bar, baz, ...)> (a fallible
  //! function from (bar, baz) plus unspecified, erased additional arguments to int)
  template <class... _Ts>
  using function = __func::__make_function<_Ts...>::type;
}  // namespace experimental::execution

namespace exec = experimental::execution;
