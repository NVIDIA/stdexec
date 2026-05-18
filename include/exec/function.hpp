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
  namespace _func
  {
    using namespace STDEXEC;

    //! given the concrete receiver's environment, choose the frame allocator; first
    //! choice is the result of get_frame_allocator(env), second choice is
    //! get_allocator(env), and the default is std::allocator
    inline constexpr auto choose_frame_allocator =
      __first_callable{get_frame_allocator, get_allocator, __always{std::allocator<std::byte>()}};

    template <class Receiver>
    struct _receiver_wrapper : public Receiver
    {
      template <class Opstate>
      constexpr explicit _receiver_wrapper(Opstate *opstate)
        : Receiver(opstate->rcvr_)
      {}
    };

    template <class Receiver>
      requires(!__queryable_with<env_of_t<Receiver>, get_frame_allocator_t>)
    struct _receiver_wrapper<Receiver> : public Receiver
    {
      using _prop_t = prop<get_frame_allocator_t, std::pmr::polymorphic_allocator<std::byte>>;

      template <class Opstate>
      constexpr explicit _receiver_wrapper(Opstate *opstate)
        : Receiver(opstate->rcvr_)
        , env_(&opstate->env_)
      {}

      constexpr auto get_env() const noexcept  //
        -> __join_env_t<_prop_t, env_of_t<Receiver>>
      {
        return __env::__join(*env_, STDEXEC::get_env(*static_cast<Receiver const *>(this)));
      }

     private:
      _prop_t *env_;
    };

    template <class Sigs, class Queries>
    using _any_receiver_ref = ::exec::_any::_any_receiver_ref<Sigs, Queries>;

    template <class Adaptee>
    struct _memory_resource_adaptor;

    template <class Adaptee>
      requires __simple_allocator<Adaptee>
            && __same_as<std::byte, typename std::allocator_traits<Adaptee>::value_type>
    struct _memory_resource_adaptor<Adaptee>
    {
      //! Implement memory_resource in terms of an allocator<std::byte>
      class type : public std::pmr::memory_resource
      {
        using traits = std::allocator_traits<Adaptee>;
        static_assert(__same_as<std::byte, typename traits::value_type>);
        typename traits::allocator_type alloc_;

       public:
        template <class Alloc>
          requires(!__same_as<Alloc, type>)
        constexpr explicit type(Alloc const &alloc) noexcept
          : alloc_(alloc)
        {
          using rebound_traits = std::allocator_traits<Alloc>::template rebind_traits<std::byte>;
          static_assert(__same_as<traits, rebound_traits>);
        }

        constexpr void *do_allocate(std::size_t __bytes, std::size_t __align) override
        {
          return traits::allocate(alloc_, __bytes);
        }

        constexpr void do_deallocate(void *__p, std::size_t __bytes, std::size_t __align) override
        {
          traits::deallocate(alloc_, new (__p) std::byte[__bytes], __bytes);
        }

        constexpr bool do_is_equal(std::pmr::memory_resource const &__other) const noexcept override
        {
          if (auto *ptr = dynamic_cast<type const *>(&__other))
          {
            return alloc_ == ptr->alloc_;
          }

          return false;
        }
      };
    };

    template <class Adaptee>
      requires __simple_allocator<Adaptee>
    struct _memory_resource_adaptor<Adaptee>
      //! for an arbitrary allocator, inherit from the adaptor that's implemented
      //! in terms of that allocator rebound to std::byte to minimize template
      //! instantiations, and to make the dynamic_cast in do_is_equal work
      : _memory_resource_adaptor<
          typename std::allocator_traits<Adaptee>::template rebind_alloc<std::byte>>
    {};

    template <class Adaptee>
      requires __std::constructible_from<std::pmr::polymorphic_allocator<std::byte>, Adaptee>
    struct _memory_resource_adaptor<Adaptee>
    {
      //! no need to "adapt" a memory_resource
      using type = Adaptee;
    };

    template <class Adaptee>
    using _memory_resource_adaptor_t = _memory_resource_adaptor<std::remove_cvref_t<Adaptee>>::type;

    template <class Receiver, class Sigs, class Queries>
    struct _opstate_base
    {
      using _receiver_t   = _receiver_wrapper<_any_receiver_ref<Sigs, Queries>>;
      using _stop_token_t = stop_token_of_t<env_of_t<_receiver_t>>;

      _any::_state<Receiver, _stop_token_t> rcvr_;
    };

    template <class Receiver, class Sigs, class Queries>
      requires(!__queryable_with<env_of_t<_any_receiver_ref<Sigs, Queries>>, get_frame_allocator_t>)
    struct _opstate_base<Receiver, Sigs, Queries>
    {
      using _receiver_t   = _receiver_wrapper<_any_receiver_ref<Sigs, Queries>>;
      using _prop_t       = _receiver_t::_prop_t;
      using _stop_token_t = stop_token_of_t<env_of_t<_receiver_t>>;
      using _adaptee_t    = decltype(choose_frame_allocator(__declval<Receiver const &>()));

      _memory_resource_adaptor_t<_adaptee_t> resource_;
      _prop_t                                env_;
      _any::_state<Receiver, _stop_token_t>  rcvr_;

      constexpr explicit _opstate_base(Receiver &&rcvr)
        : resource_(choose_frame_allocator(rcvr))
        , env_{get_frame_allocator, &resource_}
        , rcvr_(static_cast<Receiver &&>(rcvr))
      {}
    };

    //! The concrete operation state resulting from connecting a function<...> to a
    //! concrete receiver of type Receiver. This type manages an _any::_any_opstate_base
    //! instance, which is the type-erased operation state resulting from connecting the
    //! type-erased sender to an _any::_any_receiver_ref with the given completion
    //! signatures and queries.
    template <class Receiver, class Sigs, class Queries>
    class _opstate : public _opstate_base<Receiver, Sigs, Queries>
    {
      using _base = _opstate_base<Receiver, Sigs, Queries>;
      using typename _base::_receiver_t;

      _any::_any_opstate_base op_;

     public:
      using operation_state_concept = operation_state_tag;

      template <class Factory>
      explicit constexpr _opstate(Receiver rcvr, Factory factory)
        : _base(static_cast<Receiver &&>(rcvr))
        , op_(factory(_receiver_t(this)))
      {}

      constexpr void start() & noexcept
      {
        op_.start();
      }
    };

    template <class Sigs, class Queries, class... Args>
    class _function;

    //! the main implementation of the type-erasing sender function<...>
    //
    //! \tparam Sigs The supported completion signatures
    //!
    //! \tparam Queries The list of environment queries that must be supported by
    //! the eventual receiver; it's a pack of function type like Return(Query, Args...) or
    //! Return(Query, Args...) noexcept. The named query, when given the specified
    //! arguments, must return a value convertible to Return, and it must be noexcept, or
    //! not, as appropriate
    //!
    //! \tparam Args The argument types used to construct the erased sender
    template <class Sigs, class... Queries, class... Args>
    class _function<Sigs, queries<Queries...>, Args...>
    {
      using _receiver_t = _receiver_wrapper<_any_receiver_ref<Sigs, queries<Queries...>>>;

      template <class Receiver>
      using _opstate_t = _opstate<Receiver, Sigs, queries<Queries...>>;

      template <class Factory>
      static constexpr auto _mk_opstate(void *storage, _receiver_t rcvr, Args &&...args)  //
        -> _any::_any_opstate_base
      {
        auto &make_sender = *__std::start_lifetime_as<Factory>(storage);
        using alloc_t     = decltype(choose_frame_allocator(get_env(rcvr)));
        auto alloc        = __frame_allocator_t<alloc_t>(choose_frame_allocator(get_env(rcvr)));
        return _any::_any_opstate_base(__in_place_from,
                                       std::allocator_arg,
                                       alloc,
                                       STDEXEC::connect,
                                       __invoke(make_sender, static_cast<Args &&>(args)...),
                                       static_cast<_receiver_t &&>(rcvr));
      }

      //! The curried arguments that will be passed to make_sender_ from inside make_opstate_.
      STDEXEC_ATTRIBUTE(no_unique_address)
      __tuple<Args...> args_;
      //! The type-erased operation state factory; it points to a function that knows the
      //! concrete type of the sender factory stored in make_sender_ so that it can
      //! construct the desired sender on demand and connect it to the given receiver. The
      //! expected arguments are the address of make_sender_, the _any_receiver_ref to
      //! connect the sender to, and the arguments to pass to make_sender_ to construct
      //! the sender.
      _any::_any_opstate_base (*make_opstate_)(void *, _receiver_t, Args &&...);
      //! Storage for the sender factory passed to our constructor template; make_opstate_ will
      //! reconstitute the actual factory from this bag-of-bytes with start_lifetime_as
      //! because it internally knows the concrete type of the user-provided sender
      //! factory. We're reserving 2 * sizeof(void *) bytes to permit the factory to be a
      //! pointer to member function, which usually requires two pointers.
      std::byte make_sender_[2 * sizeof(void *)]{};

     public:
      using sender_concept = sender_tag;

      template <__invocable<Args...> Factory>
        requires __not_decays_to<Factory, _function>          //
                && (STDEXEC_IS_TRIVIALLY_COPYABLE(Factory))   //
                && (sizeof(Factory) <= sizeof(make_sender_))  //
                && sender_to<__invoke_result_t<Factory, Args...>, _receiver_t>
      constexpr explicit _function(Args &&...args, Factory factory)
        noexcept(__nothrow_move_constructible<Args...>)
        : args_(static_cast<Args &&>(args)...)
        , make_opstate_(&_mk_opstate<Factory>)
      {
        std::memcpy(make_sender_, std::addressof(factory), sizeof(Factory));
      }

      //! this implementation of get_completion_signatures is taken directly from the
      //! equivalent function on any_sender_of
      template <class Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        static_assert(__decays_to_derived_from<Self, _function>);
        //! throw if Env does not contain the queries needed to type-erase the receiver:
        using _check_queries_t = __mfind_error<_any::_check_query_t<Queries, Env...>...>;
        if constexpr (__merror<_check_queries_t>)
          return __throw_compile_time_error(_check_queries_t());
        else
          return Sigs();
      }

      template <class Receiver>
      constexpr auto connect(Receiver rcvr) &&  //
        -> _opstate_t<Receiver>
      {
        auto factory = [this]<class RcvrRef>(RcvrRef rcvr)
        {
          return __apply(make_opstate_,
                         static_cast<__tuple<Args...> &&>(args_),
                         make_sender_,
                         static_cast<RcvrRef &&>(rcvr));
        };
        return _opstate_t<Receiver>{static_cast<Receiver &&>(rcvr), factory};
      }

      template <class Receiver>
        requires __std::copy_constructible<_function>
      constexpr auto connect(Receiver rcvr) const &  //
        -> _opstate_t<Receiver>
      {
        return _function(*this).connect(static_cast<Receiver &&>(rcvr));
      }
    };

    template <auto Types, template <class...> class Template, std::size_t... Is>
    consteval auto _canonicalize_splice(__indices<Is...>) noexcept
    {
      return Template<__msplice<Types[Is]>...>();
    }

    template <std::size_t Size>
    consteval auto _canonicalize_impl(__static_vector<__type_index, Size> types) noexcept
    {
      std::ranges::sort(types);
      auto const rest = std::ranges::unique(types);
      types.erase(rest.begin(), types.end());
      return types;
    }

    template <template <class...> class List, class... Types>
    consteval auto _canonicalize(List<Types...> *) noexcept
    {
      using types_t        = __static_vector<__type_index, sizeof...(Types)>;
      constexpr auto types = _func::_canonicalize_impl(types_t{__mtypeid<Types>...});
      return _func::_canonicalize_splice<types, List>(__make_indices<types.size()>());
    }

    template <class Sigs>
    using _canonical_t = decltype(_func::_canonicalize(static_cast<Sigs *>(nullptr)));

    //! Given a return type and a bool indicating whether the function is noexcept,
    //! compute the appropriate completion_signatures. The result is a set_value overload
    //! taking either Return&& or no args when Return is void, set_stopped, and, when the
    //! function type is not noexcept, set_error(std::exception_ptr)
    template <class Return, bool NoExcept>
    using _sigs_from_t = _canonical_t<__concat_completion_signatures_t<
      completion_signatures<__single_value_sig_t<Return>, set_stopped_t()>,
      __eptr_completion_unless_t<__mbool<NoExcept>>>>;

    template <class...>
    struct _make_function;

    template <class Return, class... Args>
    struct _make_function<Return(Args...)>
    {
      using type = _function<_sigs_from_t<Return, false>, queries<>, Args...>;
    };

    template <class Return, class... Args>
    struct _make_function<Return(Args...) noexcept>
    {
      using type = _function<_sigs_from_t<Return, true>, queries<>, Args...>;
    };

    template <class... Args, class... Sigs>
    struct _make_function<sender_tag(Args...), completion_signatures<Sigs...>>
    {
      using type = _function<_canonical_t<completion_signatures<Sigs...>>, queries<>, Args...>;
    };

    template <class Return, class... Args, class... Queries>
    struct _make_function<Return(Args...), queries<Queries...>>
    {
      using type =
        _function<_sigs_from_t<Return, false>, _canonical_t<queries<Queries...>>, Args...>;
    };

    template <class Return, class... Args, class... Queries>
    struct _make_function<Return(Args...) noexcept, queries<Queries...>>
    {
      using type =
        _function<_sigs_from_t<Return, true>, _canonical_t<queries<Queries...>>, Args...>;
    };

    template <class... Args, class... Sigs, class... Queries>
    struct _make_function<sender_tag(Args...), completion_signatures<Sigs...>, queries<Queries...>>
    {
      using type = _function<_canonical_t<completion_signatures<Sigs...>>,
                             _canonical_t<queries<Queries...>>,
                             Args...>;
    };
  }  // namespace _func

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
  template <class... Ts>
  using function = _func::_make_function<Ts...>::type;
}  // namespace experimental::execution

namespace exec = experimental::execution;
