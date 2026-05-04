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

// TODO: split this header into pieces
#include "any_sender_of.hpp"

#include <cstddef>
#include <cstring>
#include <memory>

// This file defines function<ReturnType(Arguments...)>, which is a
// type-erased sender that can complete with
//  - set_value(ReturnType)
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

  namespace _func
  {
    using namespace STDEXEC;

    template <class Receiver, class Sigs, class... Queries>
    class _func_op;

    // The concrete operation state resulting from connecting a function<...> to a concrete
    // receiver of type Receiver. This type manages a dynamically-allocated _derived_op instance,
    // which is the type-erased operation state resulting from connecting the type-erased sender
    // to an _any::_any_receiver_ref with the given completion signatures and queries.
    template <class Receiver, class... Sigs, class... Queries>
    class _func_op<Receiver, completion_signatures<Sigs...>, Queries...>
    {
      using _receiver_t =
        ::exec::_any::_any_receiver_ref<completion_signatures<Sigs...>, queries<Queries...>>;

      using _stop_token_t = stop_token_of_t<env_of_t<_receiver_t>>;

      // rcvr_ has to be initialized before op_ because our implementation of get_env
      // is empirically accessed during our constructor and depends on rcvr_ being initialized
      _any::_state<Receiver, _stop_token_t> rcvr_;
      _any::_any_opstate_base               op_;

     public:
      using operation_state_concept = operation_state_tag;

      template <class Factory>
      explicit constexpr _func_op(Receiver rcvr, Factory factory)
        : rcvr_(static_cast<Receiver &&>(rcvr))
        , op_(factory(_receiver_t(rcvr_)))
      {}

      _func_op(_func_op &&) = delete;

      constexpr ~_func_op() = default;

      constexpr void start() & noexcept
      {
        op_.start();
      }
    };

    // given the concrete receiver's environment, choose the frame allocator; first choice
    // is the result of get_frame_allocator(env), second choice is get_allocator(env), and
    // the default is std::allocator
    inline constexpr auto choose_frame_allocator =
      STDEXEC::__first_callable{get_frame_allocator,
                                get_allocator,
                                STDEXEC::__always{std::allocator<std::byte>()}};

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

      template <class Receiver>
      using _func_op_t = _func_op<Receiver, completion_signatures<Sigs...>, Queries...>;

      // The type-erased operation state factory; it points to a function that knows the concrete
      // type of the sender factory stored in make_sender_ so that it can construct the desired
      // sender on demand and connect it to the given receiver. The expected arguments are the
      // address of make_sender_, the _any_receiver_ref to connect the sender to, and the arguments
      // to pass to make_sender_ to construct the sender.
      _any::_any_opstate_base (*make_op_)(void *, _receiver_t, Args &&...);
      // Storage for the sender factory passed to our constructor template; make_op_ will
      // reconstitute the actual factory from this bag-of-bytes with start_lifetime_as
      // because it internally knows the concrete type of the user-provided sender factory.
      // We're reserving 2 * sizeof(void *) bytes to permit the factory to be a pointer to
      // member function, which usually requires two pointers.
      std::byte make_sender_[2 * sizeof(void *)]{};
      // The curried arguments that will be passed to make_sender_ from inside make_op_.
      STDEXEC_ATTRIBUTE(no_unique_address)
      STDEXEC::__tuple<Args...> args_;

     public:
      using sender_concept = SndrCncpt;

      template <STDEXEC::__invocable<Args...> Factory>
        requires STDEXEC::__not_decays_to<Factory, _func_impl>  //
              && (STDEXEC_IS_TRIVIALLY_COPYABLE(Factory))       //
              && (sizeof(Factory) <= sizeof(make_sender_))      //
              && STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>, _receiver_t>
      constexpr explicit _func_impl(Args &&...args, Factory factory)
        noexcept(STDEXEC::__nothrow_move_constructible<Args...>)
        : args_(static_cast<Args &&>(args)...)
      {
        using sender_t = __invoke_result_t<Factory, Args...>;

        std::memcpy(make_sender_, std::addressof(factory), sizeof(Factory));

        make_op_ = [](void *storage, _receiver_t rcvr, Args &&...args) -> _any::_any_opstate_base
        {
          auto &make_sender = *__std::start_lifetime_as<Factory>(storage);

          auto alloc = choose_frame_allocator(get_env(rcvr));

          return _any::_any_opstate_base(__in_place_from,
                                         std::allocator_arg,
                                         alloc,
                                         STDEXEC::connect,
                                         STDEXEC::__invoke(make_sender,
                                                           static_cast<Args &&>(args)...),
                                         static_cast<_receiver_t &&>(rcvr));
        };
      }

      template <__std::derived_from<_func_impl> Func>
        requires __not_decays_to<Func, _func_impl>
      constexpr _func_impl(Func &&other) noexcept(__nothrow_move_constructible<__tuple<Args...>>)
        : _func_impl(static_cast<_func_impl &&>(other))
      {}

      template <__std::derived_from<_func_impl> Func>
        requires __not_decays_to<Func, _func_impl> && __std::copy_constructible<__tuple<Args...>>
      constexpr _func_impl(Func const &other)
        noexcept(__nothrow_copy_constructible<__tuple<Args...>>)
        : _func_impl(static_cast<_func_impl const &>(other))
      {}

      // this implementation of get_completion_signatures is taken directly from
      // the equivalent function on any_sender_of
      template <class Self, class... Env>
      static consteval auto get_completion_signatures()
      {
        static_assert(__std::derived_from<std::remove_cvref_t<Self>, _func_impl>);

        // throw if Env does not contain the queries needed to type-erase the receiver:
        using _check_queries_t = __mfind_error<_any::_check_query_t<Queries, Env...>...>;
        if constexpr (__merror<_check_queries_t>)
        {
          return STDEXEC::__throw_compile_time_error(_check_queries_t{});
        }
        else
        {
          return completion_signatures<Sigs...>{};
        }
      }

      template <class Receiver>
      constexpr _func_op_t<Receiver> connect(Receiver rcvr) &&
      {
        return _func_op_t<Receiver>{static_cast<Receiver &&>(rcvr),
                                    [this]<class RcvrRef>(RcvrRef rcvr)
                                    {
                                      return STDEXEC::__apply(make_op_,
                                                              static_cast<__tuple<Args...> &&>(
                                                                args_),
                                                              make_sender_,
                                                              static_cast<RcvrRef &&>(rcvr));
                                    }};
      }

      template <class Receiver>
        requires STDEXEC::__std::copy_constructible<_func_impl>
      constexpr _func_op_t<Receiver> connect(Receiver rcvr) const &
      {
        return _func_impl(*this).connect(static_cast<Receiver &&>(rcvr));
      }
    };

    template <class Sigs>
    struct _canonical_fn;

    template <class... Sigs>
    struct _canonical_fn<completion_signatures<Sigs...>>
    {
      consteval auto operator()() const noexcept
      {
        constexpr auto make_sigs = []() noexcept
        {
          return __cmplsigs::__to_array(completion_signatures<Sigs...>{});
        };

        return __cmplsigs::__completion_sigs_from(make_sigs);
      }
    };

    template <class... Queries>
    struct _canonical_fn<queries<Queries...>>
    {
     private:
      // sort and unique the function types in Queries... into an array of __mtypeids
      static consteval auto get_sigs() noexcept
      {
        using sig_array_t = __static_vector<__type_index, sizeof...(Queries)>;
        auto sigs         = sig_array_t{__mtypeid<Queries>...};

        std::ranges::sort(sigs);

        auto const end = std::ranges::unique(sigs).begin();
        sigs.erase(end, sigs.end());

        return sigs;
      }

     public:
      consteval auto operator()() const noexcept
      {
        constexpr auto sigs = get_sigs();

        constexpr auto fn = [=]<std::size_t... Is>(__indices<Is...>)
        {
          return queries<__msplice<sigs[Is]>...>();
        };

        return fn(__make_indices<sigs.size()>());
      }
    };

    template <>
    struct _canonical_fn<queries<>>
    {
      consteval auto operator()() const noexcept
      {
        return queries<>();
      }
    };

    template <class Sigs>
    inline constexpr _canonical_fn<Sigs> _canonical{};

    template <class Sigs>
    using _canonical_t = decltype(_canonical<Sigs>());

    // Given a return type and a bool indicating whether the function is noexcept,
    // compute the appropriate completion_signatures. The result is a set_value
    // overload taking either Return&& or no args when Return is void, set_stopped,
    // and, when the function type is not noexcept, set_error(std::exception_ptr)
    template <class Return, bool NoExcept>
    using _sigs_from_t = _canonical_t<STDEXEC::__concat_completion_signatures_t<
      STDEXEC::completion_signatures<STDEXEC::__single_value_sig_t<Return>,
                                     STDEXEC::set_stopped_t()>,
      STDEXEC::__eptr_completion_unless_t<STDEXEC::__mbool<NoExcept>>>>;

    template <class Derived>
    struct _func_ops_crtp
    {
      template <class Base>
        requires std::same_as<Base, typename Derived::base>
      Derived &operator=(Base &&other) noexcept(STDEXEC::__nothrow_move_assignable<Base>)
      {
        Base::operator=(static_cast<Base &&>(other));
        return *static_cast<Derived *>(this);
      }

      template <class Base>
        requires std::same_as<Base, typename Derived::base> && STDEXEC::__copy_assignable<Base>
      Derived &operator=(Base const &other) noexcept(STDEXEC::__nothrow_copy_assignable<Base>)
      {
        Base::operator=(other);
        return *static_cast<Derived *>(this);
      }
    };
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
    : _func::_func_impl<STDEXEC::sender_tag(Args...), _func::_sigs_from_t<Return, false>, queries<>>
    , _func::_func_ops_crtp<function<Return(Args...)>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return, false>,
                                   queries<>>;

    using base::base;
  };

  template <class Return, class... Args>
  struct function<Return(Args...) noexcept>
    : _func::_func_impl<STDEXEC::sender_tag(Args...), _func::_sigs_from_t<Return, true>, queries<>>
    , _func::_func_ops_crtp<function<Return(Args...) noexcept>>
  {
    using base =
      _func::_func_impl<STDEXEC::sender_tag(Args...), _func::_sigs_from_t<Return, true>, queries<>>;

    using base::base;
  };

  template <class... Args, class Sigs>
    requires STDEXEC::__is_instance_of<Sigs, STDEXEC::completion_signatures>
  struct function<STDEXEC::sender_tag(Args...), Sigs>
    : _func::_func_impl<STDEXEC::sender_tag(Args...), _func::_canonical_t<Sigs>, queries<>>
    , _func::_func_ops_crtp<function<STDEXEC::sender_tag(Args...), Sigs>>
  {
    using base =
      _func::_func_impl<STDEXEC::sender_tag(Args...), _func::_canonical_t<Sigs>, queries<>>;

    using base::base;
  };

  template <class Return, class... Args, class... Queries>
  struct function<Return(Args...), queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return, false>,
                        _func::_canonical_t<queries<Queries...>>>
    , _func::_func_ops_crtp<function<Return(Args...), queries<Queries...>>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return, false>,
                                   _func::_canonical_t<queries<Queries...>>>;

    using base::base;
  };

  template <class Return, class... Args, class... Queries>
  struct function<Return(Args...) noexcept, queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return, true>,
                        _func::_canonical_t<queries<Queries...>>>
    , _func::_func_ops_crtp<function<Return(Args...) noexcept, queries<Queries...>>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return, true>,
                                   _func::_canonical_t<queries<Queries...>>>;

    using base::base;
  };

  template <class... Args, class... Sigs, class... Queries>
  struct function<STDEXEC::sender_tag(Args...),
                  STDEXEC::completion_signatures<Sigs...>,
                  queries<Queries...>>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_canonical_t<STDEXEC::completion_signatures<Sigs...>>,
                        _func::_canonical_t<queries<Queries...>>>
    , _func::_func_ops_crtp<function<STDEXEC::sender_tag(Args...),
                                     STDEXEC::completion_signatures<Sigs...>,
                                     queries<Queries...>>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_canonical_t<STDEXEC::completion_signatures<Sigs...>>,
                                   _func::_canonical_t<queries<Queries...>>>;

    using base::base;
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace std
{
  template <class... FuncArgs, template <class> class TQual, template <class> class UQual>
  struct basic_common_reference<exec::function<FuncArgs...>,
                                typename exec::function<FuncArgs...>::base,
                                TQual,
                                UQual>
  {
   private:
    using base = exec::function<FuncArgs...>::base;

   public:
    using type = common_reference_t<TQual<base>, UQual<base>>;
  };

  template <class... FuncArgs, template <class> class TQual, template <class> class UQual>
  struct basic_common_reference<typename exec::function<FuncArgs...>::base,
                                exec::function<FuncArgs...>,
                                TQual,
                                UQual>
  {
   private:
    using base = exec::function<FuncArgs...>::base;

   public:
    using type = common_reference_t<TQual<base>, UQual<base>>;
  };

  template <class... TArgs,
            class... UArgs,
            template <class> class TQual,
            template <class> class UQual>
  struct basic_common_reference<exec::function<TArgs...>, exec::function<UArgs...>, TQual, UQual>
  {
   private:
    using tbase = exec::function<TArgs...>::base;
    using ubase = exec::function<UArgs...>::base;

   public:
    using type = common_reference_t<TQual<tbase>, UQual<ubase>>;
  };
}  // namespace std
