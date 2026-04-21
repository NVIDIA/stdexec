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
    using namespace STDEXEC;

    template <bool NoThrow, class Env, class Query, class... Args>
    concept _conditionally_nothrow_queryable_with =
      (!NoThrow && __queryable_with<Env const &, Query, Args...>)
      || __nothrow_queryable_with<Env const &, Query, Args...>;

    template <class Expected, bool NoThrow, class Env, class Query, class... Args>
    concept _query_result_convertible_to =
      (!NoThrow && std::is_convertible_v<__query_result_t<Env const &, Query, Args...>, Expected>)
      || std::is_nothrow_convertible_v<__query_result_t<Env const &, Query, Args...>, Expected>;

    template <class Sig>
    struct query;

    template <class Return, class Query, class... Args, bool NoThrow>
    struct query<Return(Query, Args...) noexcept(NoThrow)>
    {
     protected:
      template <class Env>
        requires _conditionally_nothrow_queryable_with<NoThrow, Env, Query, Args...>
              && _query_result_convertible_to<Return, NoThrow, Env, Query, Args...>
      static Return query_delegate(Env const &env, Query query, Args &&...args) noexcept(NoThrow)
      {
        return __query<Query>()(env, std::forward<Args>(args)...);
      }
    };
  }  // namespace _qry_detail

  template <class... Queries>
  struct queries : _qry_detail::query<Queries>...
  {};

  namespace _func
  {
    using namespace STDEXEC;

    template <class Sig>
    struct _virt_completion;

    template <class CPO, class... Args>
    struct _virt_completion<CPO(Args...)>
    {
      constexpr _virt_completion() = default;

      _virt_completion(_virt_completion &&) = delete;

      constexpr virtual void complete(CPO, Args &&...) noexcept = 0;

     protected:
      constexpr ~_virt_completion() = default;
    };

    template <class... Queries>
    struct _env_of_queries
    {};

    template <class Return, class Query, class... Args, bool NoThrow>
    struct _env_of_queries<Return(Query, Args...) noexcept(NoThrow)>
    {
      virtual Return query(Query query, Args &&...args) const noexcept(NoThrow) = 0;
    };

    template <class Return, class Query, class... Args, bool NoThrow, class... Queries>
    struct _env_of_queries<Return(Query, Args...) noexcept(NoThrow), Queries...>
      : _env_of_queries<Queries...>
    {
      _env_of_queries() = default;

      _env_of_queries(_env_of_queries &&) = delete;

      using _env_of_queries<Queries...>::query;

      virtual Return query(Query query, Args &&...args) const noexcept(NoThrow) = 0;

     protected:
      ~_env_of_queries() = default;
    };

    template <class Base, class Derived, class... Queries>
    struct _delegate_env_base;

    template <class Base, class Derived>
    struct _delegate_env_base<Base, Derived> : public Base
    {};

    template <class Base,
              class Derived,
              class Return,
              class Query,
              class... Args,
              bool NoThrow,
              class... Queries>
    struct _delegate_env_base<Base, Derived, Return(Query, Args...) noexcept(NoThrow), Queries...>
      : _delegate_env_base<Base, Derived, Queries...>
    {
      using query_base = _qry_detail::query<Return(Query, Args...) noexcept>;

      Return query(Query qry, Args &&...args) const noexcept(NoThrow) final
      {
        auto &delegate = **static_cast<Derived const *>(this);
        return __query<Query>()(delegate, std::forward<Args>(args)...);
      }
    };

    template <class Queries>
    struct _delegate_env;

    template <>
    struct _delegate_env<queries<>>
      : _delegate_env_base<_env_of_queries<>, _delegate_env<queries<>>>
    {
      using delegate_t = _env_of_queries<>;

      explicit _delegate_env(delegate_t const &delegate) noexcept
        : delegate_(std::addressof(delegate))
      {}

     private:
      delegate_t const *delegate_;

      template <class, class, class...>
      friend class _delegte_env_base;

      delegate_t const &operator*() const noexcept
      {
        return *delegate_;
      }
    };

    template <class... Queries>
    struct _delegate_env<queries<Queries...>>
      : _delegate_env_base<_env_of_queries<Queries...>,
                           _delegate_env<queries<Queries...>>,
                           Queries...>
    {
      using delegate_t = _env_of_queries<Queries...>;

      explicit _delegate_env(delegate_t const &delegate) noexcept
        : delegate_(std::addressof(delegate))
      {}

      //using _delegate_env_base<_env_of_queries<queries
     private:
      delegate_t const *delegate_;

      template <class, class, class...>
      friend class _delegate_env_base;

      delegate_t const &operator*() const noexcept
      {
        return *delegate_;
      }
    };

    template <class Sigs, class Queries>
    struct _virt_completions;

    template <class... Sigs, class... Queries>
    struct _virt_completions<completion_signatures<Sigs...>, queries<Queries...>>
      : _virt_completion<Sigs>...
      , _env_of_queries<Queries...>
    {
      constexpr _virt_completions() = default;

      _virt_completions(_virt_completions &&) = delete;

      using _virt_completion<Sigs>::complete...;

      constexpr _delegate_env<queries<Queries...>> get_env() const noexcept
      {
        return _delegate_env<queries<Queries...>>(*this);
      }

     protected:
      constexpr ~_virt_completions() = default;
    };

    template <class Sigs, class Queries>
    class _func_rcvr;

    template <class... Sigs, class Queries>
    class _func_rcvr<completion_signatures<Sigs...>, Queries>
    {
      using completer_t = _virt_completions<completion_signatures<Sigs...>, Queries>;

      completer_t *completer_;

     public:
      using receiver_concept = receiver_tag;

      constexpr explicit _func_rcvr(completer_t &completer) noexcept
        : completer_(std::addressof(completer))
      {}

      template <class Error>
      constexpr void set_error(Error &&err) && noexcept
        requires requires { this->completer_->complete(set_error_t{}, std::forward<Error>(err)); }
      {
        this->completer_->complete(set_error_t{}, std::forward<Error>(err));
      }

      constexpr void set_stopped() && noexcept
        requires requires { this->completer_->complete(set_stopped_t{}); }
      {
        this->completer_->complete(set_stopped_t{});
      }

      template <class... Values>
      constexpr void set_value(Values &&...values) && noexcept
        requires requires {
          this->completer_->complete(set_value_t{}, std::forward<Values>(values)...);
        }
      {
        this->completer_->complete(set_value_t{}, std::forward<Values>(values)...);
      }

      constexpr auto get_env() const noexcept -> _delegate_env<Queries>
      {
        return STDEXEC::get_env(*completer_);
      }
    };

    struct _base_op
    {
      constexpr _base_op() = default;

      _base_op(_base_op &&) = delete;

      constexpr virtual ~_base_op() = default;

      constexpr virtual void start() & noexcept = 0;
    };

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

      static constexpr void operator delete(_derived_op *p, std::destroying_delete_t)
      {
        using traits = std::allocator_traits<Allocator>::template rebind_traits<_derived_op>;

        typename traits::allocator_type alloc = std::move(p->alloc_);
        traits::destroy(alloc, p);
        traits::deallocate(alloc, p, 1);
      }

     private:
      connect_result_t<Sender, Receiver> op_;
      [[no_unique_address]]
      Allocator alloc_;
    };

    template <class Base, class Derived, class... Sigs>
    struct _func_op_completion;

    template <class Base, class Derived>
    struct _func_op_completion<Base, Derived> : Base
    {};

    template <class Base, class Derived, class CPO, class... Args, class... Sigs>
    struct _func_op_completion<Base, Derived, CPO(Args...), Sigs...>
      : _func_op_completion<Base, Derived, Sigs...>
    {
      void complete(CPO, Args &&...args) noexcept final
      {
        // This seems like it ought to be true, but it fails...
        //
        // Some testing shows it's being evaluated when Derive is incomplete
        // during constraint satisfaction testing.
        //
        // static_assert(std::derived_from<_func_op_completion, Derived>);
        auto &rcvr = static_cast<Derived *>(this)->rcvr_;
        CPO{}(std::move(rcvr), std::forward<Args>(args)...);
      }
    };

    template <class Base, class Derived, class Queries>
    struct _func_op_queries;

    template <class Base, class Derived>
    struct _func_op_queries<Base, Derived, queries<>> : Base
    {};

    template <class Base,
              class Derived,
              class Return,
              class Query,
              class... Args,
              bool NoThrow,
              class... Queries>
    struct _func_op_queries<Base,
                            Derived,
                            queries<Return(Query, Args...) noexcept(NoThrow), Queries...>>
      : _func_op_queries<Base, Derived, queries<Queries...>>
    {
      Return query(Query, Args &&...args) const noexcept(NoThrow) final
      {
        using delegate_t = _qry_detail::query<Return(Args...) noexcept(NoThrow)>;

        auto const &rcvr = static_cast<Derived const *>(this)->rcvr_;
        return __query<Query>()(STDEXEC::get_env(rcvr), std::forward<Args>(args)...);
      }
    };

    template <class Receiver, class Sigs, class Queries>
    class _func_op;

    template <class Receiver, class... Sigs, class Queries>
    class _func_op<Receiver, completion_signatures<Sigs...>, Queries>
      : _func_op_completion<
          _func_op_queries<_virt_completions<completion_signatures<Sigs...>, Queries>,
                           _func_op<Receiver, completion_signatures<Sigs...>, Queries>,
                           Queries>,
          _func_op<Receiver, completion_signatures<Sigs...>, Queries>,
          Sigs...>
    {
      [[no_unique_address]]
      Receiver                  rcvr_;
      std::unique_ptr<_base_op> op_;

      template <class B, class D, class... S>
      friend struct _func_op_completion;

      template <class, class, class>
      friend struct _func_op_queries;

     public:
      using operation_state_concept = operation_state_tag;

      template <class Factory>
      constexpr _func_op(Receiver rcvr, Factory factory)
        : rcvr_(std::move(rcvr))
        , op_(factory(_func_rcvr<completion_signatures<Sigs...>, Queries>(*this)))
      {}

      _func_op(_func_op &&) = delete;

      constexpr ~_func_op() = default;

      constexpr void start() & noexcept
      {
        op_->start();
      }
    };

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

    template <class SndrCncpt, class... Args, class... Sigs, class... Queries>
    class _func_impl<SndrCncpt(Args...), completion_signatures<Sigs...>, queries<Queries...>>
    {
      std::unique_ptr<_base_op> (
        *factory_)(_func_rcvr<completion_signatures<Sigs...>, queries<Queries...>>, Args &&...);
      [[no_unique_address]]
      std::tuple<Args...> args_;

     public:
      using sender_concept = SndrCncpt;

      template <STDEXEC::__callable<Args...> Factory>
        requires STDEXEC::__not_decays_to<Factory, _func_impl>  //
              && std::constructible_from<Factory>               //
              && STDEXEC::__callable<Factory, Args...>
              && STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>,
                                    _func_rcvr<completion_signatures<Sigs...>, queries<Queries...>>>
      constexpr explicit(sizeof...(Args) == 0) _func_impl(Args &&...args, Factory &&factory)
        noexcept((std::is_nothrow_constructible_v<Args, Args> && ...))
        : args_(std::forward<Args>(args)...)
      {
        using sender_t   = std::invoke_result_t<Factory, Args...>;
        using receiver_t = _func_rcvr<completion_signatures<Sigs...>, queries<Queries...>>;

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

      template <class Sender, class RcvrEnv>
      static consteval auto get_completion_signatures() noexcept
      {
        static_assert(STDEXEC_IS_BASE_OF(_func_impl, __decay_t<Sender>));
        //static_assert(std::constructible_from<Env, RcvrEnv const &>);

        //Env env{RcvrEnv{}};

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

      template <class Receiver>
      constexpr _func_op<Receiver, completion_signatures<Sigs...>, queries<Queries...>>
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

  // TODO: think about environment forwarding
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
