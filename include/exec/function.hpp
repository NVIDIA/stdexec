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

  namespace _func
  {
    using namespace STDEXEC;

    template <class Sig>
    struct _virt_completion;

    template <class CPO, class... Args>
    struct _virt_completion<CPO(Args...)>
    {
      _virt_completion() = default;

      _virt_completion(_virt_completion&&) = delete;

      virtual void complete(CPO, Args&&...) noexcept = 0;

     protected:
      ~_virt_completion() = default;
    };

    template <class Sigs>
    struct _virt_completions;

    template <class... Sigs>
    struct _virt_completions<completion_signatures<Sigs...>> : _virt_completion<Sigs>...
    {
      _virt_completions() = default;

      _virt_completions(_virt_completions&&) = delete;

      using _virt_completion<Sigs>::complete...;

     protected:
      ~_virt_completions() = default;
    };

    template <class Sigs>
    class _func_rcvr;

    template <class... Sigs>
    class _func_rcvr<completion_signatures<Sigs...>>
    {
      using completer_t = _virt_completions<completion_signatures<Sigs...>>;

      completer_t* completer_;

     public:
      using receiver_concept = receiver_tag;

      explicit _func_rcvr(completer_t& completer) noexcept
        : completer_(std::addressof(completer))
      {}

      template <class Error>
      void set_error(Error&& err) && noexcept
        requires requires { this->completer_->complete(set_error_t{}, std::forward<Error>(err)); }
      {
        this->completer_->complete(set_error_t{}, std::forward<Error>(err));
      }

      void set_stopped() && noexcept
        requires requires { this->completer_->complete(set_stopped_t{}); }
      {
        this->completer_->complete(set_stopped_t{});
      }

      template <class... Values>
      void set_value(Values&&... values) && noexcept
        requires requires {
          this->completer_->complete(set_value_t{}, std::forward<Values>(values)...);
        }
      {
        this->completer_->complete(set_value_t{}, std::forward<Values>(values)...);
      }

      // TODO: get_env
    };

    struct _base_op
    {
      _base_op() = default;

      _base_op(_base_op&&) = delete;

      virtual ~_base_op() = default;

      virtual void start() & noexcept = 0;
    };

    template <class Sender, class Receiver, class Allocator>
    struct _derived_op : _base_op
    {
      explicit _derived_op(Sender&& sndr, Receiver rcvr, Allocator const & alloc)
        noexcept(std::is_nothrow_invocable_v<connect_t, Sender, Receiver>)
        : op_(connect(std::forward<Sender>(sndr), std::move(rcvr)))
        , alloc_(alloc)
      {}

      _derived_op(_derived_op&&) = delete;

      ~_derived_op() final = default;

      void start() & noexcept final
      {
        ::STDEXEC::start(op_);
      }

      static constexpr void operator delete(_derived_op* p, std::destroying_delete_t)
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
      void complete(CPO, Args&&... args) noexcept final
      {
        auto& rcvr = static_cast<Derived*>(this)->rcvr_;
        CPO{}(std::move(rcvr), std::forward<Args>(args)...);
      }
    };

    template <class Receiver, class Sigs>
    class _func_op;

    template <class Receiver, class... Sigs>
    class _func_op<Receiver, completion_signatures<Sigs...>>
      : private _func_op_completion<_virt_completions<completion_signatures<Sigs...>>,
                                    _func_op<Receiver, completion_signatures<Sigs...>>,
                                    Sigs...>
    {
      std::unique_ptr<_base_op> op_;
      [[no_unique_address]]
      Receiver rcvr_;

      template <class B, class D, class... S>
      friend struct _func_op_completion;

     public:
      using operation_state_concept = operation_state_tag;

      template <class Factory>
      _func_op(Receiver rcvr, Factory factory)
        : rcvr_(std::move(rcvr))
        , op_(factory(_func_rcvr<completion_signatures<Sigs...>>(*this))){};

      _func_op(_func_op&&) = delete;

      ~_func_op() = default;

      void start() & noexcept
      {
        op_->start();
      }
    };

    template <class Env>
    constexpr auto choose_frame_allocator(Env const & env) noexcept
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

    template <class Args, class Sigs, class Env>
    class _func_impl;

    template <class SndrCncpt, class... Args, class... Sigs, class Env>
    class _func_impl<SndrCncpt(Args...), completion_signatures<Sigs...>, Env>
    {
      std::unique_ptr<_base_op> (*factory_)(_func_rcvr<completion_signatures<Sigs...>>, Args&&...);
      [[no_unique_address]]
      std::tuple<Args...> args_;

     public:
      using sender_concept = SndrCncpt;

      template <STDEXEC::__callable<Args...> Factory>
        requires STDEXEC::__not_decays_to<Factory, _func_impl>  //
              && std::constructible_from<Factory>               //
              && STDEXEC::__callable<Factory, Args...>
              && STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>,
                                    _func_rcvr<completion_signatures<Sigs...>>>
      explicit(sizeof...(Args) == 0) _func_impl(Args&&... args, Factory&& factory)
        noexcept((std::is_nothrow_constructible_v<Args, Args> && ...))
        : args_(std::forward<Args>(args)...)
      {
        using sender_t   = std::invoke_result_t<Factory, Args...>;
        using receiver_t = _func_rcvr<completion_signatures<Sigs...>>;

        using op_t = _derived_op<sender_t, receiver_t, std::allocator<std::byte>>;

        factory_ = [](receiver_t rcvr, Args&&... args) -> std::unique_ptr<_base_op>
        {
          using traits = std::allocator_traits<decltype(choose_frame_allocator(
            get_env(rcvr)))>::template rebind_traits<op_t>;

          Factory factory;

          typename traits::allocator_type alloc(choose_frame_allocator(get_env(rcvr)));

          auto* op = traits::allocate(alloc, 1);

          __scope_guard guard{[&]() noexcept { traits::deallocate(alloc, op, 1); }};

          traits::construct(alloc,
                            op,
                            factory(std::forward<Args>(args)...),
                            std::move(rcvr),
                            alloc);

          guard.__dismiss();

          return std::unique_ptr<_base_op>(op);
        };
      }

      template <class Sender, class /* Env */>
      static consteval completion_signatures<Sigs...> get_completion_signatures() noexcept
      {
        // TODO: validate that the Env passed here is compatible with the class-level Env
        return {};
      }

      template <class Receiver>
      constexpr _func_op<Receiver, completion_signatures<Sigs...>> connect(Receiver rcvr)
      {
        return {std::move(rcvr),
                [&, this](auto rcvr)
                {
                  return std::apply(
                    [&](Args&&... args)
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
                        STDEXEC::env<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
                                   STDEXEC::env<>>;

    using base::base;
  };

  template <class... Args, class Sigs>
    requires STDEXEC::__is_instance_of<Sigs, STDEXEC::completion_signatures>
  struct function<STDEXEC::sender_tag(Args...), Sigs>
    : _func::_func_impl<STDEXEC::sender_tag(Args...), Sigs, STDEXEC::env<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...), Sigs, STDEXEC::env<>>;

    using base::base;
  };

  template <class Return, class... Args, bool NoThrow, class Env>
    requires STDEXEC::__is_not_instance_of<Env, STDEXEC::completion_signatures>
  struct function<Return(Args...) noexcept(NoThrow), Env>
    : _func::_func_impl<STDEXEC::sender_tag(Args...),
                        _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
                        Env>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...) noexcept(NoThrow)>,
                                   Env>;

    using base::base;
  };

  template <class... Args, class... Sigs, class Env>
    requires STDEXEC::__is_not_instance_of<Env, STDEXEC::completion_signatures>
  struct function<STDEXEC::sender_tag(Args...), STDEXEC::completion_signatures<Sigs...>, Env>
    : _func::_func_impl<STDEXEC::sender_tag(Args...), STDEXEC::completion_signatures<Sigs...>, Env>
  {
    using base =
      _func::_func_impl<STDEXEC::sender_tag(Args...), STDEXEC::completion_signatures<Sigs...>, Env>;

    using base::base;
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;
