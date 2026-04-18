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
#include "../stdexec/__detail/__receivers.hpp"
#include "../stdexec/__detail/__sender_concepts.hpp"

#include <exception>
#include <memory>
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
  namespace _func
  {
    using namespace STDEXEC;

    template <class Sig>
    struct _virt_completion;

    template <class Error>
    struct _virt_completion<set_error_t(Error)>
    {
      _virt_completion() = default;

      _virt_completion(_virt_completion&&) = delete;

      virtual void set_error(Error&& err) noexcept = 0;

     protected:
      ~_virt_completion() = default;
    };

    template <>
    struct _virt_completion<set_stopped_t()>
    {
      _virt_completion() = default;

      _virt_completion(_virt_completion&&) = delete;

      virtual void set_stopped() noexcept = 0;

     protected:
      ~_virt_completion() = default;
    };

    template <class... Values>
    struct _virt_completion<set_value_t(Values...)>
    {
      _virt_completion() = default;

      _virt_completion(_virt_completion&&) = delete;

      virtual void set_value(Values&&... values) noexcept = 0;

     protected:
      ~_virt_completion() = default;
    };

    template <class Sigs>
    struct _virt_completions;

    template <class... Sigs>
    struct _virt_completions<completion_signatures<Sigs...>> : virtual _virt_completion<Sigs>...
    {
      _virt_completions() = default;

      _virt_completions(_virt_completions&&) = delete;

     protected:
      ~_virt_completions() = default;
    };

    template <class Sig, class Derived>
    struct _func_rcvr_base;

    template <class Error, class Derived>
    struct _func_rcvr_base<set_error_t(Error), Derived>
    {
      void set_error(Error&& err) && noexcept
      {
        static_cast<Derived*>(this)->completer_->set_error(std::forward<Error>(err));
      }
    };

    template <class Derived>
    struct _func_rcvr_base<set_stopped_t(), Derived>
    {
      void set_stopped() && noexcept
      {
        static_cast<Derived*>(this)->completer_->set_stopped();
      }
    };

    template <class... Value, class Derived>
    struct _func_rcvr_base<set_value_t(Value...), Derived>
    {
      void set_value(Value&&... value) && noexcept
      {
        static_cast<Derived*>(this)->completer_->set_value(std::forward<Value>(value)...);
      }
    };

    template <class Sigs>
    class _func_rcvr;

    template <class... Sigs>
    class _func_rcvr<completion_signatures<Sigs...>>
      : public _func_rcvr_base<Sigs, _func_rcvr<completion_signatures<Sigs...>>>...
    {
      friend _func_rcvr_base<Sigs, _func_rcvr>...;

      using completer_t = _virt_completions<completion_signatures<Sigs...>>;

      completer_t* completer_;

     public:
      using receiver_concept = receiver_tag;

      explicit _func_rcvr(completer_t& completer) noexcept
        : completer_(std::addressof(completer))
      {}

      // TODO: get_env
    };

    struct _base_op
    {
      _base_op() = default;

      _base_op(_base_op&&) = delete;

      virtual ~_base_op() = default;

      virtual void start() & noexcept = 0;
    };

    template <class Sender, class Receiver>
    struct _derived_op : _base_op
    {
      explicit _derived_op(Sender&& sndr, Receiver rcvr)
        noexcept(std::is_nothrow_invocable_v<connect_t, Sender, Receiver>)
        : op_(connect(std::forward<Sender>(sndr), std::move(rcvr)))
      {}

      _derived_op(_derived_op&&) = delete;

      ~_derived_op() override = default;

      void start() & noexcept override
      {
        ::STDEXEC::start(op_);
      }

     private:
      connect_result_t<Sender, Receiver> op_;
    };

    template <class Sig, class Derived>
    struct _func_op_completion;

    template <class Error, class Derived>
    struct _func_op_completion<set_error_t(Error), Derived>
      : virtual _virt_completion<set_error_t(Error)>
    {
      void set_error(Error&& err) noexcept final
      {
        static_cast<Derived*>(this)->complete(set_error_t{}, std::forward<Error>(err));
      }
    };

    template <class Derived>
    struct _func_op_completion<set_stopped_t(), Derived> : virtual _virt_completion<set_stopped_t()>
    {
      void set_stopped() noexcept final
      {
        static_cast<Derived*>(this)->complete(set_stopped_t{});
      }
    };

    template <class... Value, class Derived>
    struct _func_op_completion<set_value_t(Value...), Derived>
      : virtual _virt_completion<set_value_t(Value...)>
    {
      void set_value(Value&&... value) noexcept final
      {
        static_cast<Derived*>(this)->complete(set_value_t{}, std::forward<Value>(value)...);
      }
    };

    template <class Receiver, class Sigs>
    class _func_op;

    template <class Receiver, class... Sigs>
    class _func_op<Receiver, completion_signatures<Sigs...>>
      : private _virt_completions<completion_signatures<Sigs...>>
      , private _func_op_completion<Sigs, _func_op<Receiver, completion_signatures<Sigs...>>>...
    {
      // TODO: use get_frame_allocator(get_env(rcvr_)) to allocate and destroy this
      std::unique_ptr<_base_op> op_;
      [[no_unique_address]]
      Receiver rcvr_;

      friend _func_op_completion<Sigs, _func_op>...;

      template <class CPO, class... Arg>
      void complete(CPO cpo, Arg&&... arg) noexcept
      {
        std::move(cpo)(std::move(rcvr_), std::forward<Arg>(arg)...);
      }

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

    template <class Args, class Sigs, class Env>
    class _func_impl;

    template <class SndrCncpt, class... Args, class... Sigs, class Env>
    class _func_impl<SndrCncpt(Args...), completion_signatures<Sigs...>, Env>
    {
      _base_op* (*factory_)(_func_rcvr<completion_signatures<Sigs...>>, Args&&...);
      [[no_unique_address]]
      std::tuple<Args...> args_;

     public:
      using sender_concept = SndrCncpt;

      template <STDEXEC::__callable<Args...> Factory>
        requires STDEXEC::__not_decays_to<Factory, _func_impl>  //
              && std::constructible_from<Factory>               //
              && STDEXEC::__callable<Factory, Args...>
      //&& STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>,
      //_func_rcvr<completion_signatures<Sigs...>>>
      explicit(sizeof...(Args) == 0) _func_impl(Args&&... args, Factory&& factory)
        noexcept((std::is_nothrow_constructible_v<Args, Args> && ...))
        : args_(std::forward<Args>(args)...)
      {
        using sender_t   = std::invoke_result_t<Factory, Args...>;
        using receiver_t = _func_rcvr<completion_signatures<Sigs...>>;

        using op_t = _derived_op<sender_t, receiver_t>;

        factory_ = [](receiver_t rcvr, Args&&... args) -> _base_op*
        {
          Factory factory;
          // TODO: query rcvr for a frame allocator and use it
          return new op_t(factory(std::forward<Args>(args)...), std::move(rcvr));
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

    template <class Sig>
    using _sigs_from_t = _sigs_from<Sig>::type;
  }  // namespace _func

  // TODO: think about environment forwarding
  template <class...>
  class function;

  template <class Return, class... Args>
  class function<Return(Args...)>
    : public _func::_func_impl<STDEXEC::sender_tag(Args...),
                               _func::_sigs_from_t<Return(Args...)>,
                               STDEXEC::env<>>
  {
    using base = _func::_func_impl<STDEXEC::sender_tag(Args...),
                                   _func::_sigs_from_t<Return(Args...)>,
                                   STDEXEC::env<>>;

    using base::base;
  };
}  // namespace experimental::execution

namespace exec = experimental::execution;
