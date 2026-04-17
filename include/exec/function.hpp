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

  // TODO: think about environment forwarding
  template <class T, class Env = STDEXEC::env<>>
  struct function;

  template <class R>
  struct completer
  {
    completer() = default;

    virtual void set_value(R&& value) noexcept = 0;

    virtual void set_error(std::exception_ptr err) noexcept = 0;

    virtual void set_stopped() noexcept = 0;

   protected:
    ~completer() = default;
  };

  template <>
  struct completer<void>
  {
    completer() = default;

    virtual void set_value() noexcept = 0;

    virtual void set_error(std::exception_ptr err) noexcept = 0;

    virtual void set_stopped() noexcept = 0;

   protected:
    ~completer() = default;
  };

  template <class R>
  struct function_receiver
  {
    using receiver_concept = STDEXEC::receiver_tag;

    void set_value(R&& value) noexcept
    {
      completer_->set_value(std::forward<R>(value));
    }

    void set_error(std::exception_ptr err) noexcept
    {
      completer_->set_error(std::move(err));
    }

    void set_stopped() noexcept
    {
      completer_->set_stopped();
    }

    completer<R>* completer_;
  };

  template <>
  struct function_receiver<void>
  {
    using receiver_concept = STDEXEC::receiver_tag;

    void set_value() noexcept
    {
      completer_->set_value();
    }

    void set_error(std::exception_ptr err) noexcept
    {
      completer_->set_error(std::move(err));
    }

    void set_stopped() noexcept
    {
      completer_->set_stopped();
    }

    completer<void>* completer_;
  };

  template <class Return>
  struct function_completions
  {
    template <class Sender, class /*Env*/>
    static consteval STDEXEC::completion_signatures<STDEXEC::set_value_t(Return),
                                                    STDEXEC::set_error_t(std::exception_ptr),
                                                    STDEXEC::set_stopped_t()>
    get_completion_signatures()
    {
      return {};
    }
  };

  template <>
  struct function_completions<void>
  {
    template <class Sender, class /*Env*/>
    static consteval STDEXEC::completion_signatures<STDEXEC::set_value_t(),
                                                    STDEXEC::set_error_t(std::exception_ptr),
                                                    STDEXEC::set_stopped_t()>
    get_completion_signatures()
    {
      return {};
    }
  };

  struct base_operation
  {
    base_operation()                 = default;
    base_operation(base_operation&&) = delete;
    virtual ~base_operation()        = default;

    virtual void start() & noexcept = 0;
  };

  template <class R, class Receiver>
  struct operation_storage : completer<R>
  {
    explicit operation_storage(Receiver rcvr) noexcept
      : receiver_(std::move(rcvr))
    {}

    void set_value(R&& value) noexcept final
    {
      STDEXEC::set_value(std::move(receiver_), std::forward<R>(value));
    }

    Receiver receiver_;
  };

  template <class Receiver>
  struct operation_storage<void, Receiver> : completer<void>
  {
    explicit operation_storage(Receiver rcvr) noexcept
      : receiver_(std::move(rcvr))
    {}

    void set_value() noexcept final
    {
      STDEXEC::set_value(std::move(receiver_));
    }

    Receiver receiver_;
  };

  template <class R, class Receiver>
  struct operation : operation_storage<R, Receiver>
  {
    using operation_state_concept = STDEXEC::operation_state_tag;

    template <class Factory>
    operation(Receiver rcvr, Factory factory)
      : operation_storage<R, Receiver>{std::move(rcvr)}
      , op_(factory(function_receiver<R>(this)))
    {}

    void start() & noexcept
    {
      op_->start();
    }

   private:
    std::unique_ptr<base_operation> op_;

    void set_error(std::exception_ptr err) noexcept final
    {
      STDEXEC::set_error(std::move(this->receiver_), std::move(err));
    }

    void set_stopped() noexcept final
    {
      STDEXEC::set_stopped(std::move(this->receiver_));
    }
  };

  // consider:
  //
  //   template <class R, class... Args>
  //   struct function<R(Args...) noexcept> {};
  //
  // to declare no error channel
  //
  // we allocate in connect, which could throw, but that just means connect
  // can't be noexcept; it doesn't mean we have to have an error channel after
  // we successfully connect...
  template <class R, class... Args, class Env>
    requires((std::movable<Args> || std::is_reference_v<Args>) && ...)
  struct function<R(Args...), Env> : function_completions<R>
  {
    using sender_concept = STDEXEC::sender_tag;

    template <STDEXEC::__callable<Args...> Factory>
      requires STDEXEC::__not_decays_to<Factory, function>  //
            && std::constructible_from<Factory>             //
            && STDEXEC::__callable<Factory, Args...>
            && STDEXEC::sender_to<STDEXEC::__invoke_result_t<Factory, Args...>,
                                  function_receiver<R>>
    explicit(sizeof...(Args) == 0) function(Args&&... args, Factory&& factory)
      noexcept((std::is_nothrow_constructible_v<Args, Args> && ...))
      : args_(std::forward<Args>(args)...)
    {
      using sender_t = std::invoke_result_t<Factory, Args...>;

      struct derived_operation : base_operation
      {
        explicit derived_operation(sender_t&& sndr, function_receiver<R> rcvr)  // TODO noexcept
          : op_(STDEXEC::connect(std::forward<sender_t>(sndr), std::move(rcvr)))
        {}

        ~derived_operation() override = default;

        void start() & noexcept override
        {
          STDEXEC::start(op_);
        }

       private:
        STDEXEC::connect_result_t<sender_t, function_receiver<R>> op_;
      };

      factory_ = [](function_receiver<R> rcvr, Args&&... args) -> base_operation*
      {
        Factory factory;
        // TODO: query rcvr for a frame allocator and use it
        return new derived_operation(factory(std::forward<Args>(args)...), std::move(rcvr));
      };
    }

    template <class Self, STDEXEC::receiver Receiver>
    auto connect(this Self&& sender, Receiver receiver) -> operation<R, Receiver>
    {
      return operation<R, Receiver>(std::move(receiver),
                                    [&](function_receiver<R> rcvr)
                                    {
                                      return std::apply(
                                        [&](Args&&... args)
                                        {
                                          return sender.factory_(std::move(rcvr),
                                                                 std::forward<Args>(args)...);
                                        },
                                        std::forward<Self>(sender).args_);
                                    });
    }

   private:
    base_operation* (*factory_)(function_receiver<R>, Args&&...);
    [[no_unique_address]]
    std::tuple<Args...> args_;
  };

}  // namespace experimental::execution

namespace exec = experimental::execution;
