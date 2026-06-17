/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

#include "__basic_sender.hpp"
#include "__completion_signatures_of.hpp"
#include "__diagnostics.hpp"
#include "__meta.hpp"
#include "__queries.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__transform_completion_signatures.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.then]
  namespace __then
  {
    struct __then_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Child>(__ignore, __ignore, _Child const & __child) noexcept
      {
        return __sync_attrs{__child};
      };

      template <class _Fun>
      static consteval auto __transform_value_completion() noexcept
      {
        return []<class... _Args>()
        {
          if constexpr (__nothrow_invocable<_Fun, _Args...>)
          {
            return completion_signatures<__single_value_sig_t<__invoke_result_t<_Fun, _Args...>>>();
          }
          else if constexpr (__invocable<_Fun, _Args...>)
          {
            return completion_signatures<__single_value_sig_t<__invoke_result_t<_Fun, _Args...>>,
                                         set_error_t(std::exception_ptr)>();
          }
          else
          {
            return STDEXEC::__throw_compile_time_error(
              __callable_error_t<then_t, _Fun, _Args...>());
          }
        };
      }

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, then_t>);
        using __fn_t = __decay_t<__data_of<_Sender>>;
        return STDEXEC::__transform_completion_signatures(
          STDEXEC::get_completion_signatures<__child_of<_Sender>, _Env...>(),
          __transform_value_completion<__fn_t>());
      };

      struct __complete_fn
      {
        template <class _Tag, class _State, class... _Args>
        STDEXEC_ATTRIBUTE(host, device)
        constexpr void operator()(__ignore, _State& __state, _Tag, _Args&&... __args) const noexcept
        {
          if constexpr (__same_as<_Tag, set_value_t>)
          {
            STDEXEC::__set_value_from(static_cast<_State&&>(__state).__rcvr_,
                                      static_cast<_State&&>(__state).__data_,
                                      static_cast<_Args&&>(__args)...);
          }
          else
          {
            _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
          }
        }
      };

      static constexpr auto __complete = __complete_fn{};
    };
  }  // namespace __then

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //! @brief A pipeable sender adaptor that transforms a predecessor sender's
  //!        value completion by invoking a callable on the values it produces.
  //!
  //! @c then maps the value channel of a sender through a function while forwarding
  //! the error and stopped channels unchanged. It is the asynchronous analogue of
  //! applying <tt>std::invoke</tt> to the result of a synchronous computation, and
  //! is the most common sender adaptor in practice; most sender pipelines contain at
  //! least one @c then.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::then(sndr, f);   // direct invocation
  //! auto s2 = sndr | stdexec::then(f);  // pipe syntax
  //! @endcode
  //!
  //! The two forms are expression-equivalent. See [exec.then] in the
  //! C++26 working draft for the normative specification.
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // one or more value completions
  //! set_error_t(Es)...               // zero or more error completions
  //! set_stopped_t()                  // optional stopped completion
  //! @endcode
  //!
  //! the sender produced by <tt>then(sndr, f)</tt> has completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(R)                   // R = decltype(std::invoke(f, Vs...))
  //!                                  // (or set_value_t() when R is void)
  //! set_error_t(Es)...               // forwarded unchanged from sndr
  //! set_error_t(std::exception_ptr)  // added when invoking f may throw
  //! set_stopped_t()                  // forwarded unchanged from sndr
  //! @endcode
  //!
  //! If @c sndr has multiple value completions, @c f must be invocable with every
  //! one of them; otherwise the program is ill-formed and the diagnostic surfaces
  //! at the point where the resulting sender is connected to a receiver. The
  //! resulting value-completion arity is always one: each distinct return type
  //! @c R from invoking @c f contributes a @c set_value_t(R) overload.
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c f throws, the exception is delivered through
  //! @c set_error_t(std::exception_ptr) on the resulting sender. When @c f is
  //! @c noexcept for every value-argument pack of @c sndr, no additional
  //! @c std::exception_ptr error completion is added.
  //!
  //! **Cancellation.**
  //!
  //! @c then does not interact with the receiver's stop token. When @c sndr
  //! completes with @c set_stopped, @c f is not invoked and the stopped
  //! completion is forwarded to the downstream receiver.
  //!
  //! **Example.**
  //!
  //! @code{.cpp}
  //! #include <stdexec/execution.hpp>
  //! #include <cassert>
  //!
  //! int main() {
  //!   using namespace stdexec;
  //!
  //!   auto sndr = just(21)
  //!             | then([](int x) { return x * 2; })
  //!             | then([](int x) { return x + 1; });
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == 43);
  //! }
  //! @endcode
  //!
  //! @see stdexec::upon_error  — adapt the error channel
  //! @see stdexec::upon_stopped — adapt the stopped channel
  //! @see stdexec::let_value   — adapt the value channel with a sender-returning function
  struct then_t
  {
    //! @brief Construct a sender that adapts @c __sndr by invoking @c __fun with
    //!        each value-completion argument pack it produces.
    //!
    //! @tparam _Sender A type satisfying the @c stdexec::sender concept.
    //! @tparam _Fun    A decayed, move-constructible callable type
    //!                 (satisfying the internal <tt>__movable_value</tt> concept).
    //!
    //! @param __sndr   The predecessor sender whose value-completion is to be
    //!                 adapted. Perfect-forwarded into the resulting sender, so an
    //!                 rvalue is moved and an lvalue copied as needed.
    //! @param __fun    The function (or callable) to invoke with each
    //!                 value-completion of @c __sndr. Stored by value (decayed) in
    //!                 the resulting sender.
    //!
    //! @returns A sender that, when connected to a receiver and started, drives
    //!          @c __sndr and routes each of its value-completions through
    //!          @c __fun. The error and stopped channels are forwarded unchanged.
    //!
    //! @pre @c __fun must be invocable with every value-completion argument pack of
    //!      @c __sndr (with appropriate value categories). Otherwise the program
    //!      is ill-formed at the point where the resulting sender is connected to
    //!      a receiver.
    template <sender _Sender, __movable_value _Fun>
    constexpr auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto
    {
      return __make_sexpr<then_t>(static_cast<_Fun&&>(__fun), static_cast<_Sender&&>(__sndr));
    }

    //! @brief Construct a sender-adaptor closure that, when applied to a sender,
    //!        produces <tt>then(sndr, __fun)</tt>.
    //!
    //! This overload enables the pipe syntax: <tt>sndr | then(__fun)</tt> is
    //! equivalent to <tt>then(sndr, __fun)</tt>.
    //!
    //! @tparam _Fun  A decayed, move-constructible callable type
    //!               (satisfying the internal <tt>__movable_value</tt> concept).
    //! @param __fun  The callable to invoke on the predecessor's value completions
    //!               when the closure is later applied to a sender.
    //!
    //! @returns A sender-adaptor closure object that captures @c __fun by value.
    //!          When piped against a sender @c sndr, it yields the sender
    //!          <tt>then(sndr, std::move(__fun))</tt>.
    template <__movable_value _Fun>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Fun __fun) const
    {
      return __closure(*this, static_cast<_Fun&&>(__fun));
    }
  };

  //! @brief The customization point object for the @c then sender adaptor.
  //!
  //! @c then is an instance of @ref then_t. See @ref then_t for the full
  //! description, the completion-signature transformation rules, exception and
  //! cancellation behavior, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr then_t then{};

  template <>
  struct __sexpr_impl<then_t> : __then::__then_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
