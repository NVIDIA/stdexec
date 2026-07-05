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
#include "__senders.hpp"  // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.then]
  namespace __upon_error
  {
    struct __upon_error_impl : __sexpr_defaults
    {
      static constexpr auto __get_attrs =
        []<class _Child>(__ignore, __ignore, _Child const & __child) noexcept
      {
        return __sync_attrs{__child};
      };

      template <class _Fun>
      static consteval auto __transform_error_completion() noexcept
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
              __callable_error_t<upon_error_t, _Fun, _Args...>());
          }
        };
      }

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, upon_error_t>);
        using __fn_t = __decay_t<__data_of<_Sender>>;
        return STDEXEC::__transform_completion_signatures(
          STDEXEC::get_completion_signatures<__child_of<_Sender>, _Env...>(),
          {},
          __transform_error_completion<__fn_t>());
      };

      static constexpr auto __complete =
        []<class _Tag, class _State, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag,
                                                     _Args&&... __args) noexcept -> void
      {
        if constexpr (__same_as<_Tag, set_error_t>)
        {
          STDEXEC::__set_value_from(static_cast<_State&&>(__state).__rcvr_,
                                    static_cast<_State&&>(__state).__data_,
                                    static_cast<_Args&&>(__args)...);
        }
        else
        {
          _Tag()(static_cast<_State&&>(__state).__rcvr_, static_cast<_Args&&>(__args)...);
        }
      };
    };
  }  // namespace __upon_error

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //! @brief A pipeable sender adaptor that handles a predecessor sender's
  //!        error completion by invoking a callable on the error datum.
  //!
  //! @c upon_error maps the error channel of a sender through a function while
  //! forwarding the value and stopped channels unchanged. The function's return
  //! value becomes a *value* completion on the resulting sender — so
  //! @c upon_error is the canonical way to *recover from* an error: turn it
  //! into a substitute value and continue the pipeline as if nothing had gone
  //! wrong.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::upon_error(sndr, f);   // direct invocation
  //! auto s2 = sndr | stdexec::upon_error(f);  // pipe syntax
  //! @endcode
  //!
  //! The two forms are expression-equivalent. See [exec.then] in the
  //! C++26 working draft for the normative specification (@c upon_error is
  //! specified alongside @c then and @c upon_stopped).
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged
  //! set_error_t(Es)...               // one or more error completions
  //! set_stopped_t()                  // forwarded unchanged (if present)
  //! @endcode
  //!
  //! the sender produced by <tt>upon_error(sndr, f)</tt> has completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged from sndr
  //! set_value_t(R)                   // R = decltype(std::invoke(f, E))  for each E
  //!                                  // (or set_value_t() when R is void)
  //! set_error_t(std::exception_ptr)  // added when invoking f may throw
  //! set_stopped_t()                  // forwarded unchanged from sndr
  //! @endcode
  //!
  //! For each distinct error type @c E that @c sndr may complete with, @c f
  //! must be invocable with @c E (with appropriate value category). Otherwise
  //! the program is ill-formed at the point where the resulting sender is
  //! connected to a receiver. All original @c set_error_t completions are
  //! *replaced* by the union of value completions produced by @c f — only
  //! errors thrown by @c f itself remain on the error channel.
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c f throws, the exception is delivered through
  //! @c set_error_t(std::exception_ptr) on the resulting sender. When @c f is
  //! @c noexcept for every error type of @c sndr, no @c std::exception_ptr
  //! error completion is added.
  //!
  //! **Cancellation.**
  //!
  //! @c upon_error does not interact with the receiver's stop token. When
  //! @c sndr completes with @c set_stopped, @c f is not invoked and the
  //! stopped completion is forwarded to the downstream receiver.
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
  //!   auto sndr = just_error(std::error_code{ENOENT, std::system_category()})
  //!             | upon_error([](std::error_code) { return -1; });
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == -1);
  //! }
  //! @endcode
  //!
  //! @see stdexec::then          — adapt the value channel
  //! @see stdexec::upon_stopped  — adapt the stopped channel
  //! @see stdexec::let_error     — adapt the error channel with a sender-returning function
  struct upon_error_t
  {
    //! @brief Construct a sender that handles each error completion of @c __sndr
    //!        by invoking @c __fun on the error datum.
    //!
    //! @tparam _Sender A type satisfying the @c stdexec::sender concept.
    //! @tparam _Fun    A decayed, move-constructible callable type
    //!                 (satisfying the internal <tt>__movable_value</tt> concept).
    //!
    //! @param __sndr   The predecessor sender whose error-completions are to be
    //!                 adapted. Perfect-forwarded into the resulting sender.
    //! @param __fun    The function (or callable) to invoke with each
    //!                 error-completion datum of @c __sndr. Stored by value
    //!                 (decayed) in the resulting sender.
    //!
    //! @returns A sender that, when connected to a receiver and started, drives
    //!          @c __sndr and routes each of its error-completions through
    //!          @c __fun, delivering the result on the value channel. The
    //!          value and stopped channels of @c __sndr are forwarded unchanged.
    //!
    //! @pre @c __fun must be invocable with every error type of @c __sndr
    //!      (with appropriate value categories). Otherwise the program is
    //!      ill-formed at the point where the resulting sender is connected
    //!      to a receiver.
    template <sender _Sender, __movable_value _Fun>
    constexpr auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto
    {
      return __make_sexpr<upon_error_t>(static_cast<_Fun&&>(__fun), static_cast<_Sender&&>(__sndr));
    }

    //! @brief Construct a sender-adaptor closure that, when applied to a sender,
    //!        produces <tt>upon_error(sndr, __fun)</tt>.
    //!
    //! This overload enables the pipe syntax: <tt>sndr | upon_error(__fun)</tt>
    //! is equivalent to <tt>upon_error(sndr, __fun)</tt>.
    //!
    //! @tparam _Fun  A decayed, move-constructible callable type.
    //! @param __fun  The callable to invoke on the predecessor's error
    //!               completions when the closure is later applied to a sender.
    //!
    //! @returns A sender-adaptor closure object that captures @c __fun by value.
    //!          When piped against a sender @c sndr, it yields the sender
    //!          <tt>upon_error(sndr, std::move(__fun))</tt>.
    template <__movable_value _Fun>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto operator()(_Fun __fun) const noexcept(__nothrow_move_constructible<_Fun>)
    {
      return __closure(*this, static_cast<_Fun&&>(__fun));
    }
  };

  //! @brief The customization point object for the @c upon_error sender adaptor.
  //!
  //! @c upon_error is an instance of @ref upon_error_t. See @ref upon_error_t
  //! for the full description, completion-signature transformation rules,
  //! exception and cancellation behavior, and a usage example.
  //!
  //! @hideinitializer
  inline constexpr upon_error_t upon_error{};

  template <>
  struct __sexpr_impl<upon_error_t> : __upon_error::__upon_error_impl
  {};
}  // namespace STDEXEC

#include "__epilogue.hpp"
