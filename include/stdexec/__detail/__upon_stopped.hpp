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
#include "__meta.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"  // IWYU pragma: keep for __well_formed_sender
#include "__transform_completion_signatures.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  /////////////////////////////////////////////////////////////////////////////
  // [exec.then]
  namespace __upon_stopped
  {
    struct __upon_stopped_impl : __sexpr_defaults
    {
      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        static_assert(__sender_for<_Sender, upon_stopped_t>);
        using __fn_t = __decay_t<__data_of<_Sender>>;
        return STDEXEC::__transform_completion_signatures(
          STDEXEC::get_completion_signatures<__child_of<_Sender>, _Env...>(),
          {},
          {},
          __always{__set_value_from_t<__fn_t>()},
          __eptr_completion_unless_t<__mbool<__nothrow_callable<__fn_t>>>());
      };

      static constexpr auto __complete =
        []<class _Tag, class _State, class... _Args>(__ignore,
                                                     _State& __state,
                                                     _Tag,
                                                     _Args&&... __args) noexcept -> void
      {
        if constexpr (__same_as<_Tag, set_stopped_t>)
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
  }  // namespace __upon_stopped

  ////////////////////////////////////////////////////////////////////////////////////////////////
  //! @brief A pipeable sender adaptor that handles a predecessor sender's
  //!        stopped completion by invoking a nullary callable.
  //!
  //! @c upon_stopped maps the stopped channel of a sender into a value
  //! completion by invoking a callable with *no arguments* and forwarding its
  //! return value downstream. The value and error channels are forwarded
  //! unchanged. This is the canonical way to *recover from* cancellation:
  //! turn a stopped completion into a substitute value and continue.
  //!
  //! Both call syntaxes are supported (the second is the *pipeable* form):
  //!
  //! @code{.cpp}
  //! auto s1 = stdexec::upon_stopped(sndr, f);   // direct invocation
  //! auto s2 = sndr | stdexec::upon_stopped(f);  // pipe syntax
  //! @endcode
  //!
  //! The two forms are expression-equivalent. See [exec.then] in the
  //! C++26 working draft for the normative specification (@c upon_stopped is
  //! specified alongside @c then and @c upon_error).
  //!
  //! **Completion signatures.**
  //!
  //! Given a predecessor sender @c sndr with completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged
  //! set_error_t(Es)...               // forwarded unchanged
  //! set_stopped_t()                  // (must be present)
  //! @endcode
  //!
  //! the sender produced by <tt>upon_stopped(sndr, f)</tt> has completion signatures
  //!
  //! @code{.cpp}
  //! set_value_t(Vs...)               // forwarded unchanged from sndr
  //! set_value_t(R)                   // R = decltype(std::invoke(f))
  //!                                  // (or set_value_t() when R is void)
  //! set_error_t(Es)...               // forwarded unchanged from sndr
  //! set_error_t(std::exception_ptr)  // added when invoking f may throw
  //!                                  // (no set_stopped_t in the output)
  //! @endcode
  //!
  //! @c f must be invocable with *no* arguments — this requirement is enforced
  //! by the @c requires clause on the operator overloads. The original
  //! @c set_stopped_t completion is *consumed*: the resulting sender will
  //! never complete via @c set_stopped (unless @c f itself returns a sender
  //! that does, which @c upon_stopped does not — see @c let_stopped for that).
  //!
  //! **Exception behavior.**
  //!
  //! If invoking @c f throws, the exception is delivered through
  //! @c set_error_t(std::exception_ptr) on the resulting sender. When @c f is
  //! @c noexcept, no @c std::exception_ptr error completion is added.
  //!
  //! **Cancellation.**
  //!
  //! @c upon_stopped does not interact with the receiver's stop token. It
  //! reacts to the predecessor's @c set_stopped by invoking @c f; it does
  //! not itself initiate cancellation.
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
  //!   auto sndr = just_stopped()
  //!             | upon_stopped([] { return 42; });
  //!
  //!   auto [v] = sync_wait(std::move(sndr)).value();
  //!   assert(v == 42);
  //! }
  //! @endcode
  //!
  //! @see stdexec::then         — adapt the value channel
  //! @see stdexec::upon_error   — adapt the error channel
  //! @see stdexec::let_stopped  — adapt the stopped channel with a sender-returning function
  struct upon_stopped_t
  {
    //! @brief Construct a sender that handles a stopped completion of @c __sndr
    //!        by invoking @c __fun and delivering its return value.
    //!
    //! @tparam _Sender A type satisfying the @c stdexec::sender concept.
    //! @tparam _Fun    A decayed, move-constructible, *nullary* callable type.
    //!
    //! @param __sndr   The predecessor sender whose stopped completion is to be
    //!                 adapted. Perfect-forwarded into the resulting sender.
    //! @param __fun    The function (or callable) to invoke when @c __sndr
    //!                 completes via @c set_stopped. Stored by value
    //!                 (decayed) in the resulting sender.
    //!
    //! @returns A sender that, when connected to a receiver and started, drives
    //!          @c __sndr and reacts to its @c set_stopped completion by
    //!          invoking @c __fun and forwarding the result via @c set_value.
    //!          The value and error channels of @c __sndr are forwarded unchanged.
    //!
    //! @pre @c __fun must be invocable with no arguments (the @c requires
    //!      clause enforces this). Otherwise the call is not viable.
    template <sender _Sender, __movable_value _Fun>
      requires __callable<_Fun>
    auto operator()(_Sender&& __sndr, _Fun __fun) const -> __well_formed_sender auto
    {
      return __make_sexpr<upon_stopped_t>(static_cast<_Fun&&>(__fun),
                                          static_cast<_Sender&&>(__sndr));
    }

    //! @brief Construct a sender-adaptor closure that, when applied to a sender,
    //!        produces <tt>upon_stopped(sndr, __fun)</tt>.
    //!
    //! This overload enables the pipe syntax: <tt>sndr | upon_stopped(__fun)</tt>
    //! is equivalent to <tt>upon_stopped(sndr, __fun)</tt>.
    //!
    //! @tparam _Fun  A decayed, move-constructible, *nullary* callable type.
    //! @param __fun  The callable to invoke on the predecessor's stopped
    //!               completion when the closure is later applied to a sender.
    //!
    //! @returns A sender-adaptor closure object that captures @c __fun by value.
    //!          When piped against a sender @c sndr, it yields the sender
    //!          <tt>upon_stopped(sndr, std::move(__fun))</tt>.
    template <__movable_value _Fun>
      requires __callable<_Fun>
    STDEXEC_ATTRIBUTE(always_inline)
    auto operator()(_Fun __fun) const noexcept(__nothrow_move_constructible<_Fun>)
    {
      return __closure(*this, static_cast<_Fun&&>(__fun));
    }
  };

  //! @brief The customization point object for the @c upon_stopped sender adaptor.
  //!
  //! @c upon_stopped is an instance of @ref upon_stopped_t. See
  //! @ref upon_stopped_t for the full description, completion-signature
  //! transformation rules, exception and cancellation behavior, and a usage
  //! example.
  //!
  //! @hideinitializer
  inline constexpr upon_stopped_t upon_stopped{};

  template <>
  struct __sexpr_impl<upon_stopped_t> : __upon_stopped::__upon_stopped_impl
  {};

}  // namespace STDEXEC

#include "__epilogue.hpp"
