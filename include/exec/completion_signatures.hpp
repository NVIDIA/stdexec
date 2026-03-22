/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "../stdexec/__detail/__get_completion_signatures.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__sender_concepts.hpp"
#include "../stdexec/__detail/__transform_completion_signatures.hpp"

namespace experimental::execution
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // make_completion_signatures
  namespace detail
  {
    template <class Tag, class... As>
    constexpr auto normalize_impl(As&&...) -> Tag (*)(As...);

    template <class Tag, class... As>
    constexpr auto normalize(Tag (*)(As...))
      -> decltype(detail::normalize_impl<Tag>(STDEXEC::__declval<As>()...));

    template <class... Sigs>
    constexpr auto make_unique(Sigs*...)
      -> STDEXEC::__mcall<STDEXEC::__munique<STDEXEC::__qq<STDEXEC::completion_signatures>>,
                          Sigs...>;

    template <class... Sigs>
    using make_completion_signatures_t = decltype(detail::make_unique(
      detail::normalize(static_cast<Sigs*>(nullptr))...));
  }  // namespace detail

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // get_child_completion_signatures
  template <STDEXEC::sender _Parent, STDEXEC::sender _Child, class... _Env>
  [[nodiscard]]
  consteval auto get_child_completion_signatures()
  {
    return STDEXEC::get_completion_signatures<STDEXEC::__copy_cvref_t<_Parent, _Child>,
                                              STDEXEC::__fwd_env_t<_Env>...>();
  }

  //! Creates a compile-time completion signatures type from explicit and deduced signature types.
  //!
  //! This function is a compile-time helper that constructs a completion signatures type
  //! by combining explicitly provided signature types with those deduced from pointer
  //! arguments.
  //!
  //! \tparam ExplicitSigs Explicitly specified completion signature types. Must be a pack
  //!                      of function types, the returns types of which must be one of
  //!                      \c set_value_t, \c set_error_t, or \c set_stopped_t.
  //! \tparam DeducedSigs  Completion signature types to be deduced from the function
  //!                      arguments.
  //! \param unnamed       Pointer arguments (unused) for type deduction of
  //!                      \c DeducedSigs. Must be a pack of function pointer types, the
  //!                      returns types of which must be one of \c set_value_t,
  //!                      \c set_error_t, or \c set_stopped_t.
  //!
  //! \return An instance of \c STDEXEC::completion_signatures containing the combined
  //!         signatures.
  //!
  //! \note This is a \c consteval function, meaning it is only callable in constant
  //!       evaluation contexts (compile-time). It always returns a default-constructed
  //!       instance of the result type.
  //!
  //! \note The function uses pointer arguments for type deduction without requiring
  //!       actual object instances.
  template <class... ExplicitSigs, class... DeducedSigs>
  [[nodiscard]]
  consteval auto make_completion_signatures(DeducedSigs*...) noexcept
    -> detail::make_completion_signatures_t<ExplicitSigs..., DeducedSigs...>
  {
    return {};
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // concat_completion_signatures
  inline constexpr STDEXEC::__detail::__concat_completion_signatures_fn
    concat_completion_signatures{};

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // throw_compile_time_error
  template <class... What, class... Values>
  [[nodiscard]]
  consteval auto throw_compile_time_error(Values... vals)
  {
    return STDEXEC::__throw_compile_time_error<What...>(static_cast<Values&&>(vals)...);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // transform_completion_signatures

  template <class _SetTag>
  using keep_completion = STDEXEC::__keep_completion<_SetTag>;

  using ignore_completion = STDEXEC::__ignore_completion;

  template <class _SetTag, template <class> class _Fn, class... _AlgoTag>
  using transform_arguments =
    STDEXEC::__transform_arguments<_SetTag, STDEXEC::__q1<_Fn>, _AlgoTag...>;

  template <class _SetTag, class... _AlgoTag>
  using decay_arguments = STDEXEC::__decay_arguments<_SetTag, _AlgoTag...>;

  //! \brief Transforms completion signatures using provided transformation functions.
  //!
  //! This consteval function transforms a set of completion signatures by applying
  //! custom transformation functions to value, error, and stopped completion cases.
  //! The result can be augmented with additional extra signatures.
  //!
  //! \tparam Completions The input completion signatures to transform. Must be a
  //!                     specialization of \c STDEXEC::completion_signatures.
  //! \tparam ValueFn     Function object that transforms set_value_t completions.
  //!                     Defaults to keep_completion<set_value_t>.
  //! \tparam ErrorFn     Function object that transforms set_error_t completions.
  //!                     Defaults to keep_completion<set_error_t>.
  //! \tparam StoppedFn   Function object that transforms set_stopped_t completions.
  //!                     Defaults to keep_completion<set_stopped_t>.
  //! \tparam ExtraSigs   Additional completion signatures to append to the result.
  //!                     Must be a specialization of \c STDEXEC::completion_signatures.
  //!                     Defaults to \c STDEXEC::completion_signatures().
  //!
  //! \param completions  The input completion signatures object.
  //! \param value_fn     Value transformation function instance.
  //! \param error_fn     Error transformation function instance.
  //! \param stopped_fn   Stopped transformation function instance.
  //! \param extra_sigs   Extra signatures to append to the result.
  //!
  //! \return A transformed completion_signatures object combining the transformed
  //!         input signatures with the extra signatures.
  //!
  //! \par Example
  //!
  //! The following example demonstrates how to use \c transform_completion_signatures
  //! to compute the completion signatures of the \c then sender.
  //!
  //! \code{.cpp}
  //! namespace ex = STDEXEC;
  //!
  //! // A helper function to transform the value types of the child sender into the value
  //! // types of the then sender.
  //! template <class Fn, class... Args>
  //! consteval auto _transform_values()
  //! {
  //!   if constexpr (!std::invocable<Fn, Args...>)
  //!   {
  //!     // If Fn cannot be invoked with the given arguments, produce a compile-time
  //!     // error.
  //!     return exec::throw_compile_time_error<
  //!       WHAT(FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS),
  //!       WHERE(IN_ALGORITHM, then_t),
  //!       WITH_FUNCTION(Fn),
  //!       WITH_ARGUMENTS(Args...)>();
  //!   }
  //!   else
  //!   {
  //!     // transform the value types of the child sender into the value types of the
  //!     // then sender by applying Fn to them.
  //!     using result_t         = std::invoke_result_t<Fn, Args...>;
  //!     constexpr bool is_void = std::is_void_v<result_t>;
  //!     constexpr bool nothrow = std::is_nothrow_invocable_v<Fn, Args...>;
  //!     if constexpr (is_void && nothrow)
  //!     {
  //!       return ex::completion_signatures<set_value_t()>();
  //!     }
  //!     else if constexpr (is_void && !nothrow)
  //!     {
  //!       return ex::completion_signatures<set_value_t(),
  //!                                        set_error_t(std::exception_ptr)>();
  //!     }
  //!     else if constexpr (!is_void && nothrow)
  //!     {
  //!       return ex::completion_signatures<set_value_t(result_t)>();
  //!     }
  //!     else /* !is_void && !nothrow */
  //!     {
  //!       return ex::completion_signatures<set_value_t(result_t),
  //!                                        set_error_t(std::exception_ptr)>();
  //!     }
  //!   }
  //! }
  //!
  //! template <class Child, class Fn>
  //! struct then_sender
  //! {
  //!   using sender_concept = ex::sender_t;
  //!
  //!   template <class Self, class... Env>
  //!   static consteval auto get_completion_signatures()
  //!   {
  //!     // Compute the completion signatures of the child sender, and then transform
  //!     // them into the completion signatures of the `then` sender.
  //!     auto child_completions = exec::get_child_completion_signatures<Self, Child, Env...>();
  //!     auto value_fn = []<class... Args>() { return _transform_values<Fn, Args...>(); };
  //!
  //!     return exec::transform_completion_signatures(child_completions, value_fn);
  //!   }
  //!
  //!   // ...
  //! };
  //! \endcode
  //!
  //! \note This function is evaluated at compile-time (consteval).
  template <class Completions,
            class ValueFn   = keep_completion<STDEXEC::set_value_t>,
            class ErrorFn   = keep_completion<STDEXEC::set_error_t>,
            class StoppedFn = keep_completion<STDEXEC::set_stopped_t>,
            class ExtraSigs = STDEXEC::completion_signatures<>>
  consteval auto transform_completion_signatures(Completions,
                                                 ValueFn   value_fn   = {},
                                                 ErrorFn   error_fn   = {},
                                                 StoppedFn stopped_fn = {},
                                                 ExtraSigs            = {})
  {
    return STDEXEC::__transform_completion_signatures(Completions{},
                                                      value_fn,
                                                      error_fn,
                                                      stopped_fn,
                                                      ExtraSigs{});
  }
}  // namespace experimental::execution

namespace exec = experimental::execution;
