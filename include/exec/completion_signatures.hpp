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

namespace exec {
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // make_completion_signatures
  namespace detail {
    template <class Tag, class... As>
    constexpr auto normalize_impl(As&&...) -> Tag (*)(As...);

    template <class Tag, class... As>
    constexpr auto normalize(Tag (*)(As...))
      -> decltype(detail::normalize_impl<Tag>(STDEXEC::__declval<As>()...));

    template <class... Sigs>
    constexpr auto make_unique(Sigs*...)
      -> STDEXEC::__mapply_q<STDEXEC::completion_signatures, STDEXEC::__mmake_set<Sigs...>>;

    template <class... Sigs>
    using make_completion_signatures_t = decltype(detail::make_unique(
      detail::normalize(static_cast<Sigs*>(nullptr))...));
  } // namespace detail

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // get_child_completion_signatures
  template <STDEXEC::sender _Parent, STDEXEC::sender _Child, class... _Env>
  [[nodiscard]]
  consteval auto get_child_completion_signatures() {
    return STDEXEC::get_completion_signatures<
      STDEXEC::__copy_cvref_t<_Parent, _Child>,
      STDEXEC::__fwd_env_t<_Env>...
    >();
  }

  //! Creates a compile-time completion signatures type from explicit and deduced signature types.
  //!
  //! This function is a compile-time helper that constructs a completion signatures type
  //! by combining explicitly provided signature types with those deduced from pointer
  //! arguments.
  //!
  //! @tparam ExplicitSigs Explicitly specified completion signature types.
  //! @tparam DeducedSigs Completion signature types to be deduced from the function arguments.
  //! @param unnamed Pointer arguments (unused) used for type deduction of DeducedSigs.
  //!
  //! @return An instance of `STDEXEC::completion_signatures` containing the combined
  //!         signatures.
  //!
  //! @note This is a `consteval` function, meaning it is only callable in constant evaluation
  //!       contexts (compile-time). It always returns a default-constructed instance of the result
  //!       type.
  //!
  //! @note The function uses pointer arguments for type deduction without requiring actual object
  //!       instances.
  template <class... ExplicitSigs, class... DeducedSigs>
  [[nodiscard]]
  consteval auto make_completion_signatures(DeducedSigs*...) noexcept
    -> detail::make_completion_signatures_t<ExplicitSigs..., DeducedSigs...> {
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
  consteval auto throw_compile_time_error(Values... vals) {
    return STDEXEC::__throw_compile_time_error<What...>(static_cast<Values&&>(vals)...);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // transform_completion_signatures
  template <class _SetTag>
  using keep_completion = STDEXEC::__keep_completion<_SetTag>;

  using ignore_completion = STDEXEC::__ignore_completion;

  template <class _SetTag, class _Fn, class... _AlgoTag>
  using transform_arguments = STDEXEC::__transform_arguments<_SetTag, _Fn, _AlgoTag...>;

  template <class _SetTag, class... _AlgoTag>
  using decay_arguments = STDEXEC::__decay_arguments<_SetTag, _AlgoTag...>;

  template <
    class _Completions,
    class _ValueFn = keep_completion<STDEXEC::set_value_t>,
    class _ErrorFn = keep_completion<STDEXEC::set_error_t>,
    class _StoppedFn = keep_completion<STDEXEC::set_stopped_t>,
    class _ExtraSigs = STDEXEC::completion_signatures<>
  >
  consteval auto transform_completion_signatures(
    _Completions,
    _ValueFn __value_fn = {},
    _ErrorFn __error_fn = {},
    _StoppedFn __stopped_fn = {},
    _ExtraSigs = {}) {
    return STDEXEC::__transform_completion_signatures(
      _Completions{}, __value_fn, __error_fn, __stopped_fn, _ExtraSigs{});
  }
} // namespace exec
