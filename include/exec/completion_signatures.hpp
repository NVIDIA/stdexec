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
#include "../stdexec/__detail/__tuple.hpp" // IWYU pragma: keep for STDEXEC::__tuple

namespace exec {
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // make_completion_signatures
  namespace detail {
    template <class Tag, class... As>
    auto normalize_impl(As&&...) -> Tag (*)(As...);

    template <class Tag, class... As>
    auto normalize(Tag (*)(As...))
      -> decltype(detail::normalize_impl<Tag>(STDEXEC::__declval<As>()...));

    template <class... Sigs>
    auto make_unique(Sigs*...)
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
  //! @return An instance of `detail::make_completion_signatures_t` containing the combined
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
  namespace detail {
    struct concat_completion_signatures_t {
      template <class... Sigs>
      [[nodiscard]]
      consteval auto
        operator()(Sigs...) const noexcept -> STDEXEC::__concat_completion_signatures<Sigs...> {
        return {};
      }
    };
  } // namespace detail

  inline constexpr detail::concat_completion_signatures_t concat_completion_signatures{};

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // invalid_completion_signature
  template <class... What, class... Values>
  [[nodiscard]]
  consteval auto invalid_completion_signature(Values... vals) {
    return STDEXEC::__invalid_completion_signature<What...>(static_cast<Values&&>(vals)...);
  }

  struct IN_TRANSFORM_COMPLETION_SIGNATURES;
  struct A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION;
  struct COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS;

  namespace detail {
    template <class Fn, class... As>
    using meta_call_result_t = decltype(STDEXEC::__declval<Fn>().template operator()<As...>());

    template <class Ay, class... As, class Fn>
    [[nodiscard]]
    consteval auto _transform_expr(const Fn& fn) -> meta_call_result_t<const Fn&, Ay, As...> {
      return fn.template operator()<Ay, As...>();
    }

    template <class Fn>
    [[nodiscard]]
    consteval auto _transform_expr(const Fn& fn) -> STDEXEC::__call_result_t<const Fn&> {
      return fn();
    }

    template <class Fn, class... As>
    using _transform_expr_t = decltype(detail::_transform_expr<As...>(
      STDEXEC::__declval<const Fn&>()));

    // transform_completion_signatures:
    template <class... As, class Fn>
    [[nodiscard]]
    consteval auto _apply_transform(const Fn& fn) {
      if constexpr (STDEXEC::__mvalid<_transform_expr_t, Fn, As...>) {
        using _completions_t = _transform_expr_t<Fn, As...>;
        if constexpr (STDEXEC::__well_formed_completions<_completions_t>) {
          return detail::_transform_expr<As...>(fn);
        } else {
          (void) detail::_transform_expr<As...>(fn); // potentially throwing
          return invalid_completion_signature<
            IN_TRANSFORM_COMPLETION_SIGNATURES,
            A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
            STDEXEC::_WITH_FUNCTION_<Fn>,
            STDEXEC::_WITH_ARGUMENTS_<As...>
          >();
        }
      } else {
        return invalid_completion_signature<
          IN_TRANSFORM_COMPLETION_SIGNATURES,
          COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS,
          STDEXEC::_WITH_FUNCTION_<Fn>,
          STDEXEC::_WITH_ARGUMENTS_<As...>
        >();
      }
    }

    template <class ValueFn, class ErrorFn, class StoppedFn>
    struct _transform_one {
      ValueFn value_fn;
      ErrorFn error_fn;
      StoppedFn stopped_fn;

      template <class Tag, class... Ts>
      [[nodiscard]]
      consteval auto operator()(Tag (*)(Ts...)) const {
        if constexpr (Tag{} == STDEXEC::set_value) {
          return detail::_apply_transform<Ts...>(value_fn);
        } else if constexpr (Tag{} == STDEXEC::set_error) {
          return detail::_apply_transform<Ts...>(error_fn);
        } else {
          return detail::_apply_transform<Ts...>(stopped_fn);
        }
      }
    };

    template <class TransformOne>
    struct _transform_all_fn {
      TransformOne tfx1;

      template <class... Sigs>
      [[nodiscard]]
      consteval auto operator()(Sigs*... sigs) const {
        return concat_completion_signatures(tfx1(sigs)...);
      }
    };

    template <class TransformOne>
    _transform_all_fn(TransformOne) -> _transform_all_fn<TransformOne>;
  } // namespace detail

  template <class Tag>
  struct keep_completion {
    template <class... Ts>
    consteval auto operator()() const noexcept -> STDEXEC::completion_signatures<Tag(Ts...)> {
      return {};
    }
  };

  struct ignore_completion {
    template <class... Ts>
    consteval auto operator()() const noexcept -> STDEXEC::completion_signatures<> {
      return {};
    }
  };

  template <class Tag, class Fn>
  struct transform_arguments {
    template <class... Ts>
    consteval auto operator()() const noexcept
      -> STDEXEC::completion_signatures<Tag(STDEXEC::__minvoke<Fn, Ts>...)> {
      return {};
    }
  };

  template <class Tag>
  struct decay_arguments : transform_arguments<Tag, STDEXEC::__q1<STDEXEC::__decay_t>> { };

  template <
    class Completions,
    class ValueFn = keep_completion<STDEXEC::set_value_t>,
    class ErrorFn = keep_completion<STDEXEC::set_error_t>,
    class StoppedFn = keep_completion<STDEXEC::set_stopped_t>,
    class ExtraSigs = STDEXEC::completion_signatures<>
  >
  consteval auto transform_completion_signatures(
    Completions,
    ValueFn value_fn = {},
    ErrorFn error_fn = {},
    StoppedFn stopped_fn = {},
    ExtraSigs = {}) {
    STDEXEC_COMPLSIGS_LET(completions, Completions{}) {
      STDEXEC_COMPLSIGS_LET(extra_sigs, ExtraSigs{}) {
        detail::_transform_one<ValueFn, ErrorFn, StoppedFn> tfx1{value_fn, error_fn, stopped_fn};
        return concat_completion_signatures(
          completions.__apply(detail::_transform_all_fn{tfx1}), extra_sigs);
      }
    }
  }
} // namespace exec
