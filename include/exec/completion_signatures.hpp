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

#include "../stdexec/__detail/__completion_signatures.hpp"
#include "../stdexec/__detail/__meta.hpp"

namespace exec {
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // make_completion_signatures
  namespace detail {
    template <class Tag, class... As>
    auto normalize_impl(As&&...) -> Tag (*)(As...);

    template <class Tag, class... As>
    auto normalize(Tag (*)(As...))
      -> decltype(detail::normalize_impl<Tag>(stdexec::__declval<As>()...));

    template <class... Sigs>
    auto make_unique(Sigs*...)
      -> stdexec::__mapply_q<stdexec::completion_signatures, stdexec::__mmake_set<Sigs...>>;

    template <class... Sigs>
    using make_completion_signatures_t = decltype(detail::make_unique(
      detail::normalize(static_cast<Sigs*>(nullptr))...));
  } // namespace detail

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
        operator()(Sigs...) const noexcept -> stdexec::__concat_completion_signatures<Sigs...> {
        return {};
      }
    };
  } // namespace detail

  inline constexpr detail::concat_completion_signatures_t concat_completion_signatures{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// invalid_completion_signature
#if !STDEXEC_STD_NO_EXCEPTIONS()                                                                   \
  && __cpp_constexpr_exceptions >= 202411L // C++26, https://wg21.link/p3068

  template <class... What, class... Values>
  [[noreturn, nodiscard]]
  consteval auto invalid_completion_signature(Values... values) -> completion_signatures<> {
    if constexpr (sizeof...(Values) == 1) {
      throw sender_type_check_failure<Values..., What...>(values...);
    } else {
      throw sender_type_check_failure<::cuda::std::tuple<Values...>, What...>(
        ::cuda::std::tuple{values...});
    }
  }

#else // ^^^ constexpr exceptions ^^^ / vvv no constexpr exceptions vvv

  template <class... What, class... Values>
  [[nodiscard]]
  consteval auto invalid_completion_signature(Values...) {
    return stdexec::__mexception<What...>{};
  }

#endif // ^^^ no constexpr exceptions ^^^

  struct IN_TRANSFORM_COMPLETION_SIGNATURES;
  struct A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION;
  struct COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS;

  namespace detail {
    template <class Fn, class... As>
    using meta_call_result_t = decltype(stdexec::__declval<Fn>().template operator()<As...>());

    template <class Ay, class... As, class Fn>
    [[nodiscard]]
    consteval auto _transform_expr(const Fn& fn) -> meta_call_result_t<const Fn&, Ay, As...> {
      return fn.template operator()<Ay, As...>();
    }

    template <class Fn>
    [[nodiscard]]
    consteval auto _transform_expr(const Fn& fn) -> stdexec::__call_result_t<const Fn&> {
      return fn();
    }

    template <class Fn, class... As>
    using _transform_expr_t = decltype(detail::_transform_expr<As...>(
      stdexec::__declval<const Fn&>()));

    // transform_completion_signatures:
    template <class... As, class Fn>
    [[nodiscard]]
    consteval auto _apply_transform(const Fn& fn) {
      if constexpr (stdexec::__mvalid<_transform_expr_t, Fn, As...>) {
        using completions = _transform_expr_t<Fn, As...>;
        if constexpr (
          stdexec::__valid_completion_signatures<completions>
          || stdexec::__is_instance_of<completions, stdexec::_ERROR_>
          || std::is_base_of_v<stdexec::dependent_sender_error, completions>) {
          return detail::_transform_expr<As...>(fn);
        } else {
          (void) detail::_transform_expr<As...>(fn); // potentially throwing
          return invalid_completion_signature<
            IN_TRANSFORM_COMPLETION_SIGNATURES,
            A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
            stdexec::_WITH_FUNCTION_<Fn>,
            stdexec::_WITH_ARGUMENTS_<As...>
          >();
        }
      } else {
        return invalid_completion_signature<
          IN_TRANSFORM_COMPLETION_SIGNATURES,
          COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS,
          stdexec::_WITH_FUNCTION_<Fn>,
          stdexec::_WITH_ARGUMENTS_<As...>
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
        if constexpr (Tag{} == stdexec::set_value) {
          return detail::_apply_transform<Ts...>(value_fn);
        } else if constexpr (Tag{} == stdexec::set_error) {
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
    consteval auto operator()() const noexcept -> stdexec::completion_signatures<Tag(Ts...)> {
      return {};
    }
  };

  struct ignore_completion {
    template <class... Ts>
    consteval auto operator()() const noexcept -> stdexec::completion_signatures<> {
      return {};
    }
  };

  template <class Tag, class Fn>
  struct transform_arguments {
    template <class... Ts>
    consteval auto operator()() const noexcept
      -> stdexec::completion_signatures<Tag(stdexec::__minvoke<Fn, Ts>...)> {
      return {};
    }
  };

  template <class Tag>
  struct decay_arguments : transform_arguments<Tag, stdexec::__q1<stdexec::__decay_t>> { };

  template <
    class Completions,
    class ValueFn = keep_completion<stdexec::set_value_t>,
    class ErrorFn = keep_completion<stdexec::set_error_t>,
    class StoppedFn = keep_completion<stdexec::set_stopped_t>,
    class ExtraSigs = stdexec::completion_signatures<>
  >
  consteval auto transform_completion_signatures(
    Completions,
    ValueFn value_fn = {},
    ErrorFn error_fn = {},
    StoppedFn stopped_fn = {},
    ExtraSigs = {}) {
    STDEXEC_COMPLSIGS_LET(auto(completions) = Completions{}) {
      STDEXEC_COMPLSIGS_LET(auto(extra_sigs) = ExtraSigs{}) {
        detail::_transform_one<ValueFn, ErrorFn, StoppedFn> tfx1{value_fn, error_fn, stopped_fn};
        return concat_completion_signatures(
          completions.__apply(detail::_transform_all_fn{tfx1}), extra_sigs);
      }
    }
  }
} // namespace exec
