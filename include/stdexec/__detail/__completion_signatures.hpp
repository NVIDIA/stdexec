/*
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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#  include "__concat_completion_signatures.hpp"
#  include "__concepts.hpp"
#  include "__meta.hpp"
#  include "__partition_completion_signatures.hpp"
#  include "__utility.hpp"

#  if !STDEXEC_USE_MODULES()
#    include <exception>
#    include <type_traits>
#  endif

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  //////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __cmplsigs
  {
    STDEXEC_MODULE_EXPORT_AUTHORING
    template <class _Sig>
    inline constexpr bool __is_valid_cmplsig = false;
    template <class... _Args>
    inline constexpr bool __is_valid_cmplsig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_valid_cmplsig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_valid_cmplsig<set_stopped_t()> = true;
  }  // namespace __cmplsigs

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sig>
  concept __completion_signature = __cmplsigs::__is_valid_cmplsig<_Sig>;

  namespace __detail
  {
    template <class _Fn, class _Sig>
    concept __filter_pass = requires { requires _Fn()(__midentity<_Sig*>()); };

    template <class _Fn, class _Sig>
    using __filer_one_t =
      __if_c<__filter_pass<_Fn, _Sig>, completion_signatures<_Sig>, completion_signatures<>>;
  }  // namespace __detail

  struct _IN_COMPLETION_SIGNATURES_APPLY_;

  //! @brief Represents a set of completion signatures for senders in the CUDA
  //! C++ execution model.
  //!
  //! The `completion_signatures` class template is used to describe the
  //! possible ways a sender may complete. Each signature is a function type of
  //! the form `set_value_t(Ts...)`, `set_error_t(E)`, or `set_stopped_t()`.
  //! This type provides compile-time utilities for querying, combining, and
  //! transforming sets of completion signatures.
  //!
  //! @tparam _Sigs... The completion signature types to include in this set.
  //!
  //! @headerfile <stdexec/execution.hpp>
  //!
  //! Example usage:
  //! @code
  //! constexpr auto sigs = completion_signatures<set_value_t(int),
  //!                                             set_error_t(float),
  //!                                             set_stopped_t()>{};
  //! static_assert(sigs.size() == 3);
  //! static_assert(sigs.contains<set_value_t(int)>());
  //! @endcode
  template <class... _Sigs>
  struct completion_signatures
  {
    static_assert((__completion_signature<_Sigs> && ...),
                  "All types in completion_signatures must be valid completion signatures.");

    //! @brief Partitioned view of the completion signatures for efficient
    //! querying.
    struct __partitioned
    {
      // This is defined in a nested struct to avoid computing these types if
      // they are not needed.
      using __t = __cmplsigs::__partition_completion_signatures_t<_Sigs...>;
    };

    //! @brief Type set view of the completion signatures for set operations.
    struct __type_set
    {
      // This is defined in a nested struct to avoid computing this type if it
      // is not needed.
      using __t = __mmake_set<_Sigs...>;
    };

    //! @brief Returns the number of completion signatures in the set.
    //! @return The number of signatures.
    static constexpr auto __size() noexcept -> std::size_t
    {
      return sizeof...(_Sigs);
    }

    //! @brief Counts the number of signatures with the given tag.
    //! @tparam _Tag The tag to count (e.g., set_value, set_error, set_stopped).
    //! @return The number of signatures with the given tag.
    template <class _Tag>
    [[nodiscard]]
    static consteval auto __count(_Tag) noexcept -> std::size_t
    {
      if constexpr (_Tag() == set_value)
      {
        return __partitioned::__t::__count_values::value;
      }
      else if constexpr (_Tag() == set_error)
      {
        return __partitioned::__t::__count_errors::value;
      }
      else
      {
        return __partitioned::__t::__count_stopped::value;
      }
    }

    //! @brief Checks if the set contains the given signature.
    //! @tparam _Sig The signature type to check.
    //! @return true if the signature is present, false otherwise.
    template <class _Sig>
    [[nodiscard]]
    static consteval auto __contains(_Sig* = nullptr) noexcept -> bool
    {
      return __mset_contains<__t<__type_set>, _Sig>;
    }

    //! @brief Applies a callable to all signatures in the set.
    //! @tparam _Fn The callable to apply.
    //! @param __fn The callable instance.
    //! @return The result of calling __fn with all signatures as arguments.
    template <class _Fn>
    static consteval auto __apply(_Fn __fn)
    {
      if constexpr (__callable<_Fn, _Sigs*...>)
      {
        return __fn(static_cast<_Sigs*>(nullptr)...);
      }
      else
      {
        return STDEXEC::__throw_compile_time_error<
          _WHERE_(_IN_COMPLETION_SIGNATURES_APPLY_),
          _FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_,
          _WITH_FUNCTION_(_Fn),
          _WITH_ARGUMENTS_(_Sigs * ...)>();
      }
    }

    //! @brief Filters the set using a predicate, returning a new set with only
    //! matching signatures.
    //! @tparam _Fn The predicate type. Must be empty and trivially
    //! constructible. Must satisfy `(std::predicate<_Fn, _Sigs*> &&...)`.
    //! @return A new completion_signatures set with only the signatures for
    //! which the predicate returns true.
    template <class _Fn>
    [[nodiscard]]
    static consteval auto __filter(_Fn)
    {
      static_assert(std::is_empty_v<_Fn> && std::is_trivially_constructible_v<_Fn>,
                    "The filter function must be empty and trivially constructible.");
      return __concat_completion_signatures_t<__detail::__filer_one_t<_Fn, _Sigs>...>{};
    }

    //! @brief Selects all signatures with the given tag.
    //! @tparam _Tag The tag to select (e.g., set_value, set_error,
    //! set_stopped).
    //! @return A new completion_signatures set containing only signatures with
    //! the given tag.
    template <__completion_tag _Tag>
    [[nodiscard]]
    static consteval auto __select(_Tag) noexcept
    {
      if constexpr (_Tag() == set_value)
      {
        return __value_types_t<completion_signatures,
                               __qf<set_value_t>,
                               __qq<STDEXEC::completion_signatures>>();
      }
      else if constexpr (_Tag() == set_error)
      {
        return __error_types_t<completion_signatures,
                               __qq<STDEXEC::completion_signatures>,
                               __qf<set_error_t>>();
      }
      else
      {
        return __stopped_types_t<completion_signatures,  //
                                 __qq<STDEXEC::completion_signatures>>();
      }
    }

    //! @brief Applies a transform and then reduces the results.
    //! @tparam _Transform The transform callable.
    //! @tparam _Reduce The reduce callable.
    //! @param __transform The transform instance.
    //! @param __reduce The reduce instance.
    //! @return The result of reducing the transformed signatures.
    template <class _Transform, class _Reduce>
    [[nodiscard]]
    static consteval auto __transform_reduce(_Transform __transform, _Reduce __reduce)
      -> __call_result_t<_Reduce, __call_result_t<_Transform, _Sigs*>...>
    {
      return __reduce(__transform(static_cast<_Sigs*>(nullptr))...);
    }

    //! NOT TO SPEC
    template <class... _OtherSigs>
    [[nodiscard]]
    consteval bool operator==(completion_signatures<_OtherSigs...> const &) const noexcept
    {
      using __other_set = completion_signatures<_OtherSigs...>::__type_set::__t;
      return __mset_eq<__t<__type_set>, __other_set>;
    }
  };

  STDEXEC_MODULE_EXPORT_AUTHORING
  using __eptr_completion_t = completion_signatures<set_error_t(std::exception_ptr)>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _NoExcept>
  using __eptr_completion_unless_t = __if<_NoExcept, completion_signatures<>, __eptr_completion_t>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
