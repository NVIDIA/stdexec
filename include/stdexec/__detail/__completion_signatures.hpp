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

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__meta.hpp"

#include <type_traits>

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __sigs {
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;

    template <class>
    inline constexpr bool __is_completion_signatures = false;
    template <class... _Sigs>
    inline constexpr bool __is_completion_signatures<completion_signatures<_Sigs...>> = true;

    // __partitions is a cache of completion signatures for fast
    // access. The completion_signatures<Sigs...>::__partitioned nested struct
    // inherits from __partitions. If the cache is never accessed,
    // it is never instantiated.
    template <
      class _ValueTuplesList = __types<>,
      class _ErrorsList = __types<>,
      class _StoppedList = __types<>
    >
    struct __partitions;

    template <class... _ValueTuples, class... _Errors, class... _Stopped>
    struct __partitions<__types<_ValueTuples...>, __types<_Errors...>, __types<_Stopped...>> {
      template <class _Tuple, class _Variant>
      using __value_types = _Variant::template __f<__mapply<_Tuple, _ValueTuples>...>;

      template <class _Variant, class _Transform = __q1<__midentity>>
      using __error_types = _Variant::template __f<typename _Transform::template __f<_Errors>...>;

      template <class _Variant, class _Type = set_stopped_t()>
      using __stopped_types = _Variant::template __f<__msecond<_Stopped, _Type>...>;

      using __count_values = __msize_t<sizeof...(_ValueTuples)>;
      using __count_errors = __msize_t<sizeof...(_Errors)>;
      using __count_stopped = __msize_t<sizeof...(_Stopped)>;

      struct __nothrow_decay_copyable {
        // These aliases are placed in a separate struct to avoid computing them
        // if they are not needed.
        using __values = __mand_t<__mapply_q<__nothrow_decay_copyable_t, _ValueTuples>...>;
        using __errors = __nothrow_decay_copyable_t<_Errors...>;
        using __all = __mand_t<__values, __errors>;
      };
    };

    template <class _Tag>
    struct __partitioned_fold_fn;

    template <>
    struct __partitioned_fold_fn<set_value_t> {
      template <class... _ValueTuples, class _Errors, class _Stopped, class _Values>
      auto operator()(
        __partitions<__types<_ValueTuples...>, _Errors, _Stopped>&,
        __undefined<_Values>&) const
        -> __undefined<__partitions<__types<_ValueTuples..., _Values>, _Errors, _Stopped>>&;
    };

    template <>
    struct __partitioned_fold_fn<set_error_t> {
      template <class _Values, class... _Errors, class _Stopped, class _Error>
      auto operator()(
        __partitions<_Values, __types<_Errors...>, _Stopped>&,
        __undefined<__types<_Error>>&) const
        -> __undefined<__partitions<_Values, __types<_Errors..., _Error>, _Stopped>>&;
    };

    template <>
    struct __partitioned_fold_fn<set_stopped_t> {
      template <class _Values, class _Errors, class _Stopped>
      auto operator()(__partitions<_Values, _Errors, _Stopped>&, __ignore) const
        -> __undefined<__partitions<_Values, _Errors, __types<set_stopped_t()>>>&;
    };

    // The following overload of binary operator* is used to build up the cache of completion
    // signatures. We fold over operator*, accumulating the completion signatures in the
    // cache. `__undefined` is used here to prevent the instantiation of the intermediate
    // types.
    template <class _Partitioned, class _Tag, class... _Args>
    auto operator*(__undefined<_Partitioned>&, _Tag (*)(_Args...)) -> __call_result_t<
      __partitioned_fold_fn<_Tag>,
      _Partitioned&,
      __undefined<__types<_Args...>>&
    >;

    // This function declaration is used to extract the cache from the `__undefined` type.
    template <class _Partitioned>
    auto __unpack_partitioned_completions(__undefined<_Partitioned>&) -> _Partitioned;

    template <class... _Sigs>
    using __partition_completion_signatures_t = //
      decltype(__sigs::__unpack_partitioned_completions(
        (__declval<__undefined<__partitions<>>&>() * ... * static_cast<_Sigs*>(nullptr))));

    template <class _Completions>
    using __partitions_of_t = _Completions::__partitioned::__t;
  } // namespace __sigs

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // completion signatures type traits
  template <class _Completions>
  concept __valid_completion_signatures = __ok<_Completions>
                                       && __sigs::__is_completion_signatures<_Completions>;

  template <
    class _Sigs,
    class _Tuple = __qq<__decayed_std_tuple>,
    class _Variant = __qq<__std_variant>
  >
  using __value_types_t =
    __sigs::__partitions_of_t<_Sigs>::template __value_types<_Tuple, _Variant>;

  template <
    class _Sndr,
    class _Env = env<>,
    class _Tuple = __qq<__decayed_std_tuple>,
    class _Variant = __qq<__std_variant>
  >
  using __value_types_of_t =
    __value_types_t<__completion_signatures_of_t<_Sndr, _Env>, _Tuple, _Variant>;

  template <
    class _Sndr,
    class _Env = env<>,
    template <class...> class _Tuple = __decayed_std_tuple,
    template <class...> class _Variant = __std_variant
  >
  using value_types_of_t =
    __value_types_t<__completion_signatures_of_t<_Sndr, _Env>, __q<_Tuple>, __q<_Variant>>;

  template <class _Sigs, class _Variant = __qq<__std_variant>, class _Transform = __q1<__midentity>>
  using __error_types_t =
    __sigs::__partitions_of_t<_Sigs>::template __error_types<_Variant, _Transform>;

  template <
    class _Sender,
    class _Env = env<>,
    class _Variant = __qq<__std_variant>,
    class _Transform = __q1<__midentity>
  >
  using __error_types_of_t =
    __error_types_t<__completion_signatures_of_t<_Sender, _Env>, _Variant, _Transform>;

  template <class _Sndr, class _Env = env<>, template <class...> class _Variant = __std_variant>
  using error_types_of_t =
    __error_types_t<__completion_signatures_of_t<_Sndr, _Env>, __q<_Variant>>;

  template <class _Sigs, class _Variant, class _Type = set_stopped_t()>
  using __stopped_types_t =
    __sigs::__partitions_of_t<_Sigs>::template __stopped_types<_Variant, _Type>;

  template <__valid_completion_signatures _Sigs>
  inline constexpr bool __sends_stopped =
    __v<typename __sigs::__partitions_of_t<_Sigs>::__count_stopped> != 0;

  template <class _Sndr, class... _Env>
    requires __valid_completion_signatures<__completion_signatures_of_t<_Sndr, _Env...>>
  inline constexpr bool sends_stopped =
    __sends_stopped<__completion_signatures_of_t<_Sndr, _Env...>>;

  template <class... _Sigs>
  using __concat_completion_signatures =
    __mconcat<__qq<completion_signatures>>::__f<__mconcat<__qq<__mmake_set>>::__f<_Sigs...>>;

  namespace __detail {
    template <class _Fn, class _Sig>
    concept __filter_pass = requires { requires _Fn()(__midentity<_Sig*>()); };

    template <class _Fn, class _Sig>
    using __filer_one_t =
      __if_c<__filter_pass<_Fn, _Sig>, completion_signatures<_Sig>, completion_signatures<>>;
  } // namespace __detail

  //! @brief Represents a set of completion signatures for senders in the CUDA C++ execution
  //! model.
  //!
  //! The `completion_signatures` class template is used to describe the possible ways a
  //! sender may complete. Each signature is a function type of the form
  //! `set_value_t(Ts...)`, `set_error_t(E)`, or `set_stopped_t()`. This type provides
  //! compile-time utilities for querying, combining, and transforming sets of completion
  //! signatures.
  //!
  //! @tparam _Sigs... The completion signature types to include in this set.
  //!
  //! @headerfile <stdexec/execution.hpp>
  //!
  //! Example usage:
  //! @code
  //! constexpr auto sigs = completion_signatures<set_value_t(int), set_error_t(float), set_stopped_t()>{};
  //! static_assert(sigs.size() == 3);
  //! static_assert(sigs.contains<set_value_t(int)>());
  //! @endcode
  template <class... _Sigs>
  struct completion_signatures {
    //! @brief Partitioned view of the completion signatures for efficient querying.
    struct __partitioned {
      // This is defined in a nested struct to avoid computing these types if they are not
      // needed.
      using __t = __sigs::__partition_completion_signatures_t<_Sigs...>;
    };

    //! @brief Type set view of the completion signatures for set operations.
    struct __type_set {
      // This is defined in a nested struct to avoid computing this type if it is not
      // needed.
      using __t = __mmake_set<_Sigs...>;
    };

    //! @brief Returns the number of completion signatures in the set.
    //! @return The number of signatures.
    static constexpr auto __size() noexcept -> std::size_t {
      return sizeof...(_Sigs);
    }

    //! @brief Counts the number of signatures with the given tag.
    //! @tparam _Tag The tag to count (e.g., set_value, set_error, set_stopped).
    //! @return The number of signatures with the given tag.
    template <class _Tag>
    [[nodiscard]]
    static consteval auto __count(_Tag) noexcept -> std::size_t {
      if constexpr (_Tag{} == set_value) {
        return __v<typename __partitioned::__t::__count_values>;
      } else if constexpr (_Tag{} == set_error) {
        return __v<typename __partitioned::__t::__count_errors>;
      } else {
        return __v<typename __partitioned::__t::__count_stopped>;
      }
    }

    //! @brief Checks if the set contains the given signature.
    //! @tparam _Sig The signature type to check.
    //! @return true if the signature is present, false otherwise.
    template <class _Sig>
    [[nodiscard]]
    static consteval auto __contains(_Sig* = nullptr) noexcept -> bool {
      return __mset_contains<__t<__type_set>, _Sig>;
    }

    //! @brief Applies a callable to all signatures in the set.
    //! @tparam _Fn The callable to apply.
    //! @param __fn The callable instance.
    //! @return The result of calling __fn with all signatures as arguments.
    template <class _Fn>
    static consteval auto __apply(_Fn __fn) -> __call_result_t<_Fn, _Sigs*...> {
      return __fn(static_cast<_Sigs*>(nullptr)...);
    }

    //! @brief Filters the set using a predicate, returning a new set with only matching
    //! signatures.
    //! @tparam _Fn The predicate type. Must be empty and trivially constructible. Must
    //! satisfy `(std::predicate<_Fn, _Sigs*> &&...)`.
    //! @return A new completion_signatures set with only the signatures for which the
    //! predicate returns true.
    template <class _Fn>
    [[nodiscard]]
    static consteval auto __filter(_Fn) {
      static_assert(
        std::is_empty_v<_Fn> && std::is_trivially_constructible_v<_Fn>,
        "The filter function must be empty and trivially constructible.");
      return __concat_completion_signatures<__detail::__filer_one_t<_Fn, _Sigs>...>{};
    }

    //! @brief Selects all signatures with the given tag.
    //! @tparam _Tag The tag to select (e.g., set_value, set_error, set_stopped).
    //! @return A new completion_signatures set containing only signatures with the given
    //! tag.
    template <__completion_tag _Tag>
    [[nodiscard]]
    static consteval auto __select(_Tag) noexcept {
      if constexpr (_Tag{} == set_value) {
        return __value_types_t<
          completion_signatures,
          __qf<set_value_t>,
          __qq<stdexec::completion_signatures>
        >{};
      } else if constexpr (_Tag{} == set_error) {
        return __error_types_t<
          completion_signatures,
          __qq<stdexec::completion_signatures>,
          __qf<set_error_t>
        >{};
      } else {
        return __stopped_types_t<completion_signatures, __qq<stdexec::completion_signatures>>{};
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
      -> __call_result_t<_Reduce, __call_result_t<_Transform, _Sigs*>...> {
      return __reduce(__transform(static_cast<_Sigs*>(nullptr))...);
    }

    template <class... _OtherSigs>
    [[nodiscard]]
    consteval bool operator==(completion_signatures<_OtherSigs...> const &) const noexcept {
      using __other_set = completion_signatures<_OtherSigs...>::__type_set::__t;
      return __mset_eq<__t<__type_set>, __other_set>;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // __gather_completion_signatures
  namespace __detail {
    template <class _WantedTag>
    struct __gather_sigs_fn;

    template <>
    struct __gather_sigs_fn<set_value_t> {
      template <class _Sigs, class _Tuple, class _Variant>
      using __f = __value_types_t<_Sigs, _Tuple, _Variant>;
    };

    template <>
    struct __gather_sigs_fn<set_error_t> {
      template <class _Sigs, class _Tuple, class _Variant>
      using __f = __error_types_t<_Sigs, _Variant, _Tuple>;
    };

    template <>
    struct __gather_sigs_fn<set_stopped_t> {
      template <class _Sigs, class _Tuple, class _Variant>
      using __f = __stopped_types_t<_Sigs, _Variant, __minvoke<_Tuple>>;
    };
  } // namespace __detail

  template <class _WantedTag, class _Sigs, class _Tuple, class _Variant>
  using __gather_completions =
    typename __detail::__gather_sigs_fn<_WantedTag>::template __f<_Sigs, _Tuple, _Variant>;

  template <class _WantedTag, class _Sender, class _Env, class _Tuple, class _Variant>
  using __gather_completions_of =
    __gather_completions<_WantedTag, __completion_signatures_of_t<_Sender, _Env>, _Tuple, _Variant>;

  namespace __detail {
    template <class _Tag, class _Sigs>
    extern const std::size_t __count_of;

    template <class _Sigs>
    inline constexpr std::size_t __count_of<set_value_t, _Sigs> =
      __v<typename __sigs::__partitions_of_t<_Sigs>::__count_values>;

    template <class _Sigs>
    inline constexpr std::size_t __count_of<set_error_t, _Sigs> =
      __v<typename __sigs::__partitions_of_t<_Sigs>::__count_errors>;

    template <class _Sigs>
    inline constexpr std::size_t __count_of<set_stopped_t, _Sigs> =
      __v<typename __sigs::__partitions_of_t<_Sigs>::__count_stopped>;
  } // namespace __detail

  template <class _Tag, class _Sender, class... _Env>
  using __count_of =
    __msize_t<__detail::__count_of<_Tag, __completion_signatures_of_t<_Sender, _Env...>>>;

  template <class _Sender, class... _Env>
  using __single_sender_value_t = __value_types_t<
    __completion_signatures_of_t<_Sender, _Env...>,
    __qq<__msingle>,
    __qq<__msingle>
  >;

  template <class _Sender, class... _Env>
  using __single_value_variant_sender_t =
    __value_types_t<__completion_signatures_of_t<_Sender, _Env...>, __qq<__types>, __qq<__msingle>>;

  template <class _Sender, class... _Env>
  using __unrecognized_sender_error =
    __mexception<_UNRECOGNIZED_SENDER_TYPE_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>...>;

  // Below is the definition of the STDEXEC_COMPLSIGS_LET portability macro. It
  // is used to check that an expression's type is a valid completion_signature
  // specialization.
  //
  // USAGE:
  //
  //   STDEXEC_COMPLSIGS_LET(auto(__cs) = <expression>)
  //   {
  //     // __cs is guaranteed to be a specialization of completion_signatures.
  //   }
  //
  // When constexpr exceptions are available (C++26), the macro simply expands to
  // the moral equivalent of:
  //
  //   // With constexpr exceptions:
  //   auto __cs = <expression>; // throws if __cs is not a completion_signatures
  //
  // When constexpr exceptions are not available, the macro expands to:
  //
  //   // Without constexpr exceptions:
  //   if constexpr (auto __cs = <expression>; !__valid_completion_signatures<decltype(__cs)>)
  //   {
  //     return __cs;
  //   }
  //   else

#if STDEXEC_STD_NO_EXCEPTIONS()                                                                    \
  && __cpp_constexpr_exceptions >= 202411L // C++26, https://wg21.link/p3068

#  define STDEXEC_COMPLSIGS_LET(...)                                                               \
    if constexpr ([[maybe_unused]] __VA_ARGS__; false) {                                           \
    } else

  template <class _Sndr>
  [[noreturn, nodiscard]]
  consteval auto __dependent_sender() -> completion_signatures<> {
    throw __dependent_sender_error<_Sndr>{};
  }

#else // ^^^ constexpr exceptions ^^^ / vvv no constexpr exceptions vvv

#  define STDEXEC_PP_EAT_AUTO_auto(_ID)    _ID STDEXEC_EAT STDEXEC_LPAREN
#  define STDEXEC_PP_EXPAND_AUTO_auto(_ID) auto _ID
#  define STDEXEC_COMPLSIGS_LET_ID(...)                                                            \
    STDEXEC_EXPAND(STDEXEC_CAT(STDEXEC_PP_EAT_AUTO_, __VA_ARGS__) STDEXEC_RPAREN)

#  define STDEXEC_COMPLSIGS_LET(...)                                                               \
    if constexpr (STDEXEC_CAT(STDEXEC_PP_EXPAND_AUTO_, __VA_ARGS__);                               \
                  !stdexec::__valid_completion_signatures<decltype(STDEXEC_COMPLSIGS_LET_ID(       \
                    __VA_ARGS__))>) {                                                              \
      return STDEXEC_COMPLSIGS_LET_ID(__VA_ARGS__);                                                \
    } else

  template <class _Sndr>
  [[nodiscard]]
  consteval auto __dependent_sender() noexcept -> __dependent_sender_error<_Sndr> {
    return __dependent_sender_error<_Sndr>{};
  }

#endif // ^^^ no constexpr exceptions ^^^
} // namespace stdexec
