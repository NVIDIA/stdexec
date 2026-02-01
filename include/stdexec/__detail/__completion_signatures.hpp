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
#include "__utility.hpp"

#include <type_traits>

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __cmplsigs {
    template <class _Sig>
    inline constexpr bool __is_compl_sig = false;
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;
  } // namespace __cmplsigs

  template <class _Sig>
  concept __completion_signature = __cmplsigs::__is_compl_sig<_Sig>;

  namespace __cmplsigs {
    // The following code is used to normalize completion signatures. "Normalization" means that
    // that rvalue-references are stripped from the types in the completion signatures. For example,
    // the completion signature `set_value_t(int &&)` would be normalized to `set_value_t(int)`,
    // but `set_value_t(int)` and `set_value_t(int &)` would remain unchanged.
    template <class _Tag, class... _Args>
    constexpr auto __normalize_sig_impl(_Args&&...) -> _Tag (*)(_Args...);

    template <class _Tag, class... _Args>
    constexpr auto __normalize_sig(_Tag (*)(_Args...))
      -> decltype(__cmplsigs::__normalize_sig_impl<_Tag>(__declval<_Args>()...));

    template <class... _Sigs>
    constexpr auto __repack_completions(_Sigs*...) -> completion_signatures<_Sigs...>;

    template <class... _Sigs>
    constexpr auto __normalize_completions(completion_signatures<_Sigs...>*)
      -> decltype(__cmplsigs::__repack_completions(
        __cmplsigs::__normalize_sig(static_cast<_Sigs*>(nullptr))...));

    template <class _Completions>
    using __normalize_completions_t = decltype(__cmplsigs::__normalize_completions(
      static_cast<_Completions*>(nullptr)));

    template <class _Sig>
    using __normalize_sig_t = decltype(__cmplsigs::__normalize_sig(static_cast<_Sig*>(nullptr)));
  } // namespace __cmplsigs

  template <class... _SigPtrs>
  using __completion_signature_ptrs_t = decltype(__cmplsigs::__repack_completions(
    static_cast<_SigPtrs>(nullptr)...));

#if STDEXEC_NO_STD_CONSTEXPR_EXCEPTIONS()

  template <class, class... _What, class... _Values>
  [[nodiscard]]
  consteval auto __throw_compile_time_error_r(_Values...) -> __mexception<_What...> {
    return {};
  }

  template <class... _What, class... _Values>
  [[nodiscard]]
  consteval auto __throw_compile_time_error(_Values...) -> __mexception<_What...> {
    return {};
  }

#else  // ^^^ no constexpr exceptions ^^^ / vvv constexpr exceptions vvv

  // C++26, https://wg21.link/p3068
  template <class _Return, class _What, class... _More, class... _Values>
  [[noreturn, nodiscard]]
  consteval auto __throw_compile_time_error_r([[maybe_unused]] _Values... __values) -> _Return {
    if constexpr (__same_as<_What, dependent_sender_error>) {
      throw __mexception<dependent_sender_error, _More...>();
    } else if constexpr (sizeof...(_Values) == 1) {
      throw __sender_type_check_failure<_Values..., _What, _More...>(__values...);
    } else {
      throw __sender_type_check_failure<__tuple<_Values...>, _What, _More...>(__tuple{__values...});
    }
  }

  template <class _What, class... _More, class... _Values>
  [[noreturn, nodiscard]]
  consteval auto
    __throw_compile_time_error([[maybe_unused]] _Values... __values) -> completion_signatures<> {
    return;
  }
#endif // ^^^ constexpr exceptions ^^^

  template <class _Return, class... _What>
  [[nodiscard]]
  consteval auto __throw_compile_time_error_r(__mexception<_What...>) {
    return STDEXEC::__throw_compile_time_error_r<_Return, _What...>();
  }

  template <class... _What>
  [[nodiscard]]
  consteval auto __throw_compile_time_error(__mexception<_What...>) {
    return STDEXEC::__throw_compile_time_error<_What...>();
  }

  namespace __cmplsigs {
    // __partitions is a cache of completion signatures for fast access. The
    // completion_signatures<Sigs...>::__partitioned nested struct contains an alias to a
    // __partitions specialization. If the cache is never accessed, it is never
    // instantiated.
    template <
      class _ValueTuplesList = __mlist<>,
      class _ErrorsList = __mlist<>,
      class _StoppedList = __mlist<>
    >
    struct __partitions;

    template <class... _ValueTuples, class... _Errors, class... _Stopped>
    struct __partitions<__mlist<_ValueTuples...>, __mlist<_Errors...>, __mlist<_Stopped...>> {
      template <class _Tuple, class _Variant>
      using __value_types = _Variant::template __f<__mapply<_Tuple, _ValueTuples>...>;

      template <class _Variant, class _Transform = __q1<__midentity>>
      using __error_types = _Variant::template __f<typename _Transform::template __f<_Errors>...>;

      template <class _Variant, class _Type = set_stopped_t()>
      using __stopped_types = _Variant::template __f<__msecond<_Stopped, _Type>...>;

      using __count_values = __msize_t<sizeof...(_ValueTuples)>;
      using __count_errors = __msize_t<sizeof...(_Errors)>;
      using __count_stopped = __msize_t<sizeof...(_Stopped)>;

      struct __decay_copyable {
        // These aliases are placed in a separate struct to avoid computing them
        // if they are not needed.
        struct __values {
          static constexpr bool value =
            (__mapply_q<__decay_copyable_t, _ValueTuples>::value && ...);
        };
        struct __errors {
          static constexpr bool value = STDEXEC::__decay_copyable<_Errors...>;
        };
        struct __all : __mand_t<__values, __errors> { };
      };

      struct __nothrow_decay_copyable {
        // These aliases are placed in a separate struct to avoid computing them
        // if they are not needed.
        struct __values {
          static constexpr bool value =
            (__mapply_q<__nothrow_decay_copyable_t, _ValueTuples>::value && ...);
        };
        struct __errors {
          static constexpr bool value = STDEXEC::__nothrow_decay_copyable<_Errors...>;
        };
        struct __all : __mand_t<__values, __errors> { };
      };
    };

    template <class _Tag>
    struct __partitioned_fold_fn;

    template <>
    struct __partitioned_fold_fn<set_value_t> {
      template <class... _ValueTuples, class _Errors, class _Stopped, class _Values>
      constexpr auto operator()(
        __partitions<__mlist<_ValueTuples...>, _Errors, _Stopped>&,
        __undefined<_Values>&) const
        -> __undefined<__partitions<__mlist<_ValueTuples..., _Values>, _Errors, _Stopped>>&;
    };

    template <>
    struct __partitioned_fold_fn<set_error_t> {
      template <class _Values, class... _Errors, class _Stopped, class _Error>
      constexpr auto operator()(
        __partitions<_Values, __mlist<_Errors...>, _Stopped>&,
        __undefined<__mlist<_Error>>&) const
        -> __undefined<__partitions<_Values, __mlist<_Errors..., _Error>, _Stopped>>&;
    };

    template <>
    struct __partitioned_fold_fn<set_stopped_t> {
      template <class _Values, class _Errors, class _Stopped>
      constexpr auto operator()(__partitions<_Values, _Errors, _Stopped>&, __ignore) const
        -> __undefined<__partitions<_Values, _Errors, __mlist<set_stopped_t()>>>&;
    };

    // The following overload of binary operator* is used to build up the cache of completion
    // signatures. We fold over operator*, accumulating the completion signatures in the
    // cache. `__undefined` is used here to prevent the instantiation of the intermediate
    // types.
    template <class _Partitioned, class _Tag, class... _Args>
    constexpr auto operator*(__undefined<_Partitioned>&, _Tag (*)(_Args...)) -> __call_result_t<
      __partitioned_fold_fn<_Tag>,
      _Partitioned&,
      __undefined<__mlist<_Args...>>&
    >;

    // This function declaration is used to extract the cache from the `__undefined` type.
    template <class _Partitioned>
    constexpr auto __unpack_partitioned_completions(__undefined<_Partitioned>&) -> _Partitioned;

    template <class... _Sigs>
    using __partition_completion_signatures_t = //
      decltype(__cmplsigs::__unpack_partitioned_completions(
        (__declval<__undefined<__partitions<>>&>() * ... * static_cast<_Sigs*>(nullptr))));

    template <class _Completions>
    using __partitions_of_t = _Completions::__partitioned::__t;
  } // namespace __cmplsigs

  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // completion signatures type traits
  template <
    class _Sigs,
    class _Tuple = __qq<__decayed_std_tuple>,
    class _Variant = __qq<__std_variant>
  >
  using __value_types_t =
    __cmplsigs::__partitions_of_t<_Sigs>::template __value_types<_Tuple, _Variant>;

  template <class _Sigs, class _Variant = __qq<__std_variant>, class _Transform = __q1<__midentity>>
  using __error_types_t =
    __cmplsigs::__partitions_of_t<_Sigs>::template __error_types<_Variant, _Transform>;

  template <class _Sigs, class _Variant, class _Type = __fn_t<set_stopped_t>>
  using __stopped_types_t =
    __cmplsigs::__partitions_of_t<_Sigs>::template __stopped_types<_Variant, _Type>;

  template <__valid_completion_signatures _Sigs>
  inline constexpr bool __sends_stopped =
    __cmplsigs::__partitions_of_t<_Sigs>::__count_stopped::value != 0;

  STDEXEC_PRAGMA_PUSH()
  STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
  STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  // concat_completion_signatures
  template <class... _Sigs>
  using __concat_completion_signatures_t =
    __mconcat<__qq<completion_signatures>>::__f<__mconcat<__qq<__mmake_set>>::__f<_Sigs...>>;

  namespace __detail {
    struct __concat_completion_signatures_fn {
      template <STDEXEC::__valid_completion_signatures... Sigs>
      [[nodiscard]]
      consteval auto operator()(Sigs...) const noexcept {
        return __concat_completion_signatures_t<Sigs...>{};
      }

      template <class... Errors>
      [[nodiscard]]
      consteval auto operator()(Errors...) const noexcept {
        // NB: this uses an overloaded comma operator on the _ERROR_ type to find an error
        // in a pack of types.
        using __error_t = decltype(+(Errors{}, ...));
        static_assert(__merror<__error_t>);
        return STDEXEC::__throw_compile_time_error(__error_t());
      }
    };
  } // namespace __detail

  inline constexpr __detail::__concat_completion_signatures_fn __concat_completion_signatures{};

  STDEXEC_PRAGMA_POP()

  namespace __detail {
    template <class _Fn, class _Sig>
    concept __filter_pass = requires { requires _Fn()(__midentity<_Sig*>()); };

    template <class _Fn, class _Sig>
    using __filer_one_t =
      __if_c<__filter_pass<_Fn, _Sig>, completion_signatures<_Sig>, completion_signatures<>>;
  } // namespace __detail

  struct _IN_COMPLETION_SIGNATURES_APPLY_;

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
    static_assert(
      (__completion_signature<_Sigs> && ...),
      "All types in completion_signatures must be valid completion signatures.");

    //! @brief Partitioned view of the completion signatures for efficient querying.
    struct __partitioned {
      // This is defined in a nested struct to avoid computing these types if they are not
      // needed.
      using __t = __cmplsigs::__partition_completion_signatures_t<_Sigs...>;
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
        return __partitioned::__t::__count_values::value;
      } else if constexpr (_Tag{} == set_error) {
        return __partitioned::__t::__count_errors::value;
      } else {
        return __partitioned::__t::__count_stopped::value;
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
    static consteval auto __apply(_Fn __fn) {
      if constexpr (__callable<_Fn, _Sigs*...>) {
        return __fn(static_cast<_Sigs*>(nullptr)...);
      } else {
        return STDEXEC::__throw_compile_time_error<
          _WHERE_(_IN_COMPLETION_SIGNATURES_APPLY_),
          _FUNCTION_IS_NOT_CALLABLE_WITH_THE_GIVEN_ARGUMENTS_,
          _WITH_FUNCTION_(_Fn),
          _WITH_ARGUMENTS_(_Sigs * ...)
        >();
      }
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
      return __concat_completion_signatures_t<__detail::__filer_one_t<_Fn, _Sigs>...>{};
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
          __qq<STDEXEC::completion_signatures>
        >{};
      } else if constexpr (_Tag{} == set_error) {
        return __error_types_t<
          completion_signatures,
          __qq<STDEXEC::completion_signatures>,
          __qf<set_error_t>
        >{};
      } else {
        return __stopped_types_t<completion_signatures, __qq<STDEXEC::completion_signatures>>{};
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
  // __gather_completion_signatures_t
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
  using __gather_completions_t =
    __detail::__gather_sigs_fn<_WantedTag>::template __f<_Sigs, _Tuple, _Variant>;

  namespace __detail {
    template <class _Tag, class _Sigs>
    inline constexpr std::size_t __count_of = 0;

    template <class _Sigs>
    inline constexpr std::size_t __count_of<set_value_t, _Sigs> =
      __cmplsigs::__partitions_of_t<_Sigs>::__count_values::value;

    template <class _Sigs>
    inline constexpr std::size_t __count_of<set_error_t, _Sigs> =
      __cmplsigs::__partitions_of_t<_Sigs>::__count_errors::value;
    template <class _Sigs>
    inline constexpr std::size_t __count_of<set_stopped_t, _Sigs> =
      __cmplsigs::__partitions_of_t<_Sigs>::__count_stopped::value;
  } // namespace __detail

  // Below is the definition of the STDEXEC_COMPLSIGS_LET portability macro. It
  // is used to check that an expression's type is a valid completion_signature
  // specialization.
  //
  // USAGE:
  //
  //   STDEXEC_COMPLSIGS_LET(__cs, <expression>)
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

#if STDEXEC_NO_STD_CONSTEXPR_EXCEPTIONS()

#  define STDEXEC_COMPLSIGS_LET(_ID, ...)                                                          \
    if constexpr ([[maybe_unused]]                                                                 \
                  auto _ID = __VA_ARGS__;                                                          \
                  !STDEXEC::__valid_completion_signatures<decltype(_ID)>) {                        \
      return _ID;                                                                                  \
    } else

  template <class _Sndr>
  [[nodiscard]]
  consteval auto __dependent_sender() noexcept -> __dependent_sender_error_t<_Sndr> {
    return {};
  }

#else // ^^^ no constexpr exceptions ^^^ / vvv constexpr exceptions vvv

#  define STDEXEC_COMPLSIGS_LET(_ID, ...)                                                          \
    if constexpr ([[maybe_unused]]                                                                 \
                  auto _ID = __VA_ARGS__;                                                          \
                  false) {                                                                         \
    } else

  template <class _Sndr>
  [[noreturn, nodiscard]]
  consteval auto __dependent_sender() -> completion_signatures<> {
    throw __dependent_sender_error_t<_Sndr>{};
  }

#endif // ^^^ constexpr exceptions ^^^
} // namespace STDEXEC
