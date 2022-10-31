/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#if __cpp_concepts < 201907L
#error This library requires support for C++20 concepts
#endif

#include <version>

// Perhaps the stdlib lacks support for concepts though:
#if __has_include(<concepts>) && __cpp_lib_concepts >= 202002
#define STDEXEC_HAS_STD_CONCEPTS_HEADER() 1
#else
#define STDEXEC_HAS_STD_CONCEPTS_HEADER() 0
#endif

#if STDEXEC_HAS_STD_CONCEPTS_HEADER()
#include <concepts>
#else
#include <type_traits>
#endif

namespace stdexec::__std_concepts {
  #if defined(__clang__)
  template<class _A, class _B>
    concept __same_as = __is_same(_A, _B);
  #elif defined(__GNUC__)
  template<class _A, class _B>
    concept __same_as = __is_same_as(_A, _B);
  #else
  template<class _A, class _B>
    inline constexpr bool __same_as = false;
  template<class _A>
    inline constexpr bool __same_as<_A, _A> = true;
  #endif

  // Make sure we're using a same_as concept that doesn't instantiate std::is_same
  template <class _A, class _B>
    concept same_as = __same_as<_A, _B> && __same_as<_B, _A>;

#if STDEXEC_HAS_STD_CONCEPTS_HEADER()

  using std::integral;
  using std::derived_from;
  using std::convertible_to;
  using std::equality_comparable;
  using std::destructible;
  using std::constructible_from;
  using std::move_constructible;
  using std::copy_constructible;

#else

  template <class T>
    concept integral = std::is_integral_v<T>;

  template<class _A, class _B>
    concept derived_from =
      std::is_base_of_v<_B, _A> &&
      std::is_convertible_v<const volatile _A*, const volatile _B*>;

  template<class _From, class _To>
    concept convertible_to =
      std::is_convertible_v<_From, _To> &&
      requires(_From (&__fun)()) {
        static_cast<_To>(__fun());
      };

  template<class _T>
    concept equality_comparable =
      requires(const std::remove_reference_t<_T>& __t) {
        { __t == __t } -> convertible_to<bool>;
        { __t != __t } -> convertible_to<bool>;
      };

  template<class _T>
    concept destructible = std::is_nothrow_destructible_v<_T>;

#if __has_builtin(__is_constructible)
  template<class _T, class... _As>
    concept constructible_from =
      destructible<_T> && __is_constructible(_T, _As...);
#else
  template<class _T, class... _As>
    concept constructible_from =
      destructible<_T> && is_constructible_v<_T, _As...>;
#endif

  template<class _T>
    concept move_constructible = constructible_from<_T, _T>;

  template<class _T>
    concept copy_constructible =
      move_constructible<_T> &&
      constructible_from<_T, _T const&>;
#endif
} // namespace stdexec::__std_concepts

namespace stdexec {
  using namespace __std_concepts;
  using std::decay_t;

  // // TODO: this makes nvc++ sad. Find out why.
  // template <class _Ty>
  //   _Ty __decay__(const _Ty&);
  // template <class _Ty>
  //   _Ty* __decay__(_Ty*);

  // template <class _Ty>
  //   auto __decay_(_Ty&&(*__fn)()) -> decltype((__decay__)(__fn()));
  // template <class>
  //   void __decay_(...);

  // template <class _Ty>
  //   using decay_t = decltype((__decay_<_Ty>)(0));

  // C++20 concepts
  template<class _T, class _U>
    concept __decays_to =
      same_as<decay_t<_T>, _U>;

  template <class>
    concept __true = true;

  template <class _C>
    concept __class =
      __true<int _C::*> && (!__same_as<const _C, _C>);

  template <class _T, class... _As>
    concept __one_of =
      (same_as<_T, _As> ||...);

  template <class _T, class... _Us>
    concept __all_of =
      (same_as<_T, _Us> &&...);

  template <class _T, class... _Us>
    concept __none_of =
      ((!same_as<_T, _Us>) &&...);

  // Not exactly right, but close.
  template <class _T>
    concept __boolean_testable_ =
      convertible_to<_T, bool>;

  template <class _T>
    concept __movable_value =
      move_constructible<decay_t<_T>> &&
      constructible_from<decay_t<_T>, _T>;

  template <class _Trait>
    concept __is_true = _Trait::value;

  template <class, template <class...> class>
    constexpr bool __is_instance_of_ = false;
  template <class... _As, template <class...> class _T>
    constexpr bool __is_instance_of_<_T<_As...>, _T> = true;

  template <class _Ty, template <class...> class _T>
    concept __is_instance_of =
      __is_instance_of_<_Ty, _T>;

  template <class _Ty, template <class...> class _T>
    concept __is_not_instance_of =
      !__is_instance_of<_Ty, _T>;

#if __has_builtin(__is_nothrow_constructible)
  template<class _T, class... _As>
    concept __nothrow_constructible_from =
      constructible_from<_T, _As...> && __is_nothrow_constructible(_T, _As...);
#else
  template<class _T, class... _As>
    concept __nothrow_constructible_from =
      constructible_from<_T, _As...> && std::is_nothrow_constructible_v<_T, _As...>;
#endif

  template <class _Ty>
    concept __nothrow_decay_copyable =
      __nothrow_constructible_from<decay_t<_Ty>, _Ty>;
} // namespace stdexec

#if !STDEXEC_HAS_STD_CONCEPTS_HEADER()
namespace std {
  using namespace stdexec::__std_concepts;
}
#endif
