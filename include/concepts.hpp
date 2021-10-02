/*
 * Copyright (c) NVIDIA
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

#include <version>

#if __has_include(<concepts>) && __cpp_lib_concepts	>= 202002
#include <concepts>
#else
#include <type_traits>

namespace std {
  // C++20 concepts
  #if defined(__clang__)
  template<class A, class B>
    concept same_as = __is_same(A, B) && __is_same(B, A);
  #elif defined(__GNUC__)
  template<class A, class B>
    concept same_as = __is_same_as(A, B) && __is_same_as(B, A);
  #else
  template<class A, class B>
    inline constexpr bool __same_as_v = false;
  template<class A>
    inline constexpr bool __same_as_v<A, A> = true;

  template<class A, class B>
    concept same_as = __same_as_v<A, B> && __same_as_v<B, A>;
  #endif

  template<class A, class B>
    concept derived_from =
      is_base_of_v<B, A> &&
      is_convertible_v<const volatile A*, const volatile B*>;

  template<class From, class To>
    concept convertible_to =
      is_convertible_v<From, To> &&
      requires(From (&f)()) {
        static_cast<To>(f());
      };

  template<class T>
    concept equality_comparable =
      requires(const remove_reference_t<T>& t) {
        { t == t } -> convertible_to<bool>;
        { t != t } -> convertible_to<bool>;
      };

  template<class T>
    concept destructible = is_nothrow_destructible_v<T>;

  template<class T, class... As>
    concept constructible_from =
      destructible<T> && is_constructible_v<T, As...>;

  template<class T>
    concept move_constructible = constructible_from<T, T>;

  template<class T>
    concept copy_constructible =
      move_constructible<T> &&
      constructible_from<T, T const&>;

  template<class F, class... As>
    concept invocable = requires {
      typename invoke_result_t<F, As...>;
    };
}
#endif

namespace std {
  template<class T, class U>
    concept __same_ = same_as<remove_cvref_t<T>, U>;

  template <class T, class... As>
    concept __one_of = (same_as<T, As> ||...);

  // Not exactly right, but close.
  template <class T>
    concept __boolean_testable =
      convertible_to<T, bool>;
}
