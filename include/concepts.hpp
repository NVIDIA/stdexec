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
  template<class _A, class _B>
    concept same_as = __is_same(_A, _B) && __is_same(_B, _A);
  #elif defined(__GNUC__)
  template<class _A, class _B>
    concept same_as = __is_same_as(_A, _B) && __is_same_as(_B, _A);
  #else
  template<class _A, class _B>
    inline constexpr bool __same_as_v = false;
  template<class _A>
    inline constexpr bool __same_as_v<_A, _A> = true;

  template<class _A, class _B>
    concept same_as = __same_as_v<_A, _B> && __same_as_v<_B, _A>;
  #endif

  template <class T>
    concept integral = std::is_integral_v<T>;

  template<class _A, class _B>
    concept derived_from =
      is_base_of_v<_B, _A> &&
      is_convertible_v<const volatile _A*, const volatile _B*>;

  template<class _From, class _To>
    concept convertible_to =
      is_convertible_v<_From, _To> &&
      requires(_From (&__fun)()) {
        static_cast<_To>(__fun());
      };

  template<class _T>
    concept equality_comparable =
      requires(const remove_reference_t<_T>& __t) {
        { __t == __t } -> convertible_to<bool>;
        { __t != __t } -> convertible_to<bool>;
      };

  template<class _T>
    concept destructible = is_nothrow_destructible_v<_T>;

  template<class _T, class... _As>
    concept constructible_from =
      destructible<_T> && is_constructible_v<_T, _As...>;

  template<class _T>
    concept move_constructible = constructible_from<_T, _T>;

  template<class _T>
    concept copy_constructible =
      move_constructible<_T> &&
      constructible_from<_T, _T const&>;

  template<class _F, class... _As>
    concept invocable = requires {
      typename invoke_result_t<_F, _As...>;
    };
}
#endif

namespace std {
  template<class _T, class _U>
    concept __decays_to =
      same_as<decay_t<_T>, _U>;

  template <class _C>
    concept __class =
      is_class_v<_C> && __decays_to<_C, _C>;

  template <class _T, class... _As>
    concept __one_of =
      (same_as<_T, _As> ||...);

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

  template <class...>
    concept __typename = true;

} // namespace std
