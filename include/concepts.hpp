#pragma once

#include <version>

#if __has_include(<concepts>) && __cpp_lib_concepts	>= 202002
#include <concepts>
#else
#include <type_traits>

namespace std {
  // C++20 concepts
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
}
