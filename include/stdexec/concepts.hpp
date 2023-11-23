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

#include "__detail/__meta.hpp"
#include "__detail/__concepts.hpp"

namespace stdexec::__std_concepts {
  // Make sure we're using a same_as concept that doesn't instantiate std::is_same
  template <class _Ap, class _Bp>
  concept same_as = __same_as<_Ap, _Bp> && __same_as<_Bp, _Ap>;

#if STDEXEC_HAS_STD_CONCEPTS_HEADER()

  using std::integral;
  using std::derived_from;
  using std::convertible_to;
  using std::equality_comparable;

#else

  template <class T>
  concept integral = std::is_integral_v<T>;

  template <class _Ap, class _Bp>
  concept derived_from =            //
    STDEXEC_IS_BASE_OF(_Bp, _Ap) && //
    STDEXEC_IS_CONVERTIBLE_TO(const volatile _Ap*, const volatile _Bp*);

  template <class _From, class _To>
  concept convertible_to =                   //
    STDEXEC_IS_CONVERTIBLE_TO(_From, _To) && //
    requires(_From (&__fun)()) { static_cast<_To>(__fun()); };

  template <class _Ty>
  concept equality_comparable = //
    requires(__cref_t<_Ty> __t) {
      { __t == __t } -> convertible_to<bool>;
      { __t != __t } -> convertible_to<bool>;
    };
#endif
} // namespace stdexec::__std_concepts

namespace stdexec {
  using namespace __std_concepts;

  // Avoid using libstdc++'s object concepts because they instantiate a
  // lot of templates.
  template <class _Ty>
  inline constexpr bool __destructible_ = //
    requires {
      { ((_Ty && (*) () noexcept) nullptr)().~_Ty() } noexcept;
    };
  template <class _Ty>
  inline constexpr bool __destructible_<_Ty&> = true;
  template <class _Ty>
  inline constexpr bool __destructible_<_Ty&&> = true;
  template <class _Ty, std::size_t _Np>
  inline constexpr bool __destructible_<_Ty[_Np]> = __destructible_<_Ty>;

  template <class T>
  concept destructible = __destructible_<T>;

#if STDEXEC_HAS_BUILTIN(__is_constructible)
  template <class _Ty, class... _As>
  concept constructible_from = //
    destructible<_Ty> &&       //
    __is_constructible(_Ty, _As...);
#else
  template <class _Ty, class... _As>
  concept constructible_from = //
    destructible<_Ty> &&       //
    std::is_constructible_v<_Ty, _As...>;
#endif

  template <class _Ty>
  concept default_initializable = //
    constructible_from<_Ty> &&    //
    requires { _Ty{}; } &&        //
    requires { ::new _Ty; };

  template <class _Ty>
  concept move_constructible = //
    constructible_from<_Ty, _Ty>;

  template <class _Ty>
  concept copy_constructible = //
    move_constructible<_Ty>    //
    && constructible_from<_Ty, _Ty const &>;

  template <class _LHS, class _RHS >
  concept assignable_from = //
    same_as<_LHS, _LHS&> && //
    // std::common_reference_with<
    //   const std::remove_reference_t<_LHS>&,
    //   const std::remove_reference_t<_RHS>&> &&
    requires(_LHS __lhs, _RHS&& __rhs) {
      { __lhs = ((_RHS&&) __rhs) } -> same_as<_LHS>;
    };

  namespace __swap {
    using std::swap;

    template <class _Ty, class _Uy>
    concept swappable_with =           //
      requires(_Ty&& __t, _Uy&& __u) { //
        swap((_Ty&&) __t, (_Uy&&) __u);
      };

    inline constexpr auto const __fn = //
      []<class _Ty, swappable_with<_Ty> _Uy>(_Ty&& __t, _Uy&& __u) noexcept(
        noexcept(swap((_Ty&&) __t, (_Uy&&) __u))) {
        swap((_Ty&&) __t, (_Uy&&) __u);
      };
  }

  using __swap::swappable_with;
  inline constexpr auto const & swap = __swap::__fn;

  template <class _Ty>
  concept swappable = //
    swappable_with<_Ty, _Ty>;

  template < class _Ty >
  concept movable =               //
    std::is_object_v<_Ty> &&      //
    move_constructible<_Ty> &&    //
    assignable_from<_Ty&, _Ty> && //
    swappable<_Ty>;

  template <class _Ty>
  concept copyable =                     //
    copy_constructible<_Ty> &&           //
    movable<_Ty> &&                      //
    assignable_from<_Ty&, _Ty&> &&       //
    assignable_from<_Ty&, const _Ty&> && //
    assignable_from<_Ty&, const _Ty>;

  template <class _Ty>
  concept semiregular = //
    copyable<_Ty> &&    //
    default_initializable<_Ty>;

  template <class _Ty>
  concept regular =     //
    semiregular<_Ty> && //
    equality_comparable<_Ty>;

  // Not exactly right, but close.
  template <class _Ty>
  concept __boolean_testable_ = convertible_to<_Ty, bool>;

  template < class T, class U >
  concept __partially_ordered_with = //
    requires(__cref_t<T> t, __cref_t<U> u) {
      { t < u } -> __boolean_testable_;
      { t > u } -> __boolean_testable_;
      { t <= u } -> __boolean_testable_;
      { t >= u } -> __boolean_testable_;
      { u < t } -> __boolean_testable_;
      { u > t } -> __boolean_testable_;
      { u <= t } -> __boolean_testable_;
      { u >= t } -> __boolean_testable_;
    };

  template < class _Ty >
  concept totally_ordered =     //
    equality_comparable<_Ty> && //
    __partially_ordered_with<_Ty, _Ty>;

  template <class _Ty>
  concept __movable_value =               //
    move_constructible<__decay_t<_Ty>> && //
    constructible_from<__decay_t<_Ty>, _Ty>;

  template <class _Ty>
  concept __nothrow_movable_value = //
    __movable_value<_Ty> &&         //
    requires(_Ty&& __t) {
      { __decay_t<_Ty>{__decay_t<_Ty>{(_Ty&&) __t}} } noexcept;
    };

#if STDEXEC_HAS_BUILTIN(__is_nothrow_constructible)
  template <class _Ty, class... _As>
  concept __nothrow_constructible_from =
    constructible_from<_Ty, _As...> && __is_nothrow_constructible(_Ty, _As...);
#else
  template <class _Ty, class... _As>
  concept __nothrow_constructible_from =
    constructible_from<_Ty, _As...> && std::is_nothrow_constructible_v<_Ty, _As...>;
#endif

  template <class _Ty>
  concept __nothrow_move_constructible = __nothrow_constructible_from<_Ty, _Ty>;

  template <class _Ty>
  concept __nothrow_copy_constructible = __nothrow_constructible_from<_Ty, const _Ty&>;

  template <class _Ty>
  concept __decay_copyable = constructible_from<__decay_t<_Ty>, _Ty>;

  template <class _Ty>
  concept __nothrow_decay_copyable = __nothrow_constructible_from<__decay_t<_Ty>, _Ty>;

  template <class _Ty, class _Up>
  concept __decays_to_derived_from = derived_from<__decay_t<_Ty>, _Up>;

} // namespace stdexec

// #if !STDEXEC_HAS_STD_CONCEPTS_HEADER()
// namespace std {
//   using namespace stdexec::__std_concepts;
// }
// #endif
