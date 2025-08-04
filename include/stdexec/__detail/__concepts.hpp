/*
 * Copyright (c) 2023 NVIDIA Corporation
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

#if __cpp_concepts < 2019'07L
#  error This library requires support for C++20 concepts
#endif

#include "__config.hpp"
#include "__type_traits.hpp"

#include <version>

// Perhaps the stdlib lacks support for concepts though:
#if __has_include(<concepts>) && __cpp_lib_concepts >= 2020'02L
#  define STDEXEC_HAS_STD_CONCEPTS_HEADER() 1
#else
#  define STDEXEC_HAS_STD_CONCEPTS_HEADER() 0
#endif

#if STDEXEC_HAS_STD_CONCEPTS_HEADER()
#  include <concepts>
#else
#  include <type_traits>
#endif

namespace stdexec {
  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <class _Fun, class... _As>
  concept __callable = requires(_Fun&& __fun, _As&&... __as) {
    static_cast<_Fun &&>(__fun)(static_cast<_As &&>(__as)...);
  };
  template <class _Fun, class... _As>
  concept __nothrow_callable = __callable<_Fun, _As...> && requires(_Fun&& __fun, _As&&... __as) {
    { static_cast<_Fun &&>(__fun)(static_cast<_As &&>(__as)...) } noexcept;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <class...>
  struct __types;

  template <class... _Ts>
  concept __typename = requires {
    typename __types<_Ts...>; // NOLINT
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <class _Ap, class _Bp>
  concept __same_as = STDEXEC_IS_SAME(_Ap, _Bp);

  // Handy concepts
  template <class _Ty, class _Up>
  concept __decays_to = __same_as<__decay_t<_Ty>, _Up>;

  template <class _Ty, class _Up>
  concept __not_decays_to = !__decays_to<_Ty, _Up>;

  template <bool _TrueOrFalse>
  concept __satisfies = _TrueOrFalse;

  template <class...>
  concept __true = true;

  template <class _Cp>
  concept __class = __true<int _Cp::*> && (!__same_as<const _Cp, _Cp>);

  template <class _Ty, class... _As>
  concept __one_of = (__same_as<_Ty, _As> || ...);

  template <class _Ty, class... _Us>
  concept __all_of = (__same_as<_Ty, _Us> && ...);

  template <class _Ty, class... _Us>
  concept __none_of = ((!__same_as<_Ty, _Us>) && ...);

  template <class, template <class...> class>
  constexpr bool __is_instance_of_ = false;
  template <class... _As, template <class...> class _Ty>
  constexpr bool __is_instance_of_<_Ty<_As...>, _Ty> = true;

  template <class _Ay, template <class...> class _Ty>
  concept __is_instance_of = __is_instance_of_<_Ay, _Ty>;

  template <class _Ay, template <class...> class _Ty>
  concept __is_not_instance_of = !__is_instance_of<_Ay, _Ty>;
} // namespace stdexec

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
  concept derived_from = STDEXEC_IS_BASE_OF(_Bp, _Ap)
                      && STDEXEC_IS_CONVERTIBLE_TO(const volatile _Ap*, const volatile _Bp*);

  template <class _From, class _To>
  concept convertible_to = STDEXEC_IS_CONVERTIBLE_TO(_From, _To)
                        && requires(_From (&__fun)()) { static_cast<_To>(__fun()); };

  template <class _Ty>
  concept equality_comparable = requires(__cref_t<_Ty> __t) {
    { __t == __t } -> convertible_to<bool>;
    { __t != __t } -> convertible_to<bool>;
  };
#endif
} // namespace stdexec::__std_concepts

namespace stdexec {
  using namespace __std_concepts;

  // Avoid using libstdc++'s object concepts because they instantiate a
  // lot of templates.
#if STDEXEC_HAS_BUILTIN(__is_nothrow_destructible) || STDEXEC_MSVC()
  template <class _Ty>
  concept destructible = __is_nothrow_destructible(_Ty);
#else
  template <class _Ty>
  inline constexpr bool __destructible_ = requires(_Ty && (&__fn)() noexcept) {
    { __fn().~_Ty() } noexcept;
  };
  template <class _Ty>
  inline constexpr bool __destructible_<_Ty&> = true;
  template <class _Ty>
  inline constexpr bool __destructible_<_Ty&&> = true;
  template <class _Ty, std::size_t _Np>
  inline constexpr bool __destructible_<_Ty[_Np]> = __destructible_<_Ty>;

  template <class T>
  concept destructible = __destructible_<T>;
#endif

  template <class _Ty, class... _As>
  concept constructible_from = destructible<_Ty> && STDEXEC_IS_CONSTRUCTIBLE(_Ty, _As...);

  template <class _Ty>
  concept default_initializable = constructible_from<_Ty> && requires { _Ty{}; }
                               && requires { ::new _Ty; };

  template <class _Ty>
  concept move_constructible = constructible_from<_Ty, _Ty>;

  template <class _Ty>
  concept copy_constructible = move_constructible<_Ty> && constructible_from<_Ty, _Ty const &>;

  template <class _LHS, class _RHS>
  concept assignable_from = same_as<_LHS, _LHS&> &&
                            // std::common_reference_with<
                            //   const std::remove_reference_t<_LHS>&,
                            //   const std::remove_reference_t<_RHS>&> &&
                            requires(_LHS __lhs, _RHS&& __rhs) {
                              { __lhs = static_cast<_RHS &&>(__rhs) } -> same_as<_LHS>;
                            };

  namespace __swap {
    using std::swap;

    template <class _Ty, class _Uy>
    concept swappable_with = requires(_Ty&& __t, _Uy&& __u) {
      swap(static_cast<_Ty &&>(__t), static_cast<_Uy &&>(__u));
    };

    inline constexpr auto const __fn =
      []<class _Ty, swappable_with<_Ty> _Uy>(_Ty&& __t, _Uy&& __u) noexcept(
        noexcept(swap(static_cast<_Ty&&>(__t), static_cast<_Uy&&>(__u)))) {
        swap(static_cast<_Ty&&>(__t), static_cast<_Uy&&>(__u));
      };
  } // namespace __swap

  using __swap::swappable_with;
  inline constexpr auto const & swap = __swap::__fn;

  template <class _Ty>
  concept swappable = requires(_Ty& a, _Ty& b) { swap(a, b); };

  template <class _Ty>
  concept movable = std::is_object_v<_Ty> && move_constructible<_Ty> && assignable_from<_Ty&, _Ty>
                 && swappable<_Ty>;

  template <class _Ty>
  concept copyable = copy_constructible<_Ty> && movable<_Ty> && assignable_from<_Ty&, _Ty&>
                  && assignable_from<_Ty&, const _Ty&> && assignable_from<_Ty&, const _Ty>;

  template <class _Ty>
  concept semiregular = copyable<_Ty> && default_initializable<_Ty>;

  template <class _Ty>
  concept regular = semiregular<_Ty> && equality_comparable<_Ty>;

  // Not exactly right, but close.
  template <class _Ty>
  concept __boolean_testable_ = convertible_to<_Ty, bool>;

  template <class T, class U>
  concept __partially_ordered_with = requires(__cref_t<T> t, __cref_t<U> u) {
    { t < u } -> __boolean_testable_;
    { t > u } -> __boolean_testable_;
    { t <= u } -> __boolean_testable_;
    { t >= u } -> __boolean_testable_;
    { u < t } -> __boolean_testable_;
    { u > t } -> __boolean_testable_;
    { u <= t } -> __boolean_testable_;
    { u >= t } -> __boolean_testable_;
  };

  template <class _Ty>
  concept totally_ordered = equality_comparable<_Ty> && __partially_ordered_with<_Ty, _Ty>;

  template <class _Ty>
  concept __movable_value = move_constructible<__decay_t<_Ty>>
                         && constructible_from<__decay_t<_Ty>, _Ty>;

  template <class _Ty>
  concept __nothrow_movable_value = __movable_value<_Ty> && requires(_Ty&& __t) {
    { __decay_t<_Ty>{__decay_t<_Ty>{static_cast<_Ty &&>(__t)}} } noexcept;
  };

  template <class _Ty, class... _As>
  concept __nothrow_constructible_from = constructible_from<_Ty, _As...>
                                      && STDEXEC_IS_NOTHROW_CONSTRUCTIBLE(_Ty, _As...);

  template <class _Ty>
  concept __nothrow_move_constructible = __nothrow_constructible_from<_Ty, _Ty>;

  template <class _Ty>
  concept __nothrow_copy_constructible = __nothrow_constructible_from<_Ty, const _Ty&>;

  template <class... _Ts>
  concept __decay_copyable = (constructible_from<__decay_t<_Ts>, _Ts> && ...);

  template <class... _Ts>
  using __decay_copyable_t = __mbool<__decay_copyable<_Ts...>>;

  template <class... _Ts>
  concept __nothrow_decay_copyable = (__nothrow_constructible_from<__decay_t<_Ts>, _Ts> && ...);

  template <class... _Ts>
  using __nothrow_decay_copyable_t = __mbool<__nothrow_decay_copyable<_Ts...>>;

  template <class _Ty, class _Up>
  concept __decays_to_derived_from = derived_from<__decay_t<_Ty>, _Up>;
} // namespace stdexec
