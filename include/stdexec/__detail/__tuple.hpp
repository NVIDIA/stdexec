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

#include "__config.hpp"
#include "__concepts.hpp"
#include "__type_traits.hpp"
#include "__meta.hpp"

#include <cstddef>

#if STDEXEC_GCC() || STDEXEC_NVHPC()
// GCC (as of v14) does not implement the resolution of CWG1835
// https://cplusplus.github.io/CWG/issues/1835.html
// See: https://godbolt.org/z/TzxrhK6ea
#  define STDEXEC_NO_CWG1835
#endif

#ifdef STDEXEC_NO_CWG1835
#  define STDEXEC_CWG1835_TEMPLATE
#else
#  define STDEXEC_CWG1835_TEMPLATE template
#endif

namespace stdexec {
  namespace __tup {
    template <class _Ty, std::size_t _Idx>
    struct __box {
      // See https://github.com/llvm/llvm-project/issues/93563
      //STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Ty __value;
    };

    template <class _Ty>
    concept __empty = //
      STDEXEC_IS_EMPTY(_Ty) && STDEXEC_IS_TRIVIALLY_CONSTRUCTIBLE(_Ty);

    template <__empty _Ty>
    inline _Ty __value{};

    // A specialization for empty types so that they don't take up space.
    template <__empty _Ty, std::size_t _Idx>
    struct __box<_Ty, _Idx> {
      __box() = default;

      constexpr __box(__not_decays_to<__box> auto &&) noexcept {
      }

      static constexpr _Ty &__value = __tup::__value<_Ty>;
    };

    template <auto _Idx, class... _Ts>
    struct __tuple;

    template <std::size_t... _Is, __indices<_Is...> _Idx, class... _Ts>
    struct __tuple<_Idx, _Ts...> : __box<_Ts, _Is>... {
      template <class... _Us>
      static __tuple __convert_from(__tuple<_Idx, _Us...> &&__tup) {
        return __tuple{
          {static_cast<_Us &&>(__tup.STDEXEC_CWG1835_TEMPLATE __box<_Us, _Is>::__value)}...};
      }

      template <class... _Us>
      static __tuple __convert_from(__tuple<_Idx, _Us...> const &__tup) {
        return __tuple{{__tup.STDEXEC_CWG1835_TEMPLATE __box<_Us, _Is>::__value}...};
      }

      template <class _Fn, class _Self, class... _Us>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) static auto apply(_Fn &&__fn, _Self &&__self, _Us &&...__us) //
        noexcept(noexcept(static_cast<_Fn &&>(__fn)(
          static_cast<_Us &&>(__us)...,
          static_cast<_Self &&>(__self).STDEXEC_CWG1835_TEMPLATE __box<_Ts, _Is>::__value...)))
          -> decltype(static_cast<_Fn &&>(__fn)(
            static_cast<_Us &&>(__us)...,
            static_cast<_Self &&>(__self).STDEXEC_CWG1835_TEMPLATE __box<_Ts, _Is>::__value...)) {
        return static_cast<_Fn &&>(__fn)(
          static_cast<_Us &&>(__us)...,
          static_cast<_Self &&>(__self).STDEXEC_CWG1835_TEMPLATE __box<_Ts, _Is>::__value...);
      }

      template <class _Fn, class _Self, class... _Us>
        requires(__callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>> && ...)
      STDEXEC_ATTRIBUTE((host, device, always_inline)) static auto for_each(_Fn &&__fn, _Self &&__self, _Us &&...__us) //
        noexcept((__nothrow_callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>> && ...)) -> void {
        return (
          static_cast<_Fn &&>(__fn)(
            static_cast<_Us &&>(__us)...,
            static_cast<_Self &&>(__self).STDEXEC_CWG1835_TEMPLATE __box<_Ts, _Is>::__value),
          ...);
      }
    };

    template <class... _Ts>
    STDEXEC_ATTRIBUTE((host, device)) __tuple(_Ts...) -> __tuple<__indices_for<_Ts...>{}, _Ts...>;

    template <class _Fn, class _Tuple, class... _Us>
    using __apply_result_t = //
      decltype(__declval<_Tuple>()
                 .apply(__declval<_Fn>(), __declval<_Tuple>(), __declval<_Us>()...));

    template <class _Fn, class _Tuple, class... _Us>
    concept __applicable = requires { typename __apply_result_t<_Fn, _Tuple, _Us...>; };

    template <class _Fn, class _Tuple, class... _Us>
    concept __nothrow_applicable =
      __applicable<_Fn, _Tuple, _Us...>
      && noexcept(
        __declval<_Tuple>().apply(__declval<_Fn>(), __declval<_Tuple>(), __declval<_Us>()...));

#if STDEXEC_GCC()
    template <class... _Ts>
    struct __mk_tuple {
      using __t = __tuple<__indices_for<_Ts...>{}, _Ts...>;
    };
#endif

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((host, device, always_inline)) constexpr _Ty &&get(__box<_Ty, _Idx> &&__self) noexcept {
      return static_cast<_Ty &&>(__self.__value);
    }

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((host, device, always_inline)) constexpr _Ty &get(__box<_Ty, _Idx> &__self) noexcept {
      return __self.__value;
    }

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((host, device, always_inline)) constexpr const _Ty &get(const __box<_Ty, _Idx> &__self) noexcept {
      return __self.__value;
    }

    template <class _Fn, class _Tuple>
    STDEXEC_ATTRIBUTE((host, device, always_inline)) auto operator<<(_Tuple &&__tup, _Fn __fn) noexcept(__nothrow_move_constructible<_Fn>) {
      return
        [&__tup, __fn]<class... _Us>(_Us &&...__us) //
        noexcept(__nothrow_applicable<_Fn, _Tuple, _Us...>)
          -> __apply_result_t<_Fn, _Tuple, _Us...> {
          return __tup.apply(__fn, static_cast<_Tuple &&>(__tup), static_cast<_Us &&>(__us)...);
        };
    }

    template <class _Fn, class... _Tuples>
    auto __cat_apply(_Fn __fn, _Tuples &&...__tups)                          //
      noexcept(noexcept((static_cast<_Tuples &&>(__tups) << ... << __fn)())) //
      -> decltype((static_cast<_Tuples &&>(__tups) << ... << __fn)()) {
      return (static_cast<_Tuples &&>(__tups) << ... << __fn)();
    }

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

    inline constexpr struct __mktuple_t {
      template <class... _Ts>
      STDEXEC_ATTRIBUTE((host, device, always_inline)) auto
        operator()(_Ts &&...__ts) const noexcept(noexcept(__tuple{static_cast<_Ts &&>(__ts)...}))
          -> decltype(__tuple{static_cast<_Ts &&>(__ts)...}) {
        return __tuple{static_cast<_Ts &&>(__ts)...};
      }
    } __mktuple{};

    STDEXEC_PRAGMA_POP()

  } // namespace __tup

  using __tup::__tuple;

#if STDEXEC_GCC()
  template <class... _Ts>
  using __tuple_for = __t<__tup::__mk_tuple<_Ts...>>;
#else
  template <class... _Ts>
  using __tuple_for = __tuple<__indices_for<_Ts...>{}, _Ts...>;
#endif

  template <class... _Ts>
  using __decayed_tuple = __tuple_for<__decay_t<_Ts>...>;

  // So we can use __tuple as a typelist
  template <auto _Idx, class... _Ts>
  struct __muncurry_<__tuple<_Idx, _Ts...>> {
    template <class _Fn>
    using __f = __minvoke<_Fn, _Ts...>;
  };
} // namespace stdexec
