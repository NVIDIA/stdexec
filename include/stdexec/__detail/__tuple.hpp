/*
 * Copyright (c) 2025 NVIDIA Corporation
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

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace stdexec {
  namespace __tup {
    template <class... _Ts>
    struct STDEXEC_ATTRIBUTE(empty_bases) __tuple;

    template <class... _Ts>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    auto __tuple_base_fn(__tuple<_Ts...>&&) noexcept -> __tuple<_Ts...>;

    template <class... _Ts>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    auto __tuple_base_fn(__tuple<_Ts...>&) noexcept -> __tuple<_Ts...>&;

    template <class... _Ts>
    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    auto __tuple_base_fn(const __tuple<_Ts...>& __tup) noexcept -> const __tuple<_Ts...>&;

    template <class _Tuple>
    using __tuple_base_t = decltype(__tup::__tuple_base_fn(__declval<_Tuple>()));

    template <class _Fn, class _Tuple, class... _Us>
    extern constexpr bool __applicable_v = __applicable_v<_Fn, __tuple_base_t<_Tuple>, _Us...>;

    template <class _Fn, class... _Ts, class... _Us>
    inline constexpr bool __applicable_v<_Fn, __tuple<_Ts...>, _Us...> =
      __callable<_Fn, _Us..., _Ts...>;

    template <class _Fn, class... _Ts, class... _Us>
    inline constexpr bool __applicable_v<_Fn, __tuple<_Ts...>&, _Us...> =
      __callable<_Fn, _Us..., _Ts&...>;

    template <class _Fn, class... _Ts, class... _Us>
    inline constexpr bool __applicable_v<_Fn, const __tuple<_Ts...>&, _Us...> =
      __callable<_Fn, _Us..., const _Ts&...>;

    template <class _Fn, class _Tuple, class... _Us>
    extern constexpr bool __nothrow_applicable_v =
      __nothrow_applicable_v<_Fn, __tuple_base_t<_Tuple>, _Us...>;

    template <class _Fn, class... _Ts, class... _Us>
    inline constexpr bool __nothrow_applicable_v<_Fn, __tuple<_Ts...>, _Us...> =
      __nothrow_callable<_Fn, _Us..., _Ts...>;

    template <class _Fn, class... _Ts, class... _Us>
    inline constexpr bool __nothrow_applicable_v<_Fn, __tuple<_Ts...>&, _Us...> =
      __nothrow_callable<_Fn, _Us..., _Ts&...>;

    template <class _Fn, class... _Ts, class... _Us>
    inline constexpr bool __nothrow_applicable_v<_Fn, const __tuple<_Ts...>&, _Us...> =
      __nothrow_callable<_Fn, _Us..., const _Ts&...>;

    template <class _Ty, std::size_t _Idx>
    struct __box {
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Ty __value;
    };

    template <class _Ty>
    concept __empty = STDEXEC_IS_EMPTY(_Ty) && STDEXEC_IS_TRIVIALLY_CONSTRUCTIBLE(_Ty)
                   && STDEXEC_IS_TRIVIALLY_COPYABLE(_Ty);

    template <__empty _Ty>
    inline _Ty __value{};

    // A specialization for empty types so that they don't take up space.
    template <__empty _Ty, std::size_t _Idx>
    struct __box<_Ty, _Idx> {
      __box() = default;

      constexpr __box(__not_decays_to<__box> auto&&) noexcept {
      }

      static constexpr _Ty& __value = __tup::__value<_Ty>;
    };

    template <class _Index, class... _Ts>
    struct __tupl_base;

    template <std::size_t... _Index, class... _Ts>
    struct STDEXEC_ATTRIBUTE(empty_bases)
      __tupl_base<__indices<_Index...>, _Ts...> : __box<_Ts, _Index>... {
      static constexpr size_t __size = sizeof...(_Ts);

      STDEXEC_EXEC_CHECK_DISABLE
      template <class _Fn, class _Self, class... _Us>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      static constexpr auto __apply(_Fn&& __fn, _Self&& __self, _Us&&... __us)
        noexcept(__nothrow_applicable_v<_Fn, _Self, _Us...>) -> decltype(auto) {
        return static_cast<_Fn&&>(__fn)(
          static_cast<_Us&&>(__us)...,
          static_cast<_Self&&>(__self).STDEXEC_CWG1835_TEMPLATE __box<_Ts, _Index>::__value...);
      }
    };

    template <class... _Ts>
    struct __tuple : __tup::__tupl_base<__make_indices<sizeof...(_Ts)>, _Ts...> { };

    template <>
    struct __tuple<> {
      static constexpr size_t __size = 0;
    };

    template <class _Tp0>
    struct __tuple<_Tp0> {
      static constexpr size_t __size = 1;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
    };

    template <class _Tp0, class _Tp1>
    struct __tuple<_Tp0, _Tp1> {
      static constexpr size_t __size = 2;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
    };

    template <class _Tp0, class _Tp1, class _Tp2>
    struct __tuple<_Tp0, _Tp1, _Tp2> {
      static constexpr size_t __size = 3;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp2 __val2;
    };

    template <class _Tp0, class _Tp1, class _Tp2, class _Tp3>
    struct __tuple<_Tp0, _Tp1, _Tp2, _Tp3> {
      static constexpr size_t __size = 4;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp2 __val2;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp3 __val3;
    };

    template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4>
    struct __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4> {
      static constexpr size_t __size = 5;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp2 __val2;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp3 __val3;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp4 __val4;
    };

    template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5>
    struct __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5> {
      static constexpr size_t __size = 6;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp2 __val2;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp3 __val3;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp4 __val4;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp5 __val5;
    };

    template <class _Tp0, class _Tp1, class _Tp2, class _Tp3, class _Tp4, class _Tp5, class _Tp6>
    struct __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5, _Tp6> {
      static constexpr size_t __size = 7;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp2 __val2;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp3 __val3;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp4 __val4;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp5 __val5;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp6 __val6;
    };

    template <
      class _Tp0,
      class _Tp1,
      class _Tp2,
      class _Tp3,
      class _Tp4,
      class _Tp5,
      class _Tp6,
      class _Tp7
    >
    struct __tuple<_Tp0, _Tp1, _Tp2, _Tp3, _Tp4, _Tp5, _Tp6, _Tp7> {
      static constexpr size_t __size = 8;

      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp0 __val0;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp1 __val1;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp2 __val2;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp3 __val3;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp4 __val4;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp5 __val5;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp6 __val6;
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Tp7 __val7;
    };

    template <class... _Ts>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __tuple(_Ts...) -> __tuple<_Ts...>;

#define STDEXEC_TUPLE_GET(_Idx) , static_cast<_Tuple&&>(__tupl).__val##_Idx

    //
    // __apply(fn, tuple, extra...)
    //
    struct __apply_t {
      STDEXEC_EXEC_CHECK_DISABLE
      template <class _Fn, class _Tuple, class... _Us>
        requires __tup::__applicable_v<_Fn, _Tuple, _Us...>
      STDEXEC_ATTRIBUTE(always_inline, host, device)
      constexpr auto operator()(_Fn&& __fn, _Tuple&& __tupl, _Us&&... __us) const
        noexcept(__tup::__nothrow_applicable_v<_Fn, _Tuple, _Us...>) -> decltype(auto) {
        constexpr size_t __size = STDEXEC_REMOVE_REFERENCE(_Tuple)::__size;

        if constexpr (__size == 0) {
          return static_cast<_Fn&&>(__fn)(static_cast<_Us&&>(__us)...);
        } else if constexpr (__size == 1) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(1, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 2) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(2, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 3) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(3, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 4) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(4, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 5) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(5, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 6) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(6, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 7) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(7, STDEXEC_TUPLE_GET));
        } else if constexpr (__size == 8) {
          return static_cast<_Fn&&>(__fn)(
            static_cast<_Us&&>(__us)... STDEXEC_REPEAT(8, STDEXEC_TUPLE_GET));
        } else {
          return __tupl.__apply(
            static_cast<_Fn&&>(__fn), static_cast<_Tuple&&>(__tupl), static_cast<_Us&&>(__us)...);
        }
      }
    };
#undef STDEXEC_TUPLE_GET
  } // namespace __tup

  using __tup::__tuple;

  inline constexpr __tup::__apply_t __apply{};

  template <class _Fn, class _Tuple, class... _Us>
  using __apply_result_t = __result_of<__apply, _Fn, _Tuple, _Us...>;

  template <class _Fn, class _Tuple, class... _Us>
  concept __applicable = __tup::__applicable_v<_Fn, _Tuple, _Us...>;

  template <class _Fn, class _Tuple, class... _Us>
  concept __nothrow_applicable = __tup::__nothrow_applicable_v<_Fn, _Tuple, _Us...>;

  //
  // __get<I>(tupl)
  //
  namespace __tup {
    template <size_t _Index, class _Value>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto __get(__box<_Value, _Index>&& __self) noexcept -> _Value&& {
      return static_cast<_Value&&>(__self.__value);
    }

    template <size_t _Index, class _Value>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto __get(__box<_Value, _Index>& __self) noexcept -> _Value& {
      return __self.__value;
    }

    template <size_t _Index, class _Value>
    STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
    constexpr auto __get(const __box<_Value, _Index>& __self) noexcept -> const _Value& {
      return __self.__value;
    }
  } // namespace __tup

  template <size_t _Index, class _Tuple>
  STDEXEC_ATTRIBUTE(nodiscard, always_inline, host, device)
  constexpr auto __get(_Tuple&& __tupl) noexcept -> auto&& {
    constexpr auto __size = STDEXEC_REMOVE_REFERENCE(_Tuple)::__size;
    static_assert(_Index < __size, "Index out of bounds in __get");

    if constexpr (_Index == 0) {
      return static_cast<_Tuple&&>(__tupl).__val0;
    } else if constexpr (_Index == 1) {
      return static_cast<_Tuple&&>(__tupl).__val1;
    } else if constexpr (_Index == 2) {
      return static_cast<_Tuple&&>(__tupl).__val2;
    } else if constexpr (_Index == 3) {
      return static_cast<_Tuple&&>(__tupl).__val3;
    } else if constexpr (_Index == 4) {
      return static_cast<_Tuple&&>(__tupl).__val4;
    } else if constexpr (_Index == 5) {
      return static_cast<_Tuple&&>(__tupl).__val5;
    } else if constexpr (_Index == 6) {
      return static_cast<_Tuple&&>(__tupl).__val6;
    } else if constexpr (_Index == 7) {
      return static_cast<_Tuple&&>(__tupl).__val7;
    } else if constexpr (_Index == 8) {
      return static_cast<_Tuple&&>(__tupl).__val8;
    } else {
      return __tup::__get<_Index>(static_cast<_Tuple&&>(__tupl));
    }
  }

  //
  // __decayed_tuple<Ts...>
  //
  template <class... _Ts>
  using __decayed_tuple = __tuple<__decay_t<_Ts>...>;

  //
  // __tuple_size_v
  //
  template <class _Tuple>
  inline constexpr size_t __tuple_size_v = __tuple_size_v<__tup::__tuple_base_t<_Tuple>>;

  template <class... _Ts>
  inline constexpr size_t __tuple_size_v<__tuple<_Ts...>> = sizeof...(_Ts);

  template <class... _Ts>
  inline constexpr size_t __tuple_size_v<const __tuple<_Ts...>> = sizeof...(_Ts);

  template <class... _Ts>
  inline constexpr size_t __tuple_size_v<__tuple<_Ts...>&> = sizeof...(_Ts);

  template <class... _Ts>
  inline constexpr size_t __tuple_size_v<const __tuple<_Ts...>&> = sizeof...(_Ts);

  //
  // __tuple_element_t
  //
  template <size_t _Index, class _Tuple>
  using __tuple_element_t = decltype(__tt::__remove_rvalue_reference_fn(
    stdexec::__get<_Index>(__declval<_Tuple>())));

  //
  // __cat_apply(fn, tups...)
  //
  namespace __tup {
    template <class _Fn, class _Tuple>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    constexpr auto
      operator<<(_Tuple&& __tup, _Fn __fn) noexcept(__nothrow_move_constructible<_Fn>) {
      return [&__tup, __fn = static_cast<_Fn&&>(__fn)]<class... _Us>(_Us&&... __us) noexcept(
               __nothrow_applicable<_Fn, _Tuple, _Us...>) -> __apply_result_t<_Fn, _Tuple, _Us...> {
        return stdexec::__apply(__fn, static_cast<_Tuple&&>(__tup), static_cast<_Us&&>(__us)...);
      };
    }

    struct __cat_apply_t {
      template <class _Fn, class... _Tuples>
      STDEXEC_ATTRIBUTE(host, device)
      constexpr auto operator()(_Fn __fn, _Tuples&&... __tups) const
        STDEXEC_AUTO_RETURN((static_cast<_Tuples&&>(__tups) << ... << __fn)())
    };
  } // namespace __tup

  inline constexpr __tup::__cat_apply_t __cat_apply{};

  //
  // __mktuple(ts...)
  //
  struct __mktuple_t {
    template <class... _Ts>
    STDEXEC_ATTRIBUTE(host, device, always_inline)
    auto operator()(_Ts&&... __ts) const STDEXEC_AUTO_RETURN(__tuple{static_cast<_Ts&&>(__ts)...})
  };

  inline constexpr __mktuple_t __mktuple{};
} // namespace stdexec

STDEXEC_PRAGMA_POP()
