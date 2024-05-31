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

    // A specialization for empty types so that they don't take up space.
    template <__empty _Ty, std::size_t _Idx>
    struct __box<_Ty, _Idx> {
      __box() = default;

      constexpr __box(__not_decays_to<__box> auto &&) noexcept {
      }

      static _Ty __value;
    };

    template <__empty _Ty, std::size_t _Idx>
    inline _Ty __box<_Ty, _Idx>::__value{};

    template <auto _Idx, class... _Ts>
    struct __tuple;

    template <std::size_t... _Is, __indices<_Is...> _Idx, class... _Ts>
    struct __tuple<_Idx, _Ts...> : __box<_Ts, _Is>... {

      template <class _Fn, class _Self, class... _Us>
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      static auto
        apply(_Fn &&__fn, _Self &&__self, _Us &&...__us) //
        noexcept(noexcept(static_cast<_Fn &&>(__fn)(
          static_cast<_Us &&>(__us)...,
          static_cast<_Self &&>(__self).__box<_Ts, _Is>::__value...)))
          -> decltype(static_cast<_Fn &&>(__fn)(
            static_cast<_Us &&>(__us)...,
            static_cast<_Self &&>(__self).__box<_Ts, _Is>::__value...)) {
        return static_cast<_Fn &&>(__fn)(
          static_cast<_Us &&>(__us)..., static_cast<_Self &&>(__self).__box<_Ts, _Is>::__value...);
      }

      template <class _Fn, class _Self, class... _Us>
        requires(__callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>> && ...)
      STDEXEC_ATTRIBUTE((host, device, always_inline))
      static auto
        for_each(_Fn &&__fn, _Self &&__self, _Us &&...__us) //
        noexcept((__nothrow_callable<_Fn, _Us..., __copy_cvref_t<_Self, _Ts>> && ...)) -> void {
        return (
          static_cast<_Fn &&>(__fn)(
            static_cast<_Us &&>(__us)..., static_cast<_Self &&>(__self).__box<_Ts, _Is>::__value),
          ...);
      }
    };

    template <class... _Ts>
    STDEXEC_ATTRIBUTE((host, device))
    __tuple(_Ts...) -> __tuple<__indices_for<_Ts...>{}, _Ts...>;

#if STDEXEC_GCC()
    template <class... _Ts>
    struct __mk_tuple {
      using __t = __tuple<__indices_for<_Ts...>{}, _Ts...>;
    };
    template <class... _Ts>
    using __tuple_for = __t<__mk_tuple<_Ts...>>;
#else
    template <class... _Ts>
    using __tuple_for = __tuple<__indices_for<_Ts...>{}, _Ts...>;
#endif

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr _Ty &&
      __get(__box<_Ty, _Idx> &&__self) noexcept {
      return static_cast<_Ty &&>(__self.__value);
    }

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr _Ty &
      __get(__box<_Ty, _Idx> &__self) noexcept {
      return __self.__value;
    }

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr const _Ty &
      __get(const __box<_Ty, _Idx> &__self) noexcept {
      return __self.__value;
    }

    template <auto _Idx, class... _Ts>
    void __tuple_like_(const __tuple<_Idx, _Ts...> &);

    template <class _Tup>
    concept __tuple_like = requires(_Tup &__tup) { __tup::__tuple_like_(__tup); };

    struct __apply_t {
      template <class _Fun, __tuple_like _Tuple>
      STDEXEC_ATTRIBUTE((always_inline))
      constexpr auto
        operator()(_Fun &&__fun, _Tuple &&__tup) const //
        noexcept(noexcept(__tup.apply(static_cast<_Fun &&>(__fun), static_cast<_Tuple &&>(__tup))))
          -> decltype(__tup.apply(static_cast<_Fun &&>(__fun), static_cast<_Tuple &&>(__tup))) {
        return __tup.apply(static_cast<_Fun &&>(__fun), static_cast<_Tuple &&>(__tup));
      }
    };

    inline constexpr __apply_t __apply{};
  } // namespace __tup

  using __tup::__tuple;

  // So we can use __tuple as a typelist and ignore the first template parameter
  template <class _Fn, auto _Idx, class... _Ts>
    requires __minvocable<_Fn, _Ts...>
  struct __uncurry_<_Fn, __tuple<_Idx, _Ts...>> {
    using __t = __minvoke<_Fn, _Ts...>;
  };
} // namespace stdexec
