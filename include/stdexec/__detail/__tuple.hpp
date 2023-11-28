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
#include "__meta.hpp"

namespace stdexec {
  namespace __tup {
    template <class _Ty, std::size_t _Idx>
    struct __box {
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS _Ty __value;
    };

    template <class _Idx, class... _Ts>
    struct __tuple;

    template <std::size_t... _Idx, class... _Ts>
    struct __tuple<__indices<_Idx...>, _Ts...> : __box<_Ts, _Idx>... { };

    template <class... _Ts>
    STDEXEC_ATTRIBUTE((host, device))
    __tuple(_Ts...) -> __tuple<__indices_for<_Ts...>, _Ts...>;

#if STDEXEC_GCC()
    template <class... _Ts>
    struct __mk_tuple {
      using __t = __tuple<__indices_for<_Ts...>, _Ts...>;
    };
    template <class... _Ts>
    using __tuple_for = __t<__mk_tuple<_Ts...>>;
#else
    template <class... _Ts>
    using __tuple_for = __tuple<__indices_for<_Ts...>, _Ts...>;
#endif

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr _Ty&& __get(__box<_Ty, _Idx>&& __self) noexcept {
      return (_Ty&&) __self.__value;
    }

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr _Ty& __get(__box<_Ty, _Idx>& __self) noexcept {
      return __self.__value;
    }

    template <std::size_t _Idx, class _Ty>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr const _Ty& __get(const __box<_Ty, _Idx>& __self) noexcept {
      return __self.__value;
    }

    template <std::size_t... _Idx, class... _Ts>
    void __tuple_like_(const __tuple<__indices<_Idx...>, _Ts...>&);

    template <class _Tup>
    concept __tuple_like = requires(_Tup& __tup) { __tup::__tuple_like_(__tup); };

    struct __apply_ {
      template <class _Fun, class _Tuple, std::size_t... _Idx, class... _Ts>
        requires __callable<_Fun, __copy_cvref_t<_Tuple, _Ts>...>
      constexpr auto
        operator()(_Fun&& __fun, _Tuple&& __tup, const __tuple<__indices<_Idx...>, _Ts...>*) noexcept(
          __nothrow_callable<_Fun, __copy_cvref_t<_Tuple, _Ts>...>)
          -> __call_result_t<_Fun, __copy_cvref_t<_Tuple, _Ts>...> {
        return ((_Fun&&) __fun)(
          static_cast<__copy_cvref_t<_Tuple, __box<_Ts, _Idx>>&&>(__tup).__value...);
      }
    };

    template <class _Fun, __tuple_like _Tuple>
    STDEXEC_ATTRIBUTE((always_inline))
    constexpr auto __apply(_Fun&& __fun, _Tuple&& __tup) noexcept(
      noexcept(__apply_()((_Fun&&) __fun, (_Tuple&&) __tup, &__tup)))
      -> decltype(__apply_()((_Fun&&) __fun, (_Tuple&&) __tup, &__tup)) {
      return __apply_()((_Fun&&) __fun, (_Tuple&&) __tup, &__tup);
    }
  }

  using __tup::__tuple;

  // So we can use __tuple as a typelist and ignore the first template parameter
  template <class _Fn, class _Idx, class... _Ts>
    requires __minvocable<_Fn, _Ts...>
  struct __uncurry_<_Fn, __tuple<_Idx, _Ts...>> {
    using __t = __minvoke<_Fn, _Ts...>;
  };
}
