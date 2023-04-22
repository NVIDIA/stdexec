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
#include "__type_traits.hpp"

namespace stdexec {

  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <class _Fun, class... _As>
  concept __callable =                      //
    requires(_Fun&& __fun, _As&&... __as) { //
      ((_Fun&&) __fun)((_As&&) __as...);    //
    };
  template <class _Fun, class... _As>
  concept __nothrow_callable =  //
    __callable<_Fun, _As...> && //
    requires(_Fun&& __fun, _As&&... __as) {
      { ((_Fun&&) __fun)((_As&&) __as...) } noexcept;
    };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  template <class...>
  struct __types;

  template <class... _Ts>
  concept __typename = requires { typename __types<_Ts...>; };

  //////////////////////////////////////////////////////////////////////////////////////////////////
#if STDEXEC_CLANG()

  template <class _Ap, class _Bp>
  concept __same_as = __is_same(_Ap, _Bp);

#elif STDEXEC_GCC()

  template <class _Ap, class _Bp>
  concept __same_as = __is_same_as(_Ap, _Bp);

#else

  template <class _Ap, class _Bp>
  inline constexpr bool __same_as_v = false;

  template <class _Ap>
  inline constexpr bool __same_as_v<_Ap, _Ap> = true;

  template <class _Ap, class _Bp>
  concept __same_as = __same_as_v<_Ap, _Bp>;

#endif

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
  concept __none_of = ((!__same_as<_Ty, _Us>) &&...);

  template <class, template <class...> class>
  constexpr bool __is_instance_of_ = false;
  template <class... _As, template <class...> class _Ty>
  constexpr bool __is_instance_of_<_Ty<_As...>, _Ty> = true;

  template <class _Ay, template <class...> class _Ty>
  concept __is_instance_of = __is_instance_of_<_Ay, _Ty>;

  template <class _Ay, template <class...> class _Ty>
  concept __is_not_instance_of = !__is_instance_of<_Ay, _Ty>;

}
