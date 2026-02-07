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
#include "__utility.hpp"

#include <type_traits>

namespace STDEXEC {
  template <class _Fn, class... _Ts>
  struct __scope_guard;

  template <class _Fn>
  struct __scope_guard<_Fn> {
    STDEXEC_ATTRIBUTE(no_unique_address) _Fn __fn_;
    STDEXEC_ATTRIBUTE(no_unique_address) __immovable __hidden_ { };
    bool __dismissed_{false};

    constexpr ~__scope_guard() {
      if (!__dismissed_)
        static_cast<_Fn&&>(__fn_)();
    }

    constexpr void __dismiss() noexcept {
      __dismissed_ = true;
    }
  };

  template <class _Fn, class _T0>
  struct __scope_guard<_Fn, _T0> {
    STDEXEC_ATTRIBUTE(no_unique_address) _Fn __fn_;
    STDEXEC_ATTRIBUTE(no_unique_address) _T0 __t0_;
    STDEXEC_ATTRIBUTE(no_unique_address) __immovable __hidden_ { };

    bool __dismissed_{false};

    constexpr void __dismiss() noexcept {
      __dismissed_ = true;
    }

    constexpr ~__scope_guard() {
      if (!__dismissed_)
        static_cast<_Fn&&>(__fn_)(static_cast<_T0&&>(__t0_));
    }
  };

  template <class _Fn, class _T0, class _T1>
  struct __scope_guard<_Fn, _T0, _T1> {
    STDEXEC_ATTRIBUTE(no_unique_address) _Fn __fn_;
    STDEXEC_ATTRIBUTE(no_unique_address) _T0 __t0_;
    STDEXEC_ATTRIBUTE(no_unique_address) _T1 __t1_;
    STDEXEC_ATTRIBUTE(no_unique_address) __immovable __hidden_ { };

    bool __dismissed_{false};

    constexpr void __dismiss() noexcept {
      __dismissed_ = true;
    }

    constexpr ~__scope_guard() {
      if (!__dismissed_)
        static_cast<_Fn&&>(__fn_)(static_cast<_T0&&>(__t0_), static_cast<_T1&&>(__t1_));
    }
  };

  template <class _Fn, class _T0, class _T1, class _T2>
  struct __scope_guard<_Fn, _T0, _T1, _T2> {
    STDEXEC_ATTRIBUTE(no_unique_address) _Fn __fn_;
    STDEXEC_ATTRIBUTE(no_unique_address) _T0 __t0_;
    STDEXEC_ATTRIBUTE(no_unique_address) _T1 __t1_;
    STDEXEC_ATTRIBUTE(no_unique_address) _T2 __t2_;
    STDEXEC_ATTRIBUTE(no_unique_address) __immovable __hidden_ { };

    bool __dismissed_{false};

    constexpr void __dismiss() noexcept {
      __dismissed_ = true;
    }

    constexpr ~__scope_guard() {
      if (!__dismissed_)
        static_cast<_Fn&&>(
          __fn_)(static_cast<_T0&&>(__t0_), static_cast<_T1&&>(__t1_), static_cast<_T2&&>(__t2_));
    }
  };

  template <class _Fn, class... _Ts>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    __scope_guard(_Fn, _Ts...) -> __scope_guard<_Fn, std::unwrap_reference_t<_Ts>...>;
} // namespace STDEXEC
