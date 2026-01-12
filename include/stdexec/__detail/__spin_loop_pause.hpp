/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../stdexec/__detail/__config.hpp"

// The below code for spin_loop_pause is taken from https://github.com/max0x7ba/atomic_queue/blob/master/include/atomic_queue/defs.h
// Copyright (c) 2019 Maxim Egorushkin. MIT License.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  if STDEXEC_MSVC_HEADERS()
#    include <intrin.h>
#  endif
namespace STDEXEC {
  STDEXEC_ATTRIBUTE(always_inline) static void __spin_loop_pause() noexcept {
#  if STDEXEC_MSVC_HEADERS()
    _mm_pause();
#  else
    __builtin_ia32_pause();
#  endif
  }
} // namespace STDEXEC
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
namespace STDEXEC {
  STDEXEC_ATTRIBUTE(always_inline) static void __spin_loop_pause() noexcept {
#  if (                                                                                            \
    defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)              \
    || defined(__ARM_ARCH_6T2__) || defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__)            \
    || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)            \
    || defined(__ARM_ARCH_8A__) || defined(__aarch64__))
    asm volatile("yield" ::: "memory");
#  elif defined(_M_ARM64)
    __yield();
#  else
    asm volatile("nop" ::: "memory");
#  endif
  }
} // namespace STDEXEC
#else
namespace STDEXEC {
  STDEXEC_ATTRIBUTE(always_inline) static void __spin_loop_pause() noexcept {
  }
} // namespace STDEXEC
#endif