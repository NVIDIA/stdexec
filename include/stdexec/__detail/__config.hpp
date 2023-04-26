/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#if __cplusplus < 202002L
#error This library requires the use of C++20.
#endif

#include <cassert>

#define STDEXEC_CAT_(_XP, ...) _XP##__VA_ARGS__
#define STDEXEC_CAT(_XP, ...) STDEXEC_CAT_(_XP, __VA_ARGS__)

#define STDEXEC_EXPAND(...) __VA_ARGS__
#define STDEXEC_EVAL(_MACRO, ...) _MACRO(__VA_ARGS__)

#define STDEXEC_NOT(_XP) STDEXEC_CAT(STDEXEC_NOT_, _XP)
#define STDEXEC_NOT_0 1
#define STDEXEC_NOT_1 0

#define STDEXEC_IIF_0(_YP, ...) __VA_ARGS__
#define STDEXEC_IIF_1(_YP, ...) _YP
#define STDEXEC_IIF(_XP, _YP, ...) STDEXEC_EVAL(STDEXEC_CAT(STDEXEC_IIF_, _XP), _YP, __VA_ARGS__)

#define STDEXEC_COUNT(...) \
  STDEXEC_EXPAND(STDEXEC_COUNT_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define STDEXEC_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _NP, ...) _NP

#define STDEXEC_CHECK(...) STDEXEC_EXPAND(STDEXEC_CHECK_N(__VA_ARGS__, 0, ))
#define STDEXEC_CHECK_N(_XP, _NP, ...) _NP
#define STDEXEC_PROBE(_XP) _XP, 1,

#if defined(__NVCOMPILER)
#define STDEXEC_NVHPC() 1
#elif defined(__clang__)
#define STDEXEC_CLANG() 1
#elif defined(__GNUC__)
#define STDEXEC_GCC() 1
#elif defined(_MSC_VER)
#define STDEXEC_MSVC() 1
#endif

#ifndef STDEXEC_NVHPC
#define STDEXEC_NVHPC() 0
#endif
#ifndef STDEXEC_CLANG
#define STDEXEC_CLANG() 0
#endif
#ifndef STDEXEC_GCC
#define STDEXEC_GCC() 0
#endif
#ifndef STDEXEC_MSVC
#define STDEXEC_MSVC() 0
#endif

#if STDEXEC_CLANG()
#define STDEXEC_STRINGIZE(_ARG) #_ARG
#define STDEXEC_PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define STDEXEC_PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define STDEXEC_PRAGMA_IGNORE(_ARG) _Pragma(STDEXEC_STRINGIZE(GCC diagnostic ignored _ARG))
#else
#define STDEXEC_PRAGMA_PUSH()
#define STDEXEC_PRAGMA_POP()
#define STDEXEC_PRAGMA_IGNORE(_ARG)
#endif

#ifdef __has_builtin
#define STDEXEC_HAS_BUILTIN __has_builtin
#else
#define STDEXEC_HAS_BUILTIN(...) 0
#endif

// Before gcc-12, gcc really didn't like tuples or variants of immovable types
#if STDEXEC_GCC() && (__GNUC__ < 12)
#define STDEXEC_IMMOVABLE(_XP) _XP(_XP&&)
#else
#define STDEXEC_IMMOVABLE(_XP) _XP(_XP&&) = delete
#endif

// NVBUG #4067067
#if STDEXEC_NVHPC()
#define STDEXEC_NO_UNIQUE_ADDRESS
#else
#define STDEXEC_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

// BUG (gcc PR93711): copy elision fails when initializing a
// [[no_unique_address]] field from a function returning an object
// of class type by value
#if STDEXEC_GCC()
#define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
#else
#define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

#if STDEXEC_CLANG() && defined(__CUDACC__)
#define STDEXEC_DETAIL_CUDACC_HOST_DEVICE __host__ __device__
#else
#define STDEXEC_DETAIL_CUDACC_HOST_DEVICE
#endif

#if STDEXEC_NVHPC()
#include <nv/target>
#define STDEXEC_TERMINATE() NV_IF_TARGET(NV_IS_HOST, (std::terminate();), (__trap();)) void()
#elif STDEXEC_CLANG() && defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define STDEXEC_TERMINATE() \
  __trap(); \
  __builtin_unreachable()
#else
#define STDEXEC_TERMINATE() std::terminate()
#endif

#ifdef STDEXEC_ASSERT
#error "Redefinition of STDEXEC_ASSERT is not permitted. Define STDEXEC_ASSERT_FN instead."
#endif

#define STDEXEC_ASSERT(_XP) \
  do { \
    static_assert(noexcept(_XP)); \
    STDEXEC_ASSERT_FN(_XP); \
  } while (false)

#ifndef STDEXEC_ASSERT_FN
#define STDEXEC_ASSERT_FN assert
#endif

namespace stdexec {
}
