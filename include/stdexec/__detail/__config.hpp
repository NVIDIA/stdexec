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

#define STDEXEC_CAT_(X, ...) X##__VA_ARGS__
#define STDEXEC_CAT(X, ...) STDEXEC_CAT_(X, __VA_ARGS__)

#define STDEXEC_EXPAND(...) __VA_ARGS__
#define STDEXEC_EVAL(M, ...) M(__VA_ARGS__)

#define STDEXEC_NOT(X) STDEXEC_CAT(STDEXEC_NOT_, X)
#define STDEXEC_NOT_0 1
#define STDEXEC_NOT_1 0

#define STDEXEC_IIF_0(Y, ...) __VA_ARGS__
#define STDEXEC_IIF_1(Y, ...) Y
#define STDEXEC_IIF(X, Y, ...) STDEXEC_EVAL(STDEXEC_CAT(STDEXEC_IIF_, X), Y, __VA_ARGS__)

#define STDEXEC_COUNT(...) \
  STDEXEC_EXPAND(STDEXEC_COUNT_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define STDEXEC_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _N, ...) _N

#define STDEXEC_CHECK(...) STDEXEC_EXPAND(STDEXEC_CHECK_N(__VA_ARGS__, 0, ))
#define STDEXEC_CHECK_N(x, n, ...) n
#define STDEXEC_PROBE(x) x, 1,

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

#if STDEXEC_CLANG() && defined(__CUDACC__)
#define STDEXEC_DETAIL_CUDACC_HOST_DEVICE __host__ __device__
#else
#define STDEXEC_DETAIL_CUDACC_HOST_DEVICE
#endif

#ifdef STDEXEC_ASSERT
#error "Redefinition of STDEXEC_ASSERT is not permitted. Define STDEXEC_ASSERT_FN instead."
#endif

#define STDEXEC_ASSERT(_X) \
  do { \
    static_assert(noexcept(_X)); \
    STDEXEC_ASSERT_FN(_X); \
  } while (false)

#ifndef STDEXEC_ASSERT_FN
#define STDEXEC_ASSERT_FN assert
#endif

// #if __cpp_explicit_this_parameter >= 202110
//   #define STDEXEC_USE_EXPLICIT_THIS
// #endif
#define STDEXEC_USE_TAG_INVOKE

#define STDEXEC_EAT_THIS_this

#if defined(STDEXEC_USE_EXPLICIT_THIS)

  #define STDEXEC_DEFINE_CUSTOM(AUTO_NAME) \
    AUTO_NAME \
    /**/
  #define STDEXEC_CALL_CUSTOM(NAME, OBJ, ...) \
    (OBJ).NAME(__VA_ARGS__) \
    /**/

#elif defined(STDEXEC_USE_TAG_INVOKE)

  #define STDEXEC_DEFINE_CUSTOM(AUTO_NAME) \
    friend auto tag_invoke(STDEXEC_FUN_ARGS \
    /**/
  #define STDEXEC_FUN_ARGS(SELF, TAG, ...) \
    TAG, STDEXEC_CAT(STDEXEC_EAT_THIS_, SELF) __VA_OPT__(,) __VA_ARGS__) \
    /**/
  #define STDEXEC_CALL_CUSTOM(NAME, OBJ, TAG, ...) \
    tag_invoke(TAG, OBJ, __VA_ARGS__) \
    /**/

#else

  #define STDEXEC_DEFINE_CUSTOM(AUTO_NAME) \
    static AUTO_NAME(STDEXEC_FUN_ARGS \
    /**/
  #define STDEXEC_FUN_ARGS(...) \
    STDEXEC_CAT(STDEXEC_EAT_THIS_, __VA_ARGS__)) \
    /**/
  #define STDEXEC_CALL_CUSTOM(NAME, OBJ, ...) \
    (OBJ).NAME((OBJ), __VA_ARGS__) \
    /**/

#endif
