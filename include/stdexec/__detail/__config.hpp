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

#define STDEXEC_CAT_(_X, ...) _X ## __VA_ARGS__
#define STDEXEC_CAT(_X, ...) STDEXEC_CAT_(_X, __VA_ARGS__)

#define STDEXEC_EXPAND(...) __VA_ARGS__
#define STDEXEC_EVAL(_M, ...) _M(__VA_ARGS__)
#define STDEXEC_EAT(...)

////////////////////////////////////////////////////////////////////////////////
// STDEXEC_FOR_EACH
//   Inspired by "Recursive macros with C++20 __VA_OPT__", by David Mazières
//   https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define STDEXEC_EXPAND_R(...)  \
  STDEXEC_EXPAND_R1(STDEXEC_EXPAND_R1(STDEXEC_EXPAND_R1(STDEXEC_EXPAND_R1(__VA_ARGS__)))) \
  /**/
#define STDEXEC_EXPAND_R1(...) \
  STDEXEC_EXPAND_R2(STDEXEC_EXPAND_R2(STDEXEC_EXPAND_R2(STDEXEC_EXPAND_R2(__VA_ARGS__)))) \
  /**/
#define STDEXEC_EXPAND_R2(...) \
  STDEXEC_EXPAND_R3(STDEXEC_EXPAND_R3(STDEXEC_EXPAND_R3(STDEXEC_EXPAND_R3(__VA_ARGS__)))) \
  /**/
#define STDEXEC_EXPAND_R3(...) \
  STDEXEC_EXPAND(STDEXEC_EXPAND(STDEXEC_EXPAND(STDEXEC_EXPAND(__VA_ARGS__)))) \
  /**/

#define STDEXEC_PARENS ()
#define STDEXEC_FOR_EACH(_M, ...)                                         \
  __VA_OPT__(STDEXEC_EXPAND_R(STDEXEC_FOR_EACH_HELPER(_M, __VA_ARGS__)))  \
  /**/
#define STDEXEC_FOR_EACH_HELPER(_M, _A1, ...)                                   \
  _M(_A1) __VA_OPT__(STDEXEC_FOR_EACH_AGAIN STDEXEC_PARENS (_M, __VA_ARGS__))   \
  /**/
#define STDEXEC_FOR_EACH_AGAIN() STDEXEC_FOR_EACH_HELPER
////////////////////////////////////////////////////////////////////////////////

#define STDEXEC_NOT(_X) STDEXEC_CAT(STDEXEC_NOT_, _X)
#define STDEXEC_NOT_0 1
#define STDEXEC_NOT_1 0

#define STDEXEC_IIF_0(_Y, ...) __VA_ARGS__
#define STDEXEC_IIF_1(_Y, ...) _Y
#define STDEXEC_IIF(_X, _Y, ...) STDEXEC_EVAL(STDEXEC_CAT(STDEXEC_IIF_, _X), _Y, __VA_ARGS__)

#define STDEXEC_COUNT_M(...) +1
#define STDEXEC_COUNT(...) (0 STDEXEC_FOR_EACH(STDEXEC_COUNT_M, __VA_ARGS_))

#define STDEXEC_CHECK(...) STDEXEC_EXPAND(STDEXEC_CHECK_N(__VA_ARGS__, 0,))
#define STDEXEC_CHECK_N(_X, _N, ...) _N
#define STDEXEC_PROBE(_X) _X, 1,
#define STDEXEC_PROBE_N(_X, _N) _X, _N,

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
#define STDEXEC_STRINGIZE(__arg) #__arg
#define STDEXEC_PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define STDEXEC_PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define STDEXEC_PRAGMA_IGNORE(__arg) _Pragma(STDEXEC_STRINGIZE(GCC diagnostic ignored __arg))
#else
#define STDEXEC_PRAGMA_PUSH()
#define STDEXEC_PRAGMA_POP()
#define STDEXEC_PRAGMA_IGNORE(__arg)
#endif

#ifdef __has_builtin
#define STDEXEC_HAS_BUILTIN __has_builtin
#else
#define STDEXEC_HAS_BUILTIN(...) 0
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

  #define STDEXEC_DEFINE_CUSTOM(...) \
    __VA_ARGS__ \
    /**/
  #define STDEXEC_CALL_CUSTOM(NAME, OBJ, ...) \
    (OBJ).NAME(__VA_ARGS__) \
    /**/

#elif defined(STDEXEC_USE_TAG_INVOKE)

  #define STDEXEC_PROBE_RETURN_TYPE_auto STDEXEC_PROBE(~)

  #define STDEXEC_RETURN_TYPE(_TY, ...) \
    STDEXEC_CAT( \
      STDEXEC_RETURN_TYPE_, \
      STDEXEC_CHECK( \
        STDEXEC_CAT(STDEXEC_PROBE_RETURN_TYPE_, _TY)))( \
          _TY __VA_OPT__(,) __VA_ARGS__) \
    /**/
  #define STDEXEC_RETURN_TYPE_0(...) \
    ::stdexec::__arg_type_t<void(__VA_ARGS__())> \
    /**/
  #define STDEXEC_RETURN_TYPE_1(...) \
    auto \
    /**/
  #define STDEXEC_DEFINE_CUSTOM(...) \
    friend STDEXEC_RETURN_TYPE(__VA_ARGS__) tag_invoke(STDEXEC_FUN_ARGS \
    /**/
  #define STDEXEC_FUN_ARGS(SELF, TAG, ...) \
    TAG, STDEXEC_CAT(STDEXEC_EAT_THIS_, SELF) __VA_OPT__(,) __VA_ARGS__) \
    /**/
  #define STDEXEC_CALL_CUSTOM(NAME, OBJ, TAG, ...) \
    ::stdexec::tag_invoke(TAG, OBJ __VA_OPT__(,) __VA_ARGS__) \
    /**/

#else

  #define STDEXEC_DEFINE_CUSTOM(...) \
    static __VA_ARGS__(STDEXEC_FUN_ARGS \
    /**/
  #define STDEXEC_FUN_ARGS(...) \
    STDEXEC_CAT(STDEXEC_EAT_THIS_, __VA_ARGS__)) \
    /**/
  #define STDEXEC_CALL_CUSTOM(NAME, OBJ, ...) \
    (OBJ).NAME((OBJ) __VA_OPT__(,) __VA_ARGS__) \
    /**/

#endif

namespace stdexec {
  template <class _Ty>
  _Ty __arg_type(void(*)(_Ty(*)()));

  template <class _Ty>
  using __arg_type_t = decltype(stdexec::__arg_type((_Ty*) nullptr));
}
