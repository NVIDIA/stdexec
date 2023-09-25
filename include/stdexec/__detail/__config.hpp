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

#if __has_include(<version>)
#include <version>
#else
#include <ciso646> // For stdlib feature-test macros when <version> is not available
#endif

#include <cassert>
#include <version>

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

#if defined(__NVCC__)
#define STDEXEC_NVCC() 1
#elif defined(__NVCOMPILER)
#define STDEXEC_NVHPC() 1
#elif defined(__EDG__)
#define LEGATE_EDG() 1
#elif defined(__clang__)
#define STDEXEC_CLANG() 1
#elif defined(__GNUC__)
#define STDEXEC_GCC() 1
#elif defined(_MSC_VER)
#define STDEXEC_MSVC() 1
#endif

#ifndef STDEXEC_NVCC
#define STDEXEC_NVCC() 0
#endif
#ifndef STDEXEC_NVHPC
#define STDEXEC_NVHPC() 0
#endif
#ifndef STDEXEC_EDG
#define STDEXEC_EDG() 0
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

#define STDEXEC_STRINGIZE(_ARG) #_ARG

#if STDEXEC_NVCC()
#define STDEXEC_PRAGMA_PUSH() _Pragma("nv_diagnostic push")
#define STDEXEC_PRAGMA_POP() _Pragma("nv_diagnostic pop")
#define STDEXEC_PRAGMA_IGNORE_EDG(...) _Pragma(STDEXEC_STRINGIZE(nv_diag_suppress __VA_ARGS__))
#elif STDEXEC_NVHPC() || STDEXEC_EDG()
#define STDEXEC_PRAGMA_PUSH() \
  _Pragma("diagnostic push") STDEXEC_PRAGMA_IGNORE_EDG(invalid_error_number)
#define STDEXEC_PRAGMA_POP() _Pragma("diagnostic pop")
#define STDEXEC_PRAGMA_IGNORE_EDG(...) _Pragma(STDEXEC_STRINGIZE(diag_suppress __VA_ARGS__))
#elif STDEXEC_CLANG() || STDEXEC_GCC()
#define STDEXEC_PRAGMA_PUSH() _Pragma("GCC diagnostic push")
#define STDEXEC_PRAGMA_POP() _Pragma("GCC diagnostic pop")
#define STDEXEC_PRAGMA_IGNORE_GNU(_ARG) _Pragma(STDEXEC_STRINGIZE(GCC diagnostic ignored _ARG))
#else
#define STDEXEC_PRAGMA_PUSH()
#define STDEXEC_PRAGMA_POP()
#endif

#ifndef STDEXEC_PRAGMA_IGNORE_GNU
#define STDEXEC_PRAGMA_IGNORE_GNU(_ARG)
#endif
#ifndef STDEXEC_PRAGMA_IGNORE_EDG
#define STDEXEC_PRAGMA_IGNORE_EDG(_ARG)
#endif

#if !STDEXEC_MSVC() && defined(__has_builtin)
#define STDEXEC_HAS_BUILTIN __has_builtin
#else
#define STDEXEC_HAS_BUILTIN(...) 0
#endif

#if STDEXEC_HAS_BUILTIN(__is_trivially_copyable) || STDEXEC_MSVC()
#define STDEXEC_IS_TRIVIALLY_COPYABLE(...) __is_trivially_copyable(__VA_ARGS__)
#else
#define STDEXEC_IS_TRIVIALLY_COPYABLE(...) std::is_trivially_copyable_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_base_of) || (_MSC_VER >= 1914)
#define STDEXEC_IS_BASE_OF(...) __is_base_of(__VA_ARGS__)
#else
#define STDEXEC_IS_BASE_OF(...) std::is_base_of_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_convertible_to) || STDEXEC_MSVC()
#define STDEXEC_IS_CONVERTIBLE_TO(...) __is_convertible_to(__VA_ARGS__)
#elif STDEXEC_HAS_BUILTIN(__is_convertible)
#define STDEXEC_IS_CONVERTIBLE_TO(...) __is_convertible(__VA_ARGS__)
#else
#define STDEXEC_IS_CONVERTIBLE_TO(...) std::is_convertible_v<__VA_ARGS__>
#endif

#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define STDEXEC_UNREACHABLE() std::unreachable()
#elif STDEXEC_HAS_BUILTIN(__builtin_unreachable)
#define STDEXEC_UNREACHABLE() __builtin_unreachable()
#elif STDEXEC_MSVC()
#define STDEXEC_UNREACHABLE(...) __assume(false)
#else
#define STDEXEC_UNREACHABLE(...) std::terminate()
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

// Before clang-16, clang did not like libstdc++'s ranges implementation
#if __has_include(<ranges>) && \
  (defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L) && \
  (!STDEXEC_CLANG() || __clang_major__ >= 16 || defined(_LIBCPP_VERSION))
#define STDEXEC_HAS_STD_RANGES() 1
#else
#define STDEXEC_HAS_STD_RANGES() 0
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

#define STDEXEC_AUTO_RETURN(...) \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) { \
    return __VA_ARGS__; \
  }

// GCC 13 implements lexical friendship, but it is incomplete. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111018
#if STDEXEC_CLANG() // || (STDEXEC_GCC() && __GNUC__ >= 13)
#define STDEXEC_FRIENDSHIP_IS_LEXICAL() 1
#else
#define STDEXEC_FRIENDSHIP_IS_LEXICAL() 0
#endif

#if defined(__cpp_explicit_this_parameter) && (__cpp_explicit_this_parameter >= 202110)
#define STDEXEC_HAS_EXPLICIT_THIS() 1
#else
#define STDEXEC_HAS_EXPLICIT_THIS() 0
#endif

#if STDEXEC_HAS_EXPLICIT_THIS()
#define STDEXEC_DEFINE_EXPLICIT_THIS_MEMFN(...) __VA_ARGS__
#define STDEXEC_CALL_EXPLICIT_THIS_MEMFN(_OBJ, _NAME) (_OBJ)._NAME( STDEXEC_CALL_EXPLICIT_THIS_MEMFN_DETAIL
#define STDEXEC_CALL_EXPLICIT_THIS_MEMFN_DETAIL(...) __VA_ARGS__ )
#else
#define STDEXEC_DEFINE_EXPLICIT_THIS_MEMFN(...) static __VA_ARGS__(STDEXEC_FUN_ARGS
#define STDEXEC_CALL_EXPLICIT_THIS_MEMFN(_OBJ, _NAME) (_OBJ)._NAME((_OBJ) STDEXEC_CALL_EXPLICIT_THIS_MEMFN_DETAIL
#define STDEXEC_CALL_EXPLICIT_THIS_MEMFN_DETAIL(...) __VA_OPT__(, ) __VA_ARGS__)
#define STDEXEC_EAT_THIS_DETAIL_this
#define STDEXEC_FUN_ARGS(...) STDEXEC_CAT(STDEXEC_EAT_THIS_DETAIL_, __VA_ARGS__))
#endif

#ifndef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
// Compile times are bad enough on nvhpc. Disable extra type checking by default.
#if STDEXEC_NVHPC()
#define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 0
#else
#define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 1
#endif
#endif

namespace stdexec {
}
