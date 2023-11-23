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

#define STDEXEC_STRINGIZE(_ARG) #_ARG

#define STDEXEC_CAT_(_XP, ...) _XP##__VA_ARGS__
#define STDEXEC_CAT(_XP, ...) STDEXEC_CAT_(_XP, __VA_ARGS__)

#define STDEXEC_EXPAND(...) __VA_ARGS__
#define STDEXEC_EVAL(_MACRO, ...) _MACRO(__VA_ARGS__)
#define STDEXEC_EAT(...)

#define STDEXEC_NOT(_XP) STDEXEC_CAT(STDEXEC_NOT_, _XP)
#define STDEXEC_NOT_0 1
#define STDEXEC_NOT_1 0

#define STDEXEC_IIF_0(_YP, ...) __VA_ARGS__
#define STDEXEC_IIF_1(_YP, ...) _YP
#define STDEXEC_IIF(_XP, _YP, ...) STDEXEC_EVAL(STDEXEC_CAT(STDEXEC_IIF_, _XP), _YP, __VA_ARGS__)

#define STDEXEC_COUNT(...) \
  STDEXEC_EXPAND(STDEXEC_COUNT_(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define STDEXEC_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _NP, ...) _NP

#define STDEXEC_CHECK(...) STDEXEC_EXPAND(STDEXEC_CHECK_(__VA_ARGS__, 0, ))
#define STDEXEC_CHECK_(_XP, _NP, ...) _NP
#define STDEXEC_PROBE(...) STDEXEC_PROBE_(__VA_ARGS__, 1)
#define STDEXEC_PROBE_(_XP, _NP, ...) _XP, _NP,

////////////////////////////////////////////////////////////////////////////////
// STDEXEC_FOR_EACH
//   Inspired by "Recursive macros with C++20 __VA_OPT__", by David Mazières
//   https://www.scs.stanford.edu/~dm/blog/va-opt.html
#define STDEXEC_EXPAND_R(...) \
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
#define STDEXEC_FOR_EACH(_MACRO, ...) \
  __VA_OPT__(STDEXEC_EXPAND_R(STDEXEC_FOR_EACH_HELPER(_MACRO, __VA_ARGS__))) \
  /**/
#define STDEXEC_FOR_EACH_HELPER(_MACRO, _A1, ...) \
  _MACRO(_A1) __VA_OPT__(STDEXEC_FOR_EACH_AGAIN STDEXEC_PARENS(_MACRO, __VA_ARGS__)) /**/
#define STDEXEC_FOR_EACH_AGAIN() STDEXEC_FOR_EACH_HELPER
////////////////////////////////////////////////////////////////////////////////////////////////////

// If tail is non-empty, expand to the tail. Otherwise, expand to the head
#define STDEXEC_HEAD_OR_TAIL(_XP, ...) STDEXEC_EXPAND __VA_OPT__((__VA_ARGS__) STDEXEC_EAT)(_XP)

// If tail is non-empty, expand to nothing. Otherwise, expand to the head
#define STDEXEC_HEAD_OR_NULL(_XP, ...) STDEXEC_EXPAND __VA_OPT__(() STDEXEC_EAT)(_XP)

// When used with no arguments, these macros expand to 1 if the current
// compiler corresponds to the macro name; 0, otherwise. When used with arguments,
// they expand to the arguments if if the current compiler corresponds to the
// macro name; nothing, otherwise.
#if defined(__NVCC__)
#define STDEXEC_NVCC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#elif defined(__NVCOMPILER)
#define STDEXEC_NVHPC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#elif defined(__EDG__)
#define STDEXEC_EDG(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#elif defined(__clang__)
#define STDEXEC_CLANG(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#if defined(_MSC_VER)
#define STDEXEC_CLANG_CL(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#endif
#elif defined(__GNUC__)
#define STDEXEC_GCC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#elif defined(_MSC_VER)
#define STDEXEC_MSVC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#endif

#ifndef STDEXEC_NVCC
#define STDEXEC_NVCC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_NVHPC
#define STDEXEC_NVHPC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_EDG
#define STDEXEC_EDG(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_CLANG
#define STDEXEC_CLANG(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_CLANG_CL
#define STDEXEC_CLANG_CL(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_GCC
#define STDEXEC_GCC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_MSVC
#define STDEXEC_MSVC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
#define STDEXEC_CUDA(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#else
#define STDEXEC_CUDA(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// For portably declaring attributes on functions and types
//   Usage:
//
//   STDEXEC_ATTRIBUTE((attr1, attr2, ...))
//   void foo() { ... }
#define STDEXEC_ATTRIBUTE(_XP) STDEXEC_FOR_EACH(STDEXEC_ATTR, STDEXEC_EXPAND _XP)
#define STDEXEC_ATTR(_ATTR) \
  STDEXEC_CAT(STDEXEC_ATTR_WHICH_, STDEXEC_CHECK(STDEXEC_CAT(STDEXEC_ATTR_, _ATTR)))(_ATTR)

// unknown attributes are treated like C++-style attributes
#define STDEXEC_ATTR_WHICH_0(_ATTR) [[_ATTR]]

// custom handling for specific attribute types
#define STDEXEC_ATTR_WHICH_1(_ATTR) STDEXEC_CUDA(__host__)
#define STDEXEC_ATTR_host STDEXEC_PROBE(~, 1)
#define STDEXEC_ATTR___host__ STDEXEC_PROBE(~, 1)

#define STDEXEC_ATTR_WHICH_2(_ATTR) STDEXEC_CUDA(__device__)
#define STDEXEC_ATTR_device STDEXEC_PROBE(~, 2)
#define STDEXEC_ATTR___device__ STDEXEC_PROBE(~, 2)

#if STDEXEC_NVHPC()
// NVBUG #4067067: NVHPC does not fully support [[no_unique_address]]
#define STDEXEC_ATTR_WHICH_3(_ATTR) /*nothing*/
#elif STDEXEC_MSVC()
// MSVCBUG https://developercommunity.visualstudio.com/t/Incorrect-codegen-when-using-msvc::no_/10452874
#define STDEXEC_ATTR_WHICH_3(_ATTR) // [[msvc::no_unique_address]]
#elif STDEXEC_CLANG_CL()
// clang-cl does not support: https://reviews.llvm.org/D110485
#define STDEXEC_ATTR_WHICH_3(_ATTR) // [[msvc::no_unique_address]]
#else
#define STDEXEC_ATTR_WHICH_3(_ATTR) [[no_unique_address]]
#endif
#define STDEXEC_ATTR_no_unique_address STDEXEC_PROBE(~, 3)

#if STDEXEC_MSVC()
#define STDEXEC_ATTR_WHICH_4(_ATTR) __forceinline
#elif STDEXEC_CLANG()
#define STDEXEC_ATTR_WHICH_4(_ATTR) \
  __attribute__((__always_inline__, __artificial__, __nodebug__)) inline
#elif defined(__GNUC__)
#define STDEXEC_ATTR_WHICH_4(_ATTR) __attribute__((__always_inline__, __artificial__)) inline
#else
#define STDEXEC_ATTR_WHICH_4(_ATTR) /*nothing*/
#endif
#define STDEXEC_ATTR_always_inline STDEXEC_PROBE(~, 4)

////////////////////////////////////////////////////////////////////////////////////////////////////
// warning push/pop portability macros
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
#define STDEXEC_PRAGMA_IGNORE_GNU(...) \
  _Pragma(STDEXEC_STRINGIZE(GCC diagnostic ignored __VA_ARGS__))
#else
#define STDEXEC_PRAGMA_PUSH()
#define STDEXEC_PRAGMA_POP()
#endif

#ifndef STDEXEC_PRAGMA_IGNORE_GNU
#define STDEXEC_PRAGMA_IGNORE_GNU(...)
#endif
#ifndef STDEXEC_PRAGMA_IGNORE_EDG
#define STDEXEC_PRAGMA_IGNORE_EDG(...)
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

#if STDEXEC_HAS_BUILTIN(__is_const)
#define STDEXEC_IS_CONST(...) __is_const(__VA_ARGS__)
#else
#define STDEXEC_IS_CONST(...) stdexec::__is_const<__VA_ARGS__>
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

// BUG (gcc PR93711): copy elision fails when initializing a
// [[no_unique_address]] field from a function returning an object
// of class type by value
#if STDEXEC_GCC()
#define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
#else
#define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS STDEXEC_ATTRIBUTE((no_unique_address))
#endif

#if STDEXEC_NVHPC()
#include <nv/target>
#define STDEXEC_TERMINATE() NV_IF_TARGET(NV_IS_HOST, (std::terminate();), (__trap();)) void()
#elif STDEXEC_CLANG() && STDEXEC_CUDA() && defined(__CUDA_ARCH__)
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

#if __has_include(<memory_resource>) && \
  (defined(__cpp_lib_memory_resource) && __cpp_lib_memory_resource >= 201603L)
#define STDEXEC_HAS_STD_MEMORY_RESOURCE() 1
#else
#define STDEXEC_HAS_STD_MEMORY_RESOURCE() 0
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
#define STDEXEC_EXPLICIT_THIS(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#else
#define STDEXEC_EXPLICIT_THIS(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

#if STDEXEC_EXPLICIT_THIS()
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

// Configure extra type checking
#define STDEXEC_TYPE_CHECKING_ZERO() 0
#define STDEXEC_TYPE_CHECKING_ONE() 1
#define STDEXEC_TYPE_CHECKING_TWO() 2

#define STDEXEC_PROBE_TYPE_CHECKING_ STDEXEC_TYPE_CHECKING_ONE
#define STDEXEC_PROBE_TYPE_CHECKING_0 STDEXEC_TYPE_CHECKING_ZERO
#define STDEXEC_PROBE_TYPE_CHECKING_1 STDEXEC_TYPE_CHECKING_ONE
#define STDEXEC_PROBE_TYPE_CHECKING_STDEXEC_ENABLE_EXTRA_TYPE_CHECKING STDEXEC_TYPE_CHECKING_TWO

#define STDEXEC_TYPE_CHECKING_WHICH3(...) STDEXEC_PROBE_TYPE_CHECKING_##__VA_ARGS__
#define STDEXEC_TYPE_CHECKING_WHICH2(...) STDEXEC_TYPE_CHECKING_WHICH3(__VA_ARGS__)
#define STDEXEC_TYPE_CHECKING_WHICH STDEXEC_TYPE_CHECKING_WHICH2(STDEXEC_ENABLE_EXTRA_TYPE_CHECKING)

#ifndef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
#define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 0
#elif STDEXEC_TYPE_CHECKING_WHICH() == 2
// do nothing
#elif STDEXEC_TYPE_CHECKING_WHICH() == 0
#undef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
#define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 0
#else
#undef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
#define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 1
#endif

namespace stdexec {
}
