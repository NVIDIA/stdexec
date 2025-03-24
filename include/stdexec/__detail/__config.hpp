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

// IWYU pragma: always_keep

#if __cplusplus < 202002L
#  if defined(_MSC_VER) && !defined(__clang__)
#    error This library requires the use of C++20. Use /Zc:__cplusplus to enable __cplusplus conformance.
#  else
#    error This library requires the use of C++20.
#  endif
#endif

#if defined(_MSC_VER) && !defined(__clang__) && (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL)
#  error This library requires the use of the new conforming preprocessor enabled by /Zc:preprocessor.
#endif

#include "__preprocessor.hpp"

#if __has_include(<version>)
#  include <version>
#else
#  include <ciso646> // For stdlib feature-test macros when <version> is not available
#endif

#include <cassert>
#include <type_traits> // IWYU pragma: keep

// When used with no arguments, these macros expand to 1 if the current
// compiler corresponds to the macro name; 0, otherwise. When used with arguments,
// they expand to the arguments if if the current compiler corresponds to the
// macro name; nothing, otherwise.
#if defined(__NVCC__)
#  define STDEXEC_NVCC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#elif defined(__EDG__)
#  define STDEXEC_EDG(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  if defined(__NVCOMPILER)
#    define STDEXEC_NVHPC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  endif
#  if defined(__INTELLISENSE__)
#    define STDEXEC_INTELLISENSE(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#    define STDEXEC_MSVC_HEADERS(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  endif
#elif defined(__clang__)
#  define STDEXEC_CLANG(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  if defined(_MSC_VER)
#    define STDEXEC_CLANG_CL(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  endif
#  if defined(__apple_build_version__)
#    define STDEXEC_APPLE_CLANG(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  endif
#elif defined(__GNUC__)
#  define STDEXEC_GCC(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#elif defined(_MSC_VER)
#  define STDEXEC_MSVC(...)         STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#  define STDEXEC_MSVC_HEADERS(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#endif

#ifndef STDEXEC_NVCC
#  define STDEXEC_NVCC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_NVHPC
#  define STDEXEC_NVHPC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_EDG
#  define STDEXEC_EDG(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_CLANG
#  define STDEXEC_CLANG(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_CLANG_CL
#  define STDEXEC_CLANG_CL(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_APPLE_CLANG
#  define STDEXEC_APPLE_CLANG(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_GCC
#  define STDEXEC_GCC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_MSVC
#  define STDEXEC_MSVC(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_MSVC_HEADERS
#  define STDEXEC_MSVC_HEADERS(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif
#ifndef STDEXEC_INTELLISENSE
#  define STDEXEC_INTELLISENSE(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

#if STDEXEC_NVHPC()
#  define STDEXEC_NVHPC_VERSION() (__NVCOMPILER_MAJOR__ * 100 + __NVCOMPILER_MINOR__)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
#  define STDEXEC_CUDA(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#else
#  define STDEXEC_CUDA(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
#if STDEXEC_CLANG() && STDEXEC_CUDA()
#  define STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __host__ __device__
#else
#  define STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
#if __cpp_impl_coroutine >= 201902 && __cpp_lib_coroutine >= 201902
#  include <coroutine> // IWYU pragma: keep
#  define STDEXEC_STD_NO_COROUTINES() 0
namespace __coro = std; // NOLINT(misc-unused-alias-decls)
#elif defined(__cpp_coroutines) && __has_include(<experimental/coroutine>)
#  include <experimental/coroutine>
#  define STDEXEC_STD_NO_COROUTINES() 0
namespace __coro = std::experimental;
#else
#  define STDEXEC_STD_NO_COROUTINES() 1
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// For portably declaring attributes on functions and types
//   Usage:
//
//   STDEXEC_ATTRIBUTE((attr1, attr2, ...))
//   void foo() { ... }
#define STDEXEC_ATTRIBUTE(_XP) STDEXEC_FOR_EACH(STDEXEC_ATTR, STDEXEC_EXPAND _XP)
#define STDEXEC_ATTR(_ATTR)                                                                        \
  STDEXEC_CAT(STDEXEC_ATTR_WHICH_, STDEXEC_CHECK(STDEXEC_CAT(STDEXEC_ATTR_, _ATTR)))(_ATTR)

// unknown attributes are treated like C++-style attributes
#define STDEXEC_ATTR_WHICH_0(_ATTR) [[_ATTR]]

// custom handling for specific attribute types
#ifdef __CUDACC__
#  define STDEXEC_ATTR_WHICH_1(_ATTR) __host__
#else
#  define STDEXEC_ATTR_WHICH_1(_ATTR)
#endif
#define STDEXEC_ATTR_host     STDEXEC_PROBE(~, 1)
#define STDEXEC_ATTR___host__ STDEXEC_PROBE(~, 1)

#ifdef __CUDACC__
#  define STDEXEC_ATTR_WHICH_2(_ATTR) __device__
#else
#  define STDEXEC_ATTR_WHICH_2(_ATTR)
#endif
#define STDEXEC_ATTR_device     STDEXEC_PROBE(~, 2)
#define STDEXEC_ATTR___device__ STDEXEC_PROBE(~, 2)

#if STDEXEC_NVHPC()
// NVBUG #4067067: NVHPC does not fully support [[no_unique_address]]
#  define STDEXEC_ATTR_WHICH_3(_ATTR) /*nothing*/
#elif STDEXEC_MSVC()
// MSVCBUG https://developercommunity.visualstudio.com/t/Incorrect-codegen-when-using-msvc::no_/10452874
#  define STDEXEC_ATTR_WHICH_3(_ATTR) // [[msvc::no_unique_address]]
#elif STDEXEC_CLANG_CL()
// clang-cl does not support: https://reviews.llvm.org/D110485
#  define STDEXEC_ATTR_WHICH_3(_ATTR) // [[msvc::no_unique_address]]
#else
#  define STDEXEC_ATTR_WHICH_3(_ATTR) [[no_unique_address]]
#endif
#define STDEXEC_ATTR_no_unique_address STDEXEC_PROBE(~, 3)

#if STDEXEC_MSVC()
#  define STDEXEC_ATTR_WHICH_4(_ATTR) __forceinline
#elif STDEXEC_CLANG()
#  define STDEXEC_ATTR_WHICH_4(_ATTR)                                                              \
    __attribute__((__always_inline__, __artificial__, __nodebug__)) inline
#elif defined(__GNUC__)
#  define STDEXEC_ATTR_WHICH_4(_ATTR) __attribute__((__always_inline__, __artificial__)) inline
#else
#  define STDEXEC_ATTR_WHICH_4(_ATTR) /*nothing*/
#endif
#define STDEXEC_ATTR_always_inline STDEXEC_PROBE(~, 4)

#if STDEXEC_CLANG() || STDEXEC_GCC()
#  define STDEXEC_ATTR_WHICH_5(_ATTR) __attribute__((__weak__))
#else
#  define STDEXEC_ATTR_WHICH_5(_ATTR) /*nothing*/
#endif
#define STDEXEC_ATTR_weak     STDEXEC_PROBE(~, 5)
#define STDEXEC_ATTR___weak__ STDEXEC_PROBE(~, 5)

////////////////////////////////////////////////////////////////////////////////////////////////////
// warning push/pop portability macros
#if STDEXEC_NVCC()
#  define STDEXEC_PRAGMA_PUSH()          _Pragma("nv_diagnostic push")
#  define STDEXEC_PRAGMA_POP()           _Pragma("nv_diagnostic pop")
#  define STDEXEC_PRAGMA_IGNORE_EDG(...) _Pragma(STDEXEC_STRINGIZE(nv_diag_suppress __VA_ARGS__))
#elif STDEXEC_EDG()
#  define STDEXEC_PRAGMA_PUSH()                                                                    \
    _Pragma("diagnostic push") STDEXEC_PRAGMA_IGNORE_EDG(invalid_error_number)                     \
      STDEXEC_PRAGMA_IGNORE_EDG(invalid_error_tag)
#  define STDEXEC_PRAGMA_POP()           _Pragma("diagnostic pop")
#  define STDEXEC_PRAGMA_IGNORE_EDG(...) _Pragma(STDEXEC_STRINGIZE(diag_suppress __VA_ARGS__))
#elif STDEXEC_CLANG() || STDEXEC_GCC()
#  define STDEXEC_PRAGMA_PUSH()                                                                    \
    _Pragma("GCC diagnostic push") STDEXEC_PRAGMA_IGNORE_GNU("-Wpragmas")                          \
      STDEXEC_PRAGMA_IGNORE_GNU("-Wunknown-pragmas")                                               \
        STDEXEC_PRAGMA_IGNORE_GNU("-Wunknown-warning-option")                                      \
          STDEXEC_PRAGMA_IGNORE_GNU("-Wunknown-attributes")                                        \
            STDEXEC_PRAGMA_IGNORE_GNU("-Wattributes")
#  define STDEXEC_PRAGMA_POP() _Pragma("GCC diagnostic pop")
#  define STDEXEC_PRAGMA_IGNORE_GNU(...)                                                           \
    _Pragma(STDEXEC_STRINGIZE(GCC diagnostic ignored __VA_ARGS__))
#elif STDEXEC_MSVC()
#  define STDEXEC_PRAGMA_PUSH()           __pragma(warning(push))
#  define STDEXEC_PRAGMA_POP()            __pragma(warning(pop))
#  define STDEXEC_PRAGMA_IGNORE_MSVC(...) __pragma(warning(disable : __VA_ARGS__))
#else
#  define STDEXEC_PRAGMA_PUSH()
#  define STDEXEC_PRAGMA_POP()
#endif

#ifndef STDEXEC_PRAGMA_IGNORE_GNU
#  define STDEXEC_PRAGMA_IGNORE_GNU(...)
#endif
#ifndef STDEXEC_PRAGMA_IGNORE_EDG
#  define STDEXEC_PRAGMA_IGNORE_EDG(...)
#endif
#ifndef STDEXEC_PRAGMA_IGNORE_MSVC
#  define STDEXEC_PRAGMA_IGNORE_MSVC(...)
#endif

#if !STDEXEC_MSVC() && defined(__has_builtin)
#  define STDEXEC_HAS_BUILTIN __has_builtin
#else
#  define STDEXEC_HAS_BUILTIN(...) 0
#endif

#if !STDEXEC_MSVC() && defined(__has_feature)
#  define STDEXEC_HAS_FEATURE __has_feature
#else
#  define STDEXEC_HAS_FEATURE(...) 0
#endif

#if STDEXEC_HAS_BUILTIN(__is_trivially_copyable) || STDEXEC_MSVC()
#  define STDEXEC_IS_TRIVIALLY_COPYABLE(...) __is_trivially_copyable(__VA_ARGS__)
#else
#  define STDEXEC_IS_TRIVIALLY_COPYABLE(...) std::is_trivially_copyable_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_base_of) || (_MSC_VER >= 1914)
#  define STDEXEC_IS_BASE_OF(...) __is_base_of(__VA_ARGS__)
#else
#  define STDEXEC_IS_BASE_OF(...) std::is_base_of_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_convertible_to) || STDEXEC_MSVC()
#  define STDEXEC_IS_CONVERTIBLE_TO(...) __is_convertible_to(__VA_ARGS__)
#elif STDEXEC_HAS_BUILTIN(__is_convertible)
#  define STDEXEC_IS_CONVERTIBLE_TO(...) __is_convertible(__VA_ARGS__)
#else
#  define STDEXEC_IS_CONVERTIBLE_TO(...) std::is_convertible_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_const)
#  define STDEXEC_IS_CONST(...) __is_const(__VA_ARGS__)
#else
#  define STDEXEC_IS_CONST(...) stdexec::__is_const_<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_function)
#  define STDEXEC_IS_FUNCTION(...) __is_function(__VA_ARGS__)
#else
#  define STDEXEC_IS_FUNCTION(...)                                                                 \
    (!STDEXEC_IS_CONST(__VA_ARGS__) && !STDEXEC_IS_CONST(const __VA_ARGS__))
#endif

#if STDEXEC_HAS_BUILTIN(__is_same)
#  define STDEXEC_IS_SAME(...) __is_same(__VA_ARGS__)
#elif STDEXEC_HAS_BUILTIN(__is_same_as)
#  define STDEXEC_IS_SAME(...) __is_same_as(__VA_ARGS__)
#elif STDEXEC_MSVC()
// msvc replaces std::is_same_v with a compile-time constant
#  define STDEXEC_IS_SAME(...) std::is_same_v<__VA_ARGS__>
#else
#  define STDEXEC_IS_SAME(...) stdexec::__same_as_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_constructible) || STDEXEC_MSVC()
#  define STDEXEC_IS_CONSTRUCTIBLE(...) __is_constructible(__VA_ARGS__)
#else
#  define STDEXEC_IS_CONSTRUCTIBLE(...) std::is_constructible_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_nothrow_constructible) || STDEXEC_MSVC()
#  define STDEXEC_IS_NOTHROW_CONSTRUCTIBLE(...) __is_nothrow_constructible(__VA_ARGS__)
#else
#  define STDEXEC_IS_NOTHROW_CONSTRUCTIBLE(...) std::is_nothrow_constructible_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_trivially_constructible) || STDEXEC_MSVC()
#  define STDEXEC_IS_TRIVIALLY_CONSTRUCTIBLE(...) __is_trivially_constructible(__VA_ARGS__)
#else
#  define STDEXEC_IS_TRIVIALLY_CONSTRUCTIBLE(...) std::is_trivially_constructible_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__is_empty) || STDEXEC_MSVC()
#  define STDEXEC_IS_EMPTY(...) __is_empty(__VA_ARGS__)
#else
#  define STDEXEC_IS_EMPTY(...) std::is_empty_v<__VA_ARGS__>
#endif

#if STDEXEC_HAS_BUILTIN(__remove_reference)
namespace stdexec {
  template <class Ty>
  using _remove_reference_t = __remove_reference(Ty);
} // namespace stdexec

#  define STDEXEC_REMOVE_REFERENCE(...) stdexec::_remove_reference_t<__VA_ARGS__>
#elif STDEXEC_HAS_BUILTIN(__remove_reference_t)
namespace stdexec {
  template <class Ty>
  using _remove_reference_t = __remove_reference_t(Ty);
} // namespace stdexec

#  define STDEXEC_REMOVE_REFERENCE(...) stdexec::_remove_reference_t<__VA_ARGS__>
#else
#  define STDEXEC_REMOVE_REFERENCE(...) ::std::remove_reference_t<__VA_ARGS__>
#endif

namespace stdexec {
  template <class _Ap, class _Bp>
  inline constexpr bool __same_as_v = false;

  template <class _Ap>
  inline constexpr bool __same_as_v<_Ap, _Ap> = true;
} // namespace stdexec

#if defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#  define STDEXEC_UNREACHABLE() std::unreachable()
#elif STDEXEC_HAS_BUILTIN(__builtin_unreachable)
#  define STDEXEC_UNREACHABLE() __builtin_unreachable()
#elif STDEXEC_MSVC()
#  define STDEXEC_UNREACHABLE(...) __assume(false)
#else
#  define STDEXEC_UNREACHABLE(...) std::terminate()
#endif

// Before gcc-12, gcc really didn't like tuples or variants of immovable types
#if STDEXEC_GCC() && (__GNUC__ < 12)
#  define STDEXEC_IMMOVABLE(_XP) _XP(_XP&&)
#else
#  define STDEXEC_IMMOVABLE(_XP) _XP(_XP&&) = delete
#endif

// BUG (gcc PR93711): copy elision fails when initializing a
// [[no_unique_address]] field from a function returning an object
// of class type by value
#if STDEXEC_GCC()
#  define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
#else
#  define STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS STDEXEC_ATTRIBUTE((no_unique_address))
#endif

#if STDEXEC_NVHPC()
#  include <nv/target>
#  define STDEXEC_TERMINATE() NV_IF_TARGET(NV_IS_HOST, (std::terminate();), (__trap();)) void()
#elif STDEXEC_CLANG() && STDEXEC_CUDA() && defined(__CUDA_ARCH__)
#  define STDEXEC_TERMINATE()                                                                      \
    __trap();                                                                                      \
    __builtin_unreachable()
#else
#  define STDEXEC_TERMINATE() std::terminate()
#endif

#if STDEXEC_HAS_FEATURE(thread_sanitizer) || defined(__SANITIZE_THREAD__)
#  define STDEXEC_TSAN(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#else
#  define STDEXEC_TSAN(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

// Before clang-16, clang did not like libstdc++'s ranges implementation
#if __has_include(<ranges>) && \
  (defined(__cpp_lib_ranges) && __cpp_lib_ranges >= 201911L) && \
  (!STDEXEC_CLANG() || __clang_major__ >= 16 || defined(_LIBCPP_VERSION))
#  define STDEXEC_HAS_STD_RANGES() 1
#else
#  define STDEXEC_HAS_STD_RANGES() 0
#endif

#if __has_include(<memory_resource>) && \
  (defined(__cpp_lib_memory_resource) && __cpp_lib_memory_resource >= 201603L)
#  define STDEXEC_HAS_STD_MEMORY_RESOURCE() 1
#else
#  define STDEXEC_HAS_STD_MEMORY_RESOURCE() 0
#endif

#ifdef STDEXEC_ASSERT
#  error "Redefinition of STDEXEC_ASSERT is not permitted. Define STDEXEC_ASSERT_FN instead."
#endif

#define STDEXEC_ASSERT(_XP)                                                                        \
  do {                                                                                             \
    static_assert(noexcept(_XP));                                                                  \
    STDEXEC_ASSERT_FN(_XP);                                                                        \
  } while (false)

#ifndef STDEXEC_ASSERT_FN
#  define STDEXEC_ASSERT_FN assert
#endif

#define STDEXEC_AUTO_RETURN(...)                                                                   \
  noexcept(noexcept(__VA_ARGS__))->decltype(__VA_ARGS__) {                                         \
    return __VA_ARGS__;                                                                            \
  }

// GCC 13 implements lexical friendship, but it is incomplete. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111018
#if STDEXEC_CLANG() // || (STDEXEC_GCC() && __GNUC__ >= 13)
#  define STDEXEC_FRIENDSHIP_IS_LEXICAL() 1
#else
#  define STDEXEC_FRIENDSHIP_IS_LEXICAL() 0
#endif

#if defined(__cpp_explicit_this_parameter) && (__cpp_explicit_this_parameter >= 202110)
#  define STDEXEC_EXPLICIT_THIS(...) STDEXEC_HEAD_OR_TAIL(1, __VA_ARGS__)
#else
#  define STDEXEC_EXPLICIT_THIS(...) STDEXEC_HEAD_OR_NULL(0, __VA_ARGS__)
#endif

// Configure extra type checking
#define STDEXEC_TYPE_CHECKING_ZERO()                                   0
#define STDEXEC_TYPE_CHECKING_ONE()                                    1
#define STDEXEC_TYPE_CHECKING_TWO()                                    2

#define STDEXEC_PROBE_TYPE_CHECKING_                                   STDEXEC_TYPE_CHECKING_ONE
#define STDEXEC_PROBE_TYPE_CHECKING_0                                  STDEXEC_TYPE_CHECKING_ZERO
#define STDEXEC_PROBE_TYPE_CHECKING_1                                  STDEXEC_TYPE_CHECKING_ONE
#define STDEXEC_PROBE_TYPE_CHECKING_STDEXEC_ENABLE_EXTRA_TYPE_CHECKING STDEXEC_TYPE_CHECKING_TWO

#define STDEXEC_TYPE_CHECKING_WHICH3(...)                              STDEXEC_PROBE_TYPE_CHECKING_##__VA_ARGS__
#define STDEXEC_TYPE_CHECKING_WHICH2(...)                              STDEXEC_TYPE_CHECKING_WHICH3(__VA_ARGS__)
#define STDEXEC_TYPE_CHECKING_WHICH                                    STDEXEC_TYPE_CHECKING_WHICH2(STDEXEC_ENABLE_EXTRA_TYPE_CHECKING)

#ifndef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
#  define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 0
#elif STDEXEC_TYPE_CHECKING_WHICH() == 2
// do nothing
#elif STDEXEC_TYPE_CHECKING_WHICH() == 0
#  undef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
#  define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 0
#else
#  undef STDEXEC_ENABLE_EXTRA_TYPE_CHECKING
#  define STDEXEC_ENABLE_EXTRA_TYPE_CHECKING() 1
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// clang-tidy struggles with the CUDA function annotations
#if STDEXEC_CLANG() && STDEXEC_CUDA() && defined(STDEXEC_CLANG_TIDY_INVOKED)
#  include <cuda_runtime_api.h> // IWYU pragma: keep
#  if !defined(__launch_bounds__)
#    define __launch_bounds__(...)
#  endif

#  if !defined(__host__)
#    define __host__
#  endif

#  if !defined(__device__)
#    define __device__
#  endif

#  if !defined(__global__)
#    define __global__
#  endif
#endif

namespace stdexec {
}
