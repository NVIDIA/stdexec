/*
 * Copyright (c) 2024 NVIDIA Corporation
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
#include "__execution_fwd.hpp"

///////////////////////////////////////////////////////////////////////////////
/// To hook a customization point like stdexec::get_env, first bring the names
/// in stdexec::tags into scope:
///
/// @code
/// using namespace stdexec::tags;
/// @endcode
///
/// Then define a member function like this:
///
/// @code
/// STDEXEC_MEMFN_DECL(auto get_env)(this const MySender& self) {
///   return ...;
/// }
/// @endcode
#define STDEXEC_MEMFN_DECL(...)                                                                    \
  friend STDEXEC_EVAL(                                                                             \
    STDEXEC_MEMFN_DECL_TAG_INVOKE, STDEXEC_MEMFN_DECL_WHICH(__VA_ARGS__), __VA_ARGS__)

#define STDEXEC_MEMFN_DECL_WHICH(_A1, ...)                                                         \
  STDEXEC_CAT(STDEXEC_MEMFN_DECL_WHICH_, STDEXEC_FRONT(__VA_OPT__(1, ) 0))(_A1, __VA_ARGS__)
#define STDEXEC_MEMFN_DECL_WHICH_0(_A1, ...)                                                       \
  STDEXEC_CHECK(STDEXEC_MEMFN_DECL_PROBE_##_A1), STDEXEC_CHECK(_A1##_STDEXEC_MEMFN_DECL_PROBE)
#define STDEXEC_MEMFN_DECL_WHICH_1(_A1, ...)                                                       \
  0, STDEXEC_CHECK(STDEXEC_CAT(STDEXEC_BACK(__VA_ARGS__), _STDEXEC_MEMFN_DECL_PROBE))

#define STDEXEC_MEMFN_DECL_TAG_INVOKE(_WHICH, _QUERY, ...)                                         \
  STDEXEC_MEMFN_DECL_RETURN_ ## _WHICH(__VA_ARGS__)                                                \
  tag_invoke(STDEXEC_MEMFN_DECL_TAG_ ## _WHICH ## _QUERY(__VA_ARGS__)

#define STDEXEC_MEMFN_DECL_ARGS(...)                                                               \
  STDEXEC_CAT(STDEXEC_EAT_THIS_, __VA_ARGS__))

#define STDEXEC_MEMFN_DECL_QUERY(_SELF, _TAG, ...)                                                 \
  _TAG, STDEXEC_CAT(STDEXEC_EAT_THIS_, _SELF) __VA_OPT__(, __VA_ARGS__))

#define STDEXEC_EAT_THIS_this
#define STDEXEC_EAT_AUTO_auto
#define STDEXEC_EAT_VOID_void

#define query_STDEXEC_MEMFN_DECL_PROBE   STDEXEC_PROBE(~, 1)
#define STDEXEC_MEMFN_DECL_PROBE_auto    STDEXEC_PROBE(~, 1)
#define STDEXEC_MEMFN_DECL_PROBE_void    STDEXEC_PROBE(~, 2)

#define STDEXEC_MEMFN_DECL_RETURN_0(...) ::stdexec::__arg_type_t<void(__VA_ARGS__())>
#define STDEXEC_MEMFN_DECL_RETURN_1(...) auto
#define STDEXEC_MEMFN_DECL_RETURN_2(...) void

#define STDEXEC_MEMFN_DECL_TAG_00(...)                                                             \
  const ::stdexec::__tag_type_t<__VA_ARGS__##_t::*>&, STDEXEC_MEMFN_DECL_ARGS
#define STDEXEC_MEMFN_DECL_TAG_10(...)                                                             \
  const STDEXEC_EAT_AUTO_##__VA_ARGS__##_t&, STDEXEC_MEMFN_DECL_ARGS
#define STDEXEC_MEMFN_DECL_TAG_20(...)                                                             \
  const STDEXEC_EAT_VOID_##__VA_ARGS__##_t&, STDEXEC_MEMFN_DECL_ARGS
#define STDEXEC_MEMFN_DECL_TAG_01(...) STDEXEC_MEMFN_DECL_QUERY
#define STDEXEC_MEMFN_DECL_TAG_11(...) STDEXEC_MEMFN_DECL_QUERY
#define STDEXEC_MEMFN_DECL_TAG_21(...) STDEXEC_MEMFN_DECL_QUERY

#define STDEXEC_MEMFN_FRIEND(_TAG)     using STDEXEC_CAT(_TAG, _t) = STDEXEC_CAT(stdexec::_TAG, _t)

#if STDEXEC_MSVC()
#  pragma deprecated(STDEXEC_CUSTOM)
#endif

#if STDEXEC_GCC() || (STDEXEC_CLANG() && STDEXEC_CLANG_VERSION < 14'00)
#  define STDEXEC_CUSTOM                                                                           \
    _Pragma("GCC warning \"STDEXEC_CUSTOM is deprecated; use STDEXEC_MEMFN_DECL instead.\"")       \
      STDEXEC_MEMFN_DECL
#else
#  define STDEXEC_CUSTOM STDEXEC_MEMFN_DECL
#endif

#if STDEXEC_CLANG() && STDEXEC_CLANG_VERSION >= 14'00
#  pragma clang deprecated(STDEXEC_CUSTOM, "use STDEXEC_MEMFN_DECL instead.")
#endif

namespace stdexec {
  template <class>
  struct __arg_type;

  template <class _Arg>
  struct __arg_type<void(_Arg (*)())> {
    using type = _Arg;
  };

  template <class _Fn>
  using __arg_type_t = typename __arg_type<_Fn>::type;

  template <class>
  struct __tag_type;

  template <class _Ret, class _Tag>
  struct __tag_type<_Ret _Tag::*> {
    using type = _Tag;
  };

  template <class _Fn>
  using __tag_type_t = typename __tag_type<_Fn>::type;

  namespace tags {
    using stdexec::set_value_t;
    using stdexec::set_error_t;
    using stdexec::set_stopped_t;
    using stdexec::connect_t;
    using stdexec::start_t;
    using stdexec::get_env_t;
    using stdexec::get_completion_signatures_t;
  } // namespace tags
} // namespace stdexec
