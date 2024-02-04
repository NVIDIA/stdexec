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

#define STDEXEC_EAT_THIS_this
#define STDEXEC_EAT_AUTO_auto

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
/// STDEXEC_CUSTOM(auto get_env)(this const MySender& self) {
///   return ...;  
/// }
/// @endcode
#define STDEXEC_CUSTOM(...) \
  friend STDEXEC_TAG_INVOKE(STDEXEC_IS_AUTO(__VA_ARGS__), __VA_ARGS__) STDEXEC_TAG_INVOKE_ARGS /**/

#define STDEXEC_TAG_INVOKE(_ISAUTO, ...) \
    STDEXEC_IIF(_ISAUTO, STDEXEC_RETURN_AUTO, STDEXEC_RETURN_TYPE)(__VA_ARGS__) \
    tag_invoke( \
    STDEXEC_IIF(_ISAUTO, STDEXEC_TAG_AUTO, STDEXEC_TAG_TYPE)(__VA_ARGS__)

#define STDEXEC_PROBE_FN_DECL_auto STDEXEC_PROBE(~)
#define STDEXEC_IS_AUTO(_TY, ...) STDEXEC_CHECK(STDEXEC_CAT(STDEXEC_PROBE_FN_DECL_, _TY))

#define STDEXEC_RETURN_TYPE(...) ::stdexec::__arg_type_t<void(__VA_ARGS__())> /**/
#define STDEXEC_RETURN_AUTO(...) auto                                         /**/

#define STDEXEC_TAG_TYPE(...) ::stdexec::__tag_type_t<STDEXEC_CAT(__VA_ARGS__, _t::*)>
#define STDEXEC_TAG_AUTO(...) STDEXEC_CAT(STDEXEC_CAT(STDEXEC_EAT_AUTO_, __VA_ARGS__), _t)

#define STDEXEC_TAG_INVOKE_ARGS(...) \
    __VA_OPT__(,) STDEXEC_CAT(STDEXEC_EAT_THIS_, __VA_ARGS__))

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
  }
}
