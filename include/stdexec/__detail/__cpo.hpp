/*
 * Copyright (c) 2021-2023 NVIDIA Corporation
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
#include "../functional.hpp"

#define STDEXEC_DEFINE_CPO(_NAME) \
  \
  struct STDEXEC_CAT(_NAME, _t);\
  extern const STDEXEC_CAT(_NAME, _t) _NAME;\
  \
  template <class _Ty, class... _Args> \
  concept __has_customized_member = \
    requires (_Ty&& __t, _Args&&... __args) { \
      ((_Ty&&) __t)._NAME(_NAME, (_Args&&) __args...); \
    }; \
  \
  template <class _Ty, class... _Args> \
  concept __has_customized_static_member = \
    requires (_Ty&& __t, _Args&&... __args) { \
      __t._NAME((_Ty&&) __t, _NAME, (_Args&&) __args...); \
    }; \
  \
  template <class _Ty, class... _Args> \
  concept __has_customized_tag_invoke = \
    requires (_Ty&& __t, _Args&&... __args) { \
      ::stdexec::tag_invoke(_NAME, (_Ty&&) __t, (_Args&&) __args...); \
    }; \
  \
  template <class _Ty> \
  _Ty&& __declval() noexcept; \
  \
  namespace __stdexec { \
    using namespace ::stdexec; \
    \
    template <class _Tag, class _Ty, class... _Args> \
    concept tag_invocable = (\
      __has_customized_member<_Ty, _Args...> || \
      __has_customized_static_member<_Ty, _Args...> || \
      __has_customized_tag_invoke<_Ty, _Args...>) && \
      ::stdexec::__static_assert_tag_decays_to<_Tag, STDEXEC_CAT(_NAME, _t)>; \
    \
    template <class _Tag, class _Ty, class... _Args> \
    using tag_invoke_result_t = \
      typename _Tag::template __result_t<_Ty, _Args...>; \
    \
    template <class _Tag, class _Ty, class... _Args> \
    concept nothrow_tag_invocable = \
      tag_invocable<_Tag, _Ty, _Args...> && \
      _Tag::template __noexcept_v<_Ty, _Args...>; \
  }\
  \
  namespace __hidden { \
    struct __base { \
      using __accessor = __base; \
      \
      template <bool _TryTagInvoke = true, class _Ty, class... _Args> \
      static constexpr auto __meta(_Ty&& __t, _Args&&... __args) noexcept {\
        if constexpr (__has_customized_member<_Ty, _Args...>) { \
          using _R = decltype(((_Ty&&) __t)._NAME(_NAME, (_Args&&) __args...)); \
          constexpr bool _N = noexcept(((_Ty&&) __t)._NAME(_NAME, (_Args&&) __args...));\
          return (_R(*)()noexcept(_N)) nullptr;\
        } \
        else if constexpr (__has_customized_static_member<_Ty, _Args...>) { \
          using _R = decltype(__t._NAME((_Ty&&) __t, _NAME, (_Args&&) __args...)); \
          constexpr bool _N = noexcept(__t._NAME((_Ty&&) __t, _NAME, (_Args&&) __args...));\
          return (_R(*)()noexcept(_N)) nullptr;\
        } \
        else if constexpr (_TryTagInvoke) { \
          if constexpr (__has_customized_tag_invoke<_Ty, _Args...>) { \
            using _R = decltype(::stdexec::tag_invoke(_NAME, (_Ty&&) __t, (_Args&&) __args...)); \
            constexpr bool _N = noexcept(::stdexec::tag_invoke(_NAME, (_Ty&&) __t, (_Args&&) __args...));\
            return (_R(*)()noexcept(_N)) nullptr;\
          } \
          else { \
            return (void(*)()noexcept) nullptr; \
          }\
        }\
        else { \
          return (void(*)()noexcept) nullptr; \
        }\
      }\
      \
      template <class _Ty, class... _Args> \
      using __result_t = \
        decltype(__meta(__declval<_Ty>(), __declval<_Args>()...)()); \
      \
      template <class _Ty, class... _Args> \
      static constexpr bool __noexcept_v = \
        noexcept(__meta(__declval<_Ty>(), __declval<_Args>()...)()); \
      \
      template <class _Ty, class... _Args> \
        requires __has_customized_member<_Ty, _Args...> || \
                __has_customized_static_member<_Ty, _Args...> \
      friend auto tag_invoke(const STDEXEC_CAT(_NAME, _t)&, _Ty&& __t, _Args&&... __args) \
        noexcept(noexcept(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)())) \
        -> decltype(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()) { \
        if constexpr (__has_customized_member<_Ty, _Args...>) { \
          return ((_Ty&&) __t)._NAME(_NAME, (_Args&&) __args...); \
        } else { \
          return __t._NAME((_Ty&&) __t, _NAME, (_Args&&) __args...); \
        } \
      } \
      \
      struct tag_invoke_t { \
        template <class _Tag, class _Ty, class... _Args> \
          requires \
            __has_customized_member<_Ty, _Args...> || \
            __has_customized_static_member<_Ty, _Args...> || \
            __has_customized_tag_invoke<_Ty, _Args...> \
        constexpr auto operator()(const _Tag& __tag, _Ty&& __t, _Args&&... __args) const \
          noexcept(__noexcept_v<_Ty, _Args...>) -> __result_t<_Ty, _Args...> {\
          if constexpr (__has_customized_member<_Ty, _Args...>) { \
            return ((_Ty&&) __t)._NAME(_NAME, (_Args&&) __args...); \
          } else if constexpr (__has_customized_static_member<_Ty, _Args...>) { \
            return __t._NAME((_Ty&&) __t, _NAME, (_Args&&) __args...); \
          } else { \
            return ::stdexec::tag_invoke(_NAME, (_Ty&&) __t, (_Args&&) __args...); \
          } \
        }\
      }; \
    }; \
  } \
  namespace __stdexec {\
    using tag_invoke_t = __hidden::__base::tag_invoke_t; \
    inline constexpr tag_invoke_t tag_invoke {};\
  }\
  \
  namespace stdexec = __stdexec; \
  using __stdexec::tag_invoke; \
  using __stdexec::tag_invoke_t; \
  using __stdexec::tag_invocable; \
  using __stdexec::nothrow_tag_invocable; \
  using __stdexec::tag_invoke_result_t; \
  \
  struct STDEXEC_CAT(_NAME, _t) : __hidden::__base \
  /**/

namespace stdexec {
  template <class _Tag1, class _Tag2>
  auto __custom_tag_decays_to() {
    static_assert(::stdexec::__decays_to<_Tag1, _Tag2>,
      "Within the definition of a customization point object, "
      "stdexec::tag_invocable can only be used with the tag being defined. "
      "Use ::stdexec::tag_invocable instead (with the leading ::).");
  }

  template <class _Tag1, class _Tag2>
  concept __static_assert_tag_decays_to =
    requires {
      { stdexec::__custom_tag_decays_to<_Tag1, _Tag2>() } -> stdexec::same_as<void>;
    };

  template <class _Tag>
  using __accessor_of = typename _Tag::__accessor;
}
