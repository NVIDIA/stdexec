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

#if !STDEXEC_NON_LEXICAL_FRIENDSHIP()

#  define STDEXEC_DEFINE_CPO_TAG_INVOKE_COMPATIBILITY(_NAME, _NAMESPACE)                           \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
      requires _NAMESPACE::__has_customized_member<_Ty, _Args...>                                  \
            || _NAMESPACE::__has_customized_static_member<_Ty, _Args...>                           \
               friend auto                                                                         \
               tag_invoke(const STDEXEC_CAT(_NAME, _t) &, _Ty &&__t, _Args &&...__args) noexcept(  \
                 noexcept(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()))               \
                 ->decltype(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()) {            \
      if constexpr (_NAMESPACE::__has_customized_member<_Ty, _Args...>) {                          \
        return ((_Ty &&) __t)._NAME(_NAME, (_Args &&) __args...);                                  \
      } else {                                                                                     \
        return __t._NAME((_Ty &&) __t, _NAME, (_Args &&) __args...);                               \
      }                                                                                            \
    }                                                                                              \
    /**/

#else

// GCC doesn't grant friend status to the hidden friend functions of a
// friend class, even though they are defined lexically within the scope
// of the friend class. As a result, we need some extra indirections to
// evaluate all expressions within the context of the class granted
// friendship.
#  define STDEXEC_DEFINE_CPO_TAG_INVOKE_COMPATIBILITY(_NAME, _NAMESPACE)                           \
    template <class _Ty, class... _Args>                                                           \
    static constexpr bool __has_customized_member =                                                \
      _NAMESPACE::__has_customized_member<_Ty, _Args...>;                                          \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
    static constexpr bool __has_customized_static_member =                                         \
      _NAMESPACE::__has_customized_static_member<_Ty, _Args...>;                                   \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
    static auto __tag_invoke(_Ty &&__t, _Args &&...__args) noexcept(                               \
      noexcept(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()))                          \
      ->decltype(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()) {                       \
      if constexpr (_NAMESPACE::__has_customized_member<_Ty, _Args...>) {                          \
        return ((_Ty &&) __t)._NAME(_NAME, (_Args &&) __args...);                                  \
      } else {                                                                                     \
        return __t._NAME((_Ty &&) __t, _NAME, (_Args &&) __args...);                               \
      }                                                                                            \
    }                                                                                              \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
      requires __has_customized_member<_Ty, _Args...>                                              \
            || __has_customized_static_member<_Ty, _Args...>                                       \
               friend auto                                                                         \
               tag_invoke(const STDEXEC_CAT(_NAME, _t) &, _Ty &&__t, _Args &&...__args) noexcept(  \
                 noexcept(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()))               \
                 ->decltype(__meta<false>(__declval<_Ty>(), __declval<_Args>()...)()) {            \
      return __tag_invoke((_Ty &&) __t, (_Args &&) __args...);                                     \
    }                                                                                              \
    /**/
#endif


#define STDEXEC_CPO_NAMESPACE(_NAME, ...)                                                          \
  STDEXEC_FRONT(__VA_ARGS__ __VA_OPT__(, ) STDEXEC_CAT(__, _NAME))                                 \
  /**/

#define STDEXEC_DEFINE_CPO_STRUCT(_NAME, ...)                                                      \
  namespace STDEXEC_CPO_NAMESPACE(_NAME, __VA_ARGS__) {                                            \
    using namespace ::stdexec;                                                                     \
    struct STDEXEC_CAT(_NAME, _t);                                                                 \
    extern const STDEXEC_CAT(_NAME, _t) _NAME;                                                     \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
    concept __has_customized_member = requires(_Ty &&__t, _Args &&...__args) {                     \
      ((_Ty &&) __t)._NAME(_NAME, (_Args &&) __args...);                                           \
    };                                                                                             \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
    concept __has_customized_static_member = requires(_Ty &&__t, _Args &&...__args) {              \
      __t._NAME((_Ty &&) __t, _NAME, (_Args &&) __args...);                                        \
    };                                                                                             \
                                                                                                   \
    template <class _Ty, class... _Args>                                                           \
    concept __has_customized_tag_invoke = requires(_Ty &&__t, _Args &&...__args) {                 \
      ::stdexec::tag_invoke(_NAME, (_Ty &&) __t, (_Args &&) __args...);                            \
    };                                                                                             \
                                                                                                   \
    namespace __inner {                                                                            \
      struct __base {                                                                              \
        using __accessor = __base;                                                                 \
                                                                                                   \
        template <bool _TryTagInvoke = true, class _Ty, class... _Args>                            \
        static constexpr auto __meta(_Ty &&__t, _Args &&...__args) noexcept {                      \
          if constexpr (__has_customized_member<_Ty, _Args...>) {                                  \
            using _R = decltype(((_Ty &&) __t)._NAME(_NAME, (_Args &&) __args...));                \
            constexpr bool _N = noexcept(((_Ty &&) __t)._NAME(_NAME, (_Args &&) __args...));       \
            return (_R(*)() noexcept(_N)) nullptr;                                                 \
          } else if constexpr (__has_customized_static_member<_Ty, _Args...>) {                    \
            using _R = decltype(__t._NAME((_Ty &&) __t, _NAME, (_Args &&) __args...));             \
            constexpr bool _N = noexcept(__t._NAME((_Ty &&) __t, _NAME, (_Args &&) __args...));    \
            return (_R(*)() noexcept(_N)) nullptr;                                                 \
          } else if constexpr (_TryTagInvoke) {                                                    \
            if constexpr (__has_customized_tag_invoke<_Ty, _Args...>) {                            \
              using _R =                                                                           \
                decltype(::stdexec::tag_invoke(_NAME, (_Ty &&) __t, (_Args &&) __args...));        \
              constexpr bool _N = noexcept(                                                        \
                ::stdexec::tag_invoke(_NAME, (_Ty &&) __t, (_Args &&) __args...));                 \
              return (_R(*)() noexcept(_N)) nullptr;                                               \
            } else {                                                                               \
              return (void (*)() noexcept) nullptr;                                                \
            }                                                                                      \
          } else {                                                                                 \
            return (void (*)() noexcept) nullptr;                                                  \
          }                                                                                        \
        }                                                                                          \
                                                                                                   \
        template <class _Ty, class... _Args>                                                       \
        using __result_t = decltype(__meta(__declval<_Ty>(), __declval<_Args>()...)());            \
                                                                                                   \
        template <class _Ty, class... _Args>                                                       \
        static constexpr bool __noexcept_v =                                                       \
          noexcept(__meta(__declval<_Ty>(), __declval<_Args>()...)());                             \
                                                                                                   \
        struct __tag_invoke_t {                                                                    \
          template <class _Tag, class _Ty, class... _Args>                                         \
            requires __has_customized_member<_Ty, _Args...>                                        \
                  || __has_customized_static_member<_Ty, _Args...>                                 \
                  || __has_customized_tag_invoke<_Ty, _Args...>                                    \
          constexpr auto operator()(const _Tag &__tag, _Ty &&__t, _Args &&...__args) const         \
            noexcept(__noexcept_v<_Ty, _Args...>) -> __result_t<_Ty, _Args...> {                   \
            if constexpr (__has_customized_member<_Ty, _Args...>) {                                \
              return ((_Ty &&) __t)._NAME(_NAME, (_Args &&) __args...);                            \
            } else if constexpr (__has_customized_static_member<_Ty, _Args...>) {                  \
              return __t._NAME((_Ty &&) __t, _NAME, (_Args &&) __args...);                         \
            } else {                                                                               \
              return ::stdexec::tag_invoke(_NAME, (_Ty &&) __t, (_Args &&) __args...);             \
            }                                                                                      \
          }                                                                                        \
        };                                                                                         \
                                                                                                   \
        STDEXEC_DEFINE_CPO_TAG_INVOKE_COMPATIBILITY(                                               \
          _NAME,                                                                                   \
          STDEXEC_CPO_NAMESPACE(_NAME, __VA_ARGS__))                                               \
      };                                                                                           \
    } /* namespace __inner */                                                                      \
                                                                                                   \
    using tag_invoke_t = __inner::__base::__tag_invoke_t;                                          \
    inline constexpr tag_invoke_t tag_invoke{};                                                    \
                                                                                                   \
    template <class _Tag, class _Ty, class... _Args>                                               \
    using tag_invoke_result_t = typename _Tag::template __result_t<_Ty, _Args...>;                 \
                                                                                                   \
    template <class _Tag, class _Ty, class... _Args>                                               \
    concept tag_invocable =                                                                        \
      (__has_customized_member<_Ty, _Args...> /*                                                */ \
       || __has_customized_static_member<_Ty, _Args...> /*                                      */ \
       || __has_customized_tag_invoke<_Ty, _Args...>) &&/*                                      */ \
      ::stdexec::__static_assert_tag_decays_to<_Tag, STDEXEC_CAT(_NAME, _t)>;                      \
                                                                                                   \
    template <class _Tag, class _Ty, class... _Args>                                               \
    concept nothrow_tag_invocable = /*                                                          */ \
      tag_invocable<_Tag, _Ty, _Args...> /*                                                     */ \
      && __nothrow_callable<tag_invoke_t, _Tag, _Ty, _Args...>;                                    \
                                                                                                   \
    namespace stdexec = STDEXEC_CPO_NAMESPACE(_NAME, __VA_ARGS__);                                 \
  } /* namespace _NAMESPACE */                                                                     \
                                                                                                   \
  using STDEXEC_CPO_NAMESPACE(_NAME, __VA_ARGS__)::STDEXEC_CAT(_NAME, _t);                         \
  struct STDEXEC_CPO_NAMESPACE(_NAME, __VA_ARGS__)::STDEXEC_CAT(_NAME, _t)                         \
    : STDEXEC_CPO_NAMESPACE(_NAME, __VA_ARGS__)::__inner::__base /**/

#define STDEXEC_CPO_ACCESS(_TAG)                                                                   \
  friend _TAG;                                                                                     \
  friend ::stdexec::__accessor_of<_TAG> /**/

namespace stdexec {
  template <class _Tag1, class _Tag2>
  auto __custom_tag_decays_to() {
    static_assert(
      ::stdexec::__decays_to<_Tag1, _Tag2>,
      "Within the definition of a customization point object, "
      "stdexec::tag_invocable can only be used with the tag being defined. "
      "Use ::stdexec::tag_invocable instead (with the leading ::).");
  }

  template <class _Tag1, class _Tag2>
  concept __static_assert_tag_decays_to = requires {
    { stdexec::__custom_tag_decays_to<_Tag1, _Tag2>() } -> stdexec::same_as<void>;
  };

  template <class _Tag>
  using __accessor_of = typename _Tag::__accessor;
}
