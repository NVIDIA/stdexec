/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
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
#include "__type_traits.hpp"

#define STDEXEC_TAG_INVOKE_DEPRECATED_MSG                                                          \
  "The use of tag_invoke as a means of customization is deprecated. Please use member functions "  \
  "instead."

namespace STDEXEC {
  // [func.tag_invoke], tag_invoke
  namespace __dispatch {
    constexpr void tag_invoke();

    template <class _Tag, class... _Args>
    concept __tag_invocable = requires(_Tag __tag, _Args&&... __args) {
      tag_invoke(static_cast<_Tag&&>(__tag), static_cast<_Args&&>(__args)...);
    };

    template <class _Ret, class _Tag, class... _Args>
    concept __tag_invocable_r = requires(_Tag __tag, _Args&&... __args) {
      {
        static_cast<_Ret>(tag_invoke(static_cast<_Tag&&>(__tag), static_cast<_Args&&>(__args)...))
      };
    };

    template <class _Tag, class... _Args>
    concept __nothrow_tag_invocable =
      __tag_invocable<_Tag, _Args...> && requires(_Tag __tag, _Args&&... __args) {
        { tag_invoke(static_cast<_Tag&&>(__tag), static_cast<_Args&&>(__args)...) } noexcept;
      };

    template <class _Tag, class... _Args>
    using __tag_invoke_result_t = decltype(tag_invoke(__declval<_Tag>(), __declval<_Args>()...));

    template <class _Tag, class... _Args>
    struct __tag_invoke_result { };

    template <class _Tag, class... _Args>
      requires __tag_invocable<_Tag, _Args...>
    struct __tag_invoke_result<_Tag, _Args...> {
      using type = __tag_invoke_result_t<_Tag, _Args...>;
    };

    struct __tag_invoke_t {
      template <class _Tag, class... _Args>
        requires __tag_invocable<_Tag, _Args...>
      [[deprecated(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)]]
      STDEXEC_ATTRIBUTE(always_inline) constexpr auto
        operator()(_Tag __tag, _Args&&... __args) const
        noexcept(__nothrow_tag_invocable<_Tag, _Args...>) -> __tag_invoke_result_t<_Tag, _Args...> {
        return tag_invoke(static_cast<_Tag&&>(__tag), static_cast<_Args&&>(__args)...);
      }
    };

  } // namespace __dispatch

  using __dispatch::__tag_invoke_t;

  namespace __ti {
    inline constexpr __tag_invoke_t __tag_invoke{};

    [[deprecated(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)]]
    inline constexpr __tag_invoke_t tag_invoke = __tag_invoke;
  } // namespace __ti

  using namespace __ti;

  using __dispatch::__tag_invocable;
  using __dispatch::__tag_invocable_r;
  using __dispatch::__nothrow_tag_invocable;
  using __dispatch::__tag_invoke_result_t;
  using __dispatch::__tag_invoke_result;

  // Deprecated interfaces
  template <class _Tag, class... _Args>
  concept tag_invocable STDEXEC_DEPRECATE_CONCEPT(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)
    = __tag_invocable<_Tag, _Args...>;

  template <class _Tag, class... _Args>
  concept nothrow_tag_invocable STDEXEC_DEPRECATE_CONCEPT(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)
    = __nothrow_tag_invocable<_Tag, _Args...>;

  template <class _Tag, class... _Args>
  using tag_invoke_result_t
    [[deprecated(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)]] = __tag_invoke_result_t<_Tag, _Args...>;

  template <class _Tag, class... _Args>
  using tag_invoke_result
    [[deprecated(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)]] = __tag_invoke_result<_Tag, _Args...>;

  template <auto& _Tag>
  using tag_t [[deprecated(STDEXEC_TAG_INVOKE_DEPRECATED_MSG)]]
  = __decay_t<decltype(_Tag)>;
} // namespace STDEXEC
