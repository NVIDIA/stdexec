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

#include "__concepts.hpp"
#include "__meta.hpp"

namespace stdexec::__std_concepts {
#if STDEXEC_HAS_STD_CONCEPTS_HEADER()
  using std::invocable;
#else
  template <class _Fun, class... _As>
  concept invocable = requires(_Fun&& __f, _As&&... __as) {
    std::invoke(static_cast<_Fun &&>(__f), static_cast<_As &&>(__as)...);
  };
#endif
} // namespace stdexec::__std_concepts

namespace std {
  using namespace stdexec::__std_concepts;
} // namespace std

namespace stdexec {
  // [func.tag_invoke], tag_invoke
  namespace __tag_invoke {
    void tag_invoke();

    // For handling queryables with a static constexpr query member function:
    template <class _Tag, class _Env>
      requires true // so this overload is preferred over the one below
    STDEXEC_ATTRIBUTE(always_inline) constexpr auto tag_invoke(_Tag, const _Env&) noexcept
      -> decltype(_Env::query(_Tag())) {
      return _Env::query(_Tag());
    }

    // For handling queryables with a query member function:
    template <class _Tag, class _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    constexpr auto tag_invoke(_Tag, const _Env& __env) noexcept(noexcept(__env.query(_Tag())))
      -> decltype(__env.query(_Tag())) {
      return __env.query(_Tag());
    }

    // NOT TO SPEC: Don't require tag_invocable to subsume invocable.
    // std::invoke is more expensive at compile time than necessary,
    // and results in diagnostics that are more verbose than necessary.
    template <class _Tag, class... _Args>
    concept tag_invocable = requires(_Tag __tag, _Args&&... __args) {
      tag_invoke(static_cast<_Tag &&>(__tag), static_cast<_Args &&>(__args)...);
    };

    template <class _Ret, class _Tag, class... _Args>
    concept __tag_invocable_r = requires(_Tag __tag, _Args&&... __args) {
      {
        static_cast<_Ret>(tag_invoke(static_cast<_Tag &&>(__tag), static_cast<_Args &&>(__args)...))
      };
    };

    // NOT TO SPEC: nothrow_tag_invocable subsumes tag_invocable
    template <class _Tag, class... _Args>
    concept nothrow_tag_invocable =
      tag_invocable<_Tag, _Args...> && requires(_Tag __tag, _Args&&... __args) {
        { tag_invoke(static_cast<_Tag &&>(__tag), static_cast<_Args &&>(__args)...) } noexcept;
      };

    template <class _Tag, class... _Args>
    using tag_invoke_result_t = decltype(tag_invoke(__declval<_Tag>(), __declval<_Args>()...));

    template <class _Tag, class... _Args>
    struct tag_invoke_result { };

    template <class _Tag, class... _Args>
      requires tag_invocable<_Tag, _Args...>
    struct tag_invoke_result<_Tag, _Args...> {
      using type = tag_invoke_result_t<_Tag, _Args...>;
    };

    struct tag_invoke_t {
      template <class _Tag, class... _Args>
        requires tag_invocable<_Tag, _Args...>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Tag __tag, _Args&&... __args) const
        noexcept(nothrow_tag_invocable<_Tag, _Args...>) -> tag_invoke_result_t<_Tag, _Args...> {
        return tag_invoke(static_cast<_Tag&&>(__tag), static_cast<_Args&&>(__args)...);
      }
    };

  } // namespace __tag_invoke

  using __tag_invoke::tag_invoke_t;

  namespace __ti {
    inline constexpr tag_invoke_t tag_invoke{};
  } // namespace __ti

  using namespace __ti;

  template <auto& _Tag>
  using tag_t = __decay_t<decltype(_Tag)>;

  using __tag_invoke::tag_invocable;
  using __tag_invoke::__tag_invocable_r;
  using __tag_invoke::nothrow_tag_invocable;
  using __tag_invoke::tag_invoke_result_t;
  using __tag_invoke::tag_invoke_result;
} // namespace stdexec
