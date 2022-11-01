/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__detail/__meta.hpp"
#include "concepts.hpp"

#include <functional>

// A std::declval that doesn't instantiate templates:
#define _DECLVAL(...) \
  ((static_cast<__VA_ARGS__(*)()noexcept>(0))())

namespace stdexec::__std_concepts {
#if STDEXEC_HAS_STD_CONCEPTS_HEADER()
  using std::invocable;
#else
  template<class _F, class... _As>
    concept invocable =
      requires(_F&& __f, _As&&... __as) {
        std::invoke((_F&&) __f, (_As&&) __as...);
      };
#endif
} // stdexec::__std_concepts

namespace std {
  using namespace stdexec::__std_concepts;
}

namespace stdexec {
  template <class _F, class... _As>
    concept __nothrow_invocable =
      invocable<_F, _As...> &&
      requires(_F&& __f, _As&&... __as) {
        { std::invoke((_F&&) __f, (_As&&) __as...) } noexcept;
      };


  template <auto _Fun>
    struct __fun_c_t {
      using _FunT = decltype(_Fun);
      template <class... _Args>
          requires __callable<_FunT, _Args...>
        auto operator()(_Args&&... __args) const
          noexcept(noexcept(_Fun((_Args&&) __args...)))
          -> __call_result_t<_FunT, _Args...> {
          return _Fun((_Args&&) __args...);
        }
    };
  template <auto _Fun>
    inline constexpr __fun_c_t<_Fun> __fun_c {};

  // [func.tag_invoke], tag_invoke
  namespace __tag_invoke {
    void tag_invoke();

    // NOT TO SPEC: Don't require tag_invocable to subsume invocable.
    // std::invoke is more expensive at compile time than necessary,
    // and results in diagnostics that are more verbose than necessary.
    template <class _Tag, class... _Args>
      concept tag_invocable =
        requires (_Tag __tag, _Args&&... __args) {
          tag_invoke((_Tag&&) __tag, (_Args&&) __args...);
        };

    // NOT TO SPEC: nothrow_tag_invocable subsumes tag_invocable
    template<class _Tag, class... _Args>
      concept nothrow_tag_invocable =
        tag_invocable<_Tag, _Args...> &&
        requires (_Tag __tag, _Args&&... __args) {
          { tag_invoke((_Tag&&) __tag, (_Args&&) __args...) } noexcept;
        };

    template<class _Tag, class... _Args>
      using tag_invoke_result_t =
        decltype(tag_invoke(__declval<_Tag>(), __declval<_Args>()...));

    template<class _Tag, class... _Args>
      struct tag_invoke_result {};

    template<class _Tag, class... _Args>
        requires tag_invocable<_Tag, _Args...>
      struct tag_invoke_result<_Tag, _Args...> {
        using type = tag_invoke_result_t<_Tag, _Args...>;
      };

    struct tag_invoke_t {
      template <class _Tag, class... _Args>
          requires tag_invocable<_Tag, _Args...>
        constexpr auto operator()(_Tag __tag, _Args&&... __args) const
          noexcept(nothrow_tag_invocable<_Tag, _Args...>)
          -> tag_invoke_result_t<_Tag, _Args...> {
          return tag_invoke((_Tag&&) __tag, (_Args&&) __args...);
        }
    };
  } // namespace __tag_invoke

  using __tag_invoke::tag_invoke_t;
  inline constexpr tag_invoke_t tag_invoke {};

  template<auto& _Tag>
    using tag_t = decay_t<decltype(_Tag)>;

  using __tag_invoke::tag_invocable;
  using __tag_invoke::nothrow_tag_invocable;
  using __tag_invoke::tag_invoke_result_t;
  using __tag_invoke::tag_invoke_result;
} // namespace stdexec
