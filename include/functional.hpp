/*
 * Copyright (c) NVIDIA
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

#include <__utility.hpp>
#include <concepts.hpp>

#include <functional>

// A std::declval that doesn't instantiate templates:
#define _DECLVAL(...) \
  ((static_cast<__VA_ARGS__(*)()noexcept>(0))())

namespace std {
  // [func.tag_invoke], tag_invoke
  inline namespace __tag_invoke {
    namespace __tinvoke_impl {
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

      struct tag_invoke_t {
        template <class _Tag, class... _Args>
            requires tag_invocable<_Tag, _Args...>
          constexpr auto operator()(_Tag __tag, _Args&&... __args) const
            noexcept(nothrow_tag_invocable<_Tag, _Args...>)
            -> tag_invoke_result_t<_Tag, _Args...> {
            return tag_invoke((_Tag&&) __tag, (_Args&&) __args...);
          }
      };
    } // namespace __tinvoke_impl

    inline constexpr struct tag_invoke_t : __tinvoke_impl::tag_invoke_t {} tag_invoke {};
  }

  template<auto& _Tag>
    using tag_t = decay_t<decltype(_Tag)>;

  using __tinvoke_impl::tag_invocable;
  using __tinvoke_impl::nothrow_tag_invocable;
  using __tinvoke_impl::tag_invoke_result_t;

  template<class _Tag, class... _Args>
    struct tag_invoke_result
      : __minvoke<
          __if<
            __bool<tag_invocable<_Tag, _Args...>>,
            __compose<__q1<__x>, __q<tag_invoke_result_t>>,
            __constant<__>>,
          _Tag,
          _Args...>
    {};
}
