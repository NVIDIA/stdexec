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

#include <concepts.hpp>

#include <functional>

namespace std {
  // [func.tag_invoke], tag_invoke
  inline namespace __tag_invoke {
    namespace __impl {
      void tag_invoke();

      template <class _Tag, class... _Args>
      concept __has_tag_invoke =
        requires (_Tag __tag, _Args&&... __args) {
          tag_invoke((_Tag&&) __tag, (_Args&&) __args...);
        };

      struct tag_invoke_t {
        template <class _Tag, class... _Args>
          requires __has_tag_invoke<_Tag, _Args...>
        constexpr decltype(auto) operator()(_Tag __tag, _Args&&... __args) const
          noexcept(noexcept(tag_invoke((_Tag&&) __tag, (_Args&&) __args...))) {
          return tag_invoke((_Tag&&) __tag, (_Args&&) __args...);
        }
      };
    } // namespace __impl

    inline constexpr struct tag_invoke_t : __impl::tag_invoke_t {} tag_invoke {};
  }

  template<auto& _Tag>
  using tag_t = decay_t<decltype(_Tag)>;

  // TODO: Don't require tag_invocable to subsume invocable.
  // std::invoke is more expensive at compile time than necessary.
  template<class _Tag, class... _Args>
  concept tag_invocable =
    invocable<decltype(tag_invoke), _Tag, _Args...>;

  // NOT TO SPEC: nothrow_tag_invocable subsumes tag_invocable
  template<class _Tag, class... _Args>
  concept nothrow_tag_invocable =
    tag_invocable<_Tag, _Args...> &&
    is_nothrow_invocable_v<decltype(tag_invoke), _Tag, _Args...>;

  template<class _Tag, class... _Args>
  using tag_invoke_result = invoke_result<decltype(tag_invoke), _Tag, _Args...>;

  template<class _Tag, class... _Args>
  using tag_invoke_result_t = invoke_result_t<decltype(tag_invoke), _Tag, _Args...>;
}
