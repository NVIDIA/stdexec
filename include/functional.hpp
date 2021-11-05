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

      template <class Tag, class... Args>
      concept __has_tag_invoke =
        requires (Tag tag, Args&&... args) {
          tag_invoke((Tag&&) tag, (Args&&) args...);
        };

      struct tag_invoke_t {
        template <class... Args, __has_tag_invoke<Args...> Tag>
        decltype(auto) operator()(Tag tag, Args&&... args) const
          noexcept(noexcept(tag_invoke((Tag&&) tag, (Args&&) args...))) {
          return tag_invoke((Tag&&) tag, (Args&&) args...);
        }
      };
    }
    inline constexpr struct tag_invoke_t : __impl::tag_invoke_t {} tag_invoke {};
  }

  template<auto& Tag>
  using tag_t = decay_t<decltype(Tag)>;

  // TODO: Don't require tag_invocable to subsume invocable.
  // std::invoke is more expensive at compile time than necessary.
  template<class Tag, class... Args>
  concept tag_invocable =
    invocable<decltype(tag_invoke), Tag, Args...>;

  // NOT TO SPEC: nothrow_tag_invocable subsumes tag_invocable
  template<class Tag, class... Args>
  concept nothrow_tag_invocable =
    tag_invocable<Tag, Args...> &&
    is_nothrow_invocable_v<decltype(tag_invoke), Tag, Args...>;

  template<class Tag, class... Args>
  using tag_invoke_result = invoke_result<decltype(tag_invoke), Tag, Args...>;

  template<class Tag, class... Args>
  using tag_invoke_result_t = invoke_result_t<decltype(tag_invoke), Tag, Args...>;
}
