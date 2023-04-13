/*
 * Copyright (c) 2023 Maikel Nadolski
 * Copyright (c) 2023 NVIDIA Corporation
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

#include "./repeat.hpp"
#include "./then_each.hpp"

namespace exec {
  struct generate_each_t {
    template <class _Fn>
      requires stdexec::tag_invocable<generate_each_t, _Fn>
    auto operator()(_Fn&& __fn) const
      noexcept(stdexec::nothrow_tag_invocable<generate_each_t, _Fn>)
        -> stdexec::tag_invoke_result_t<generate_each_t, _Fn> {
      return stdexec::tag_invoke(*this, static_cast<_Fn&&>(__fn));
    }

    template <class _Fn>
      requires(!stdexec::tag_invocable<generate_each_t, _Fn>)
    auto operator()(_Fn&& __fn) const {
      return repeat(stdexec::just()) | then_each([__fn = static_cast<_Fn&&>(__fn)]() mutable {
        return std::invoke(static_cast<_Fn&&>(__fn));
      });
    }
  };

  inline constexpr generate_each_t generate_each{};
}