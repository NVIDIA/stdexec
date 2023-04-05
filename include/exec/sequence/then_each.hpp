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

#include "./transform_each.hpp"

namespace exec {
  struct then_each_t {
    template <class _Sender, class _Fn>
      requires stdexec::tag_invocable<then_each_t, _Sender, _Fn>
    auto operator()(_Sender&& __sender, _Fn&& __fn) const
      noexcept(stdexec::nothrow_tag_invocable<then_each_t, _Sender, _Fn>)
        -> stdexec::tag_invoke_result_t<then_each_t, _Sender, _Fn> {
      return stdexec::tag_invoke(*this, static_cast<_Sender&&>(__sender), static_cast<_Fn&&>(__fn));
    }

    template <class _Sender, class _Fn>
      requires(!stdexec::tag_invocable<then_each_t, _Sender, _Fn>)
    auto operator()(_Sender&& __sender, _Fn&& __fn) const {
      return transform_each(
        static_cast<_Sender&&>(__sender), stdexec::then(static_cast<_Fn&&>(__fn)));
    }

    template <class _Fn>
    auto operator()(_Fn&& __fn) const -> stdexec::__binder_back<then_each_t, _Fn> {
      return {{}, {}, {static_cast<_Fn&&>(__fn)}};
    }
  };

  inline constexpr then_each_t then_each{};
}