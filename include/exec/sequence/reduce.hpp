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

#include "../sequence_senders.hpp"

#include "./last_value.hpp"
#include "./scan.hpp"

namespace exec {
  struct reduce_t {
    template <class _Sender, class _Ty, class _Fn>
    auto operator()(_Sender&& __sndr, _Ty&& __init, _Fn&& __fn) const {
      return last_value(
        scan(static_cast<_Sender&&>(__sndr), static_cast<_Ty&&>(__init), static_cast<_Fn&&>(__fn)));
    }
  };

  inline constexpr reduce_t reduce{};
}