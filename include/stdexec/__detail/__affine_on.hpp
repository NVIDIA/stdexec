/*
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__sender_concepts.hpp"
#include "__schedulers.hpp"

namespace STDEXEC {
  struct affine_on_t {
    template <sender _Sender, scheduler _Scheduler>
    constexpr auto operator()(_Sender&& __sndr, _Scheduler&&) const -> auto&& {
      // BUGBUG TODO: implement me
      return static_cast<_Sender&&>(__sndr);
    }
  };

  inline constexpr affine_on_t affine_on{};
} // namespace STDEXEC
