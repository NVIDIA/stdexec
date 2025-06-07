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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__sender_adaptor_closure.hpp"
#include "__let.hpp"
#include "__just.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.stopped_as_error]
  namespace __sae {
    struct stopped_as_error_t {
      template <sender _Sender, __movable_value _Error>
      auto operator()(_Sender&& __sndr, _Error __err) const {
        return let_stopped(
          static_cast<_Sender&&>(__sndr),
          [__err2 = static_cast<_Error&&>(__err)]() mutable noexcept(
            __nothrow_move_constructible<_Error>) {
            return just_error(static_cast<_Error&&>(__err2));
          });
      }

      template <__movable_value _Error>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Error __err) const -> __binder_back<stopped_as_error_t, _Error> {
        return {{static_cast<_Error&&>(__err)}, {}, {}};
      }
    };
  } // namespace __sae

  using __sae::stopped_as_error_t;
  inline constexpr stopped_as_error_t stopped_as_error{};
} // namespace stdexec
