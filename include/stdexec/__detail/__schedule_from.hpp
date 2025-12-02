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
#include "__basic_sender.hpp"
#include "__sender_introspection.hpp"
#include "__senders_core.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.schedule_from]
  namespace __schfr {
    struct schedule_from_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender&& __sndr) const {
        return __make_sexpr<schedule_from_t>({}, static_cast<_Sender&&>(__sndr));
      }
    };

    struct __schedule_from_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures =
        []<class _Sender, class... _Env>(_Sender&&, _Env&&...) noexcept
        -> __completion_signatures_of_t<__copy_cvref_t<_Sender, __child_of<_Sender>>, _Env...> {
        return {};
      };
    };
  } // namespace __schfr

  using __schfr::schedule_from_t;
  inline constexpr schedule_from_t schedule_from{};

  template <>
  struct __sexpr_impl<schedule_from_t> : __schfr::__schedule_from_impl { };
} // namespace stdexec
