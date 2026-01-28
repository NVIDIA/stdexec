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
#include "__sender_adaptor_closure.hpp"
#include "__senders.hpp"
#include "__shared.hpp"
#include "__transform_sender.hpp"

namespace STDEXEC {
  ////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.split]
  namespace __split {
    using namespace __shared;

    struct split_t {
      template <class _Env = env<>, sender_in<_Env> _CvSender>
      [[nodiscard]]
      auto operator()(_CvSender&& __sndr, _Env&& __env = {}) const -> __well_formed_sender auto {
        return STDEXEC::transform_sender(
          __make_sexpr<split_t>(static_cast<_Env&&>(__env), static_cast<_CvSender&&>(__sndr)),
          __env);
      }

      [[nodiscard]]
      constexpr auto operator()() const noexcept {
        return __closure(*this);
      }

      template <class _CvSender>
      static constexpr auto transform_sender(set_value_t, _CvSender&& __sndr, __ignore) {
        static_assert(sender_expr_for<_CvSender, split_t>);
        return __shared::__sndr{
          split_t(),
          STDEXEC::__get<2>(static_cast<_CvSender&&>(__sndr)),
          STDEXEC::__get<1>(static_cast<_CvSender&&>(__sndr))};
      }
    };
  } // namespace __split

  using __split::split_t;
  inline constexpr split_t split{};

  template <>
  struct __sexpr_impl<split_t> : __shared::__impls<split_t> { };
} // namespace STDEXEC
