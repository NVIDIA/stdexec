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

#include "../stdexec/__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "../stdexec/__detail/__sender_adaptor_closure.hpp"
#include "../stdexec/__detail/__sender_facade.hpp"
#include "../stdexec/__detail/__senders.hpp"
#include "../stdexec/__detail/__transform_sender.hpp"
#include "detail/shared.hpp"
#include "sender_for.hpp"

namespace experimental::execution
{
  //////////////////////////////////////////////////////////////////////////////
  struct split_t
  {
    // Makes a sender that describes the split sender, which its domain customizes (see
    // __shared::__describe_algo).
    template <class _Env = STDEXEC::env<>, STDEXEC::sender_in<_Env> _CvSender>
    [[nodiscard]]
    auto
    operator()(_CvSender&& __sndr, _Env&& __env = {}) const -> STDEXEC::__well_formed_sender auto
    {
      return STDEXEC::transform_sender(
        STDEXEC::__facade::__sender{split_t(),
                                    __shared::__env_copy(static_cast<_Env&&>(__env)),
                                    static_cast<_CvSender&&>(__sndr)},
        __env);
    }

    [[nodiscard]]
    constexpr auto operator()() const noexcept
    {
      return STDEXEC::__closure(*this);
    }

    // The split sender, unless a domain customizes it
    template <class _CvSender>
    static constexpr auto
    transform_sender(STDEXEC::set_value_t, _CvSender&& __sndr, STDEXEC::__ignore)
    {
      static_assert(sender_for<_CvSender, split_t>);
      return __shared::__make_sender(
        split_t(),
        STDEXEC::__facade::__get_child<0>(static_cast<_CvSender&&>(__sndr)),
        STDEXEC::__facade::__get_data(static_cast<_CvSender&&>(__sndr)));
    }
  };

  inline constexpr split_t split{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  template <>
  struct __facade::__algorithm<exec::split_t> : exec::__shared::__describe_algo<exec::split_t>
  {};
}  // namespace STDEXEC
