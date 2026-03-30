/*
 * Copyright (c) 2025 NVIDIA Corporation
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

#include "__concepts.hpp"
#include "__config.hpp"
#include "__execution_fwd.hpp"
#include "__operation_states.hpp"
#include "__receivers.hpp"

#include <memory>
#include <utility>

namespace STDEXEC
{
  template <class _RcvrPtr, class _Env = env_of_t<decltype(*_RcvrPtr())>>
  struct __pointer_receiver
  {
    using receiver_concept = receiver_t;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr explicit __pointer_receiver(_RcvrPtr __rcvr_ptr) noexcept
      : __rcvr_ptr_{__rcvr_ptr}
    {}

    template <class... _As>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_value(_As&&... __as) noexcept
    {
      STDEXEC::set_value(std::move(*__rcvr_ptr_), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_error(_Error&& __err) noexcept
    {
      STDEXEC::set_error(std::move(*__rcvr_ptr_), static_cast<_Error&&>(__err));
    }

    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_stopped() noexcept
    {
      STDEXEC::set_stopped(std::move(*__rcvr_ptr_));
    }

    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto get_env() const noexcept -> _Env
    {
      static_assert(__same_as<_Env, env_of_t<decltype(*_RcvrPtr())>>,
                    "get_env() must return the same type as env_of_t<_Rcvr>");
      return STDEXEC::get_env(*__rcvr_ptr_);
    }

   private:
    _RcvrPtr __rcvr_ptr_;
  };

  template <class _Rcvr, class _Env = env_of_t<_Rcvr>>
  struct __receiver_ref : __pointer_receiver<_Rcvr*, _Env>
  {
    STDEXEC_ATTRIBUTE(host, device)
    constexpr __receiver_ref(_Rcvr& __rcvr) noexcept
      : __pointer_receiver<_Rcvr*, _Env>(std::addressof(__rcvr))
    {}
  };
}  // namespace STDEXEC
