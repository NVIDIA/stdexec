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

namespace STDEXEC {
  template <class _Rcvr, class _Env = env_of_t<_Rcvr>>
  struct __rcvr_ref {
    using receiver_concept = receiver_t;

    STDEXEC_ATTRIBUTE(host, device)
    constexpr explicit __rcvr_ref(_Rcvr& __rcvr) noexcept
      : __rcvr_{std::addressof(__rcvr)} {
    }

    template <class... _As>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_value(_As&&... __as) noexcept {
      STDEXEC::set_value(static_cast<_Rcvr&&>(*__rcvr_), static_cast<_As&&>(__as)...);
    }

    template <class _Error>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr void set_error(_Error&& __err) noexcept {
      STDEXEC::set_error(static_cast<_Rcvr&&>(*__rcvr_), static_cast<_Error&&>(__err));
    }

    STDEXEC_ATTRIBUTE(host, device) constexpr void set_stopped() noexcept {
      STDEXEC::set_stopped(static_cast<_Rcvr&&>(*__rcvr_));
    }

    STDEXEC_ATTRIBUTE(nodiscard, host, device)
    constexpr auto get_env() const noexcept -> _Env {
      static_assert(
        __same_as<_Env, env_of_t<_Rcvr>>, "get_env() must return the same type as env_of_t<_Rcvr>");
      return STDEXEC::get_env(*__rcvr_);
    }

   private:
    _Rcvr* __rcvr_;
  };

  namespace __detail {
    template <class _Rcvr, size_t = sizeof(_Rcvr)>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __is_type_complete(int) noexcept {
      return true;
    }

    template <class _Rcvr>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr auto __is_type_complete(long) noexcept {
      return false;
    }
  } // namespace __detail

  // The __ref_rcvr function and its helpers are used to avoid wrapping a receiver in a
  // __rcvr_ref when that is possible. The logic goes as follows:

  // 1. If the receiver is an instance of __rcvr_ref, return it.
  // 2. If the type is incomplete or an operation state, return a __rcvr_ref wrapping the
  //    receiver.
  // 3. If the receiver is nothrow copy constructible, return it.
  // 4. Otherwise, return a __rcvr_ref wrapping the receiver.
  template <class _Env = void, class _Rcvr>
  STDEXEC_ATTRIBUTE(nodiscard, host, device)
  constexpr auto __ref_rcvr(_Rcvr& __rcvr) noexcept {
    if constexpr (__same_as<_Env, void>) {
      return STDEXEC::__ref_rcvr<env_of_t<_Rcvr>>(__rcvr);
    } else if constexpr (__is_instance_of<_Rcvr, __rcvr_ref>) {
      return __rcvr;
    } else if constexpr (!__detail::__is_type_complete<_Rcvr>(0)) {
      return __rcvr_ref<_Rcvr, _Env>{__rcvr};
    } else if constexpr (operation_state<_Rcvr>) {
      return __rcvr_ref<_Rcvr, _Env>{__rcvr};
    } else if constexpr (__nothrow_constructible_from<_Rcvr, const _Rcvr&>) {
      return const_cast<const _Rcvr&>(__rcvr);
    } else {
      return __rcvr_ref{__rcvr};
    }
    STDEXEC_UNREACHABLE();
  }

  template <class _Rcvr, class _Env = env_of_t<_Rcvr>>
  using __rcvr_ref_t = decltype(STDEXEC::__ref_rcvr<_Env>(STDEXEC::__declval<_Rcvr&>()));
} // namespace STDEXEC
