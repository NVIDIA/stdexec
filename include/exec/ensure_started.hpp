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

#include "../stdexec/__detail/__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "../stdexec/__detail/__basic_sender.hpp"
#include "../stdexec/__detail/__sender_adaptor_closure.hpp"
#include "../stdexec/__detail/__senders.hpp"
#include "../stdexec/__detail/__transform_sender.hpp"
#include "detail/shared.hpp"

namespace experimental::execution {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.adaptors.ensure_started]
  namespace __ensure_started {
    using namespace __shared;

    template <class _CvSender>
    concept __is_ensure_started_sender =
      __is_instance_of<__decay_t<_CvSender>, __shared::__sndr>
      && __same_as<typename __decay_t<_CvSender>::__tag_t, ensure_started_t>;
  } // namespace __ensure_started

  struct ensure_started_t {
    template <class _Env = STDEXEC::env<>, STDEXEC::sender_in<_Env> _CvSender>
    [[nodiscard]]
    auto operator()(_CvSender&& __sndr, _Env&& __env = {}) const -> STDEXEC::__well_formed_sender
      auto {
      if constexpr (__ensure_started::__is_ensure_started_sender<_CvSender>) {
        return static_cast<_CvSender&&>(__sndr);
      } else {
        return STDEXEC::transform_sender(
          STDEXEC::__make_sexpr<ensure_started_t>(
            static_cast<_Env&&>(__env), static_cast<_CvSender&&>(__sndr)),
          __env);
      }
    }

    [[nodiscard]]
    constexpr auto operator()() const noexcept {
      return STDEXEC::__closure(*this);
    }

    template <class _CvSender>
    static constexpr auto
      transform_sender(STDEXEC::set_value_t, _CvSender&& __sndr, STDEXEC::__ignore) {
      static_assert(STDEXEC::sender_expr_for<_CvSender, ensure_started_t>);
      auto __result = __shared::__sndr{
        ensure_started_t(),
        STDEXEC::__get<2>(static_cast<_CvSender&&>(__sndr)),
        STDEXEC::__get<1>(static_cast<_CvSender&&>(__sndr))};
      // eagerly start the operation:
      __result.__sh_state_->__try_start();
      return __result;
    }
  };

  inline constexpr ensure_started_t ensure_started{};
} // namespace experimental::execution

namespace exec = experimental::execution;

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::ensure_started_t> : exec::__shared::__impls<exec::ensure_started_t> { };
} // namespace STDEXEC
