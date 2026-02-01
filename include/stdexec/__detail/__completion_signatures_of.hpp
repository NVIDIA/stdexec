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
#include "__debug.hpp" // IWYU pragma: keep for STDEXEC::__debug_sender
#include "__get_completion_signatures.hpp"
#include "__sender_concepts.hpp" // IWYU pragma: export

namespace STDEXEC {
#if STDEXEC_ENABLE_EXTRA_TYPE_CHECKING()
  // __checked_completion_signatures is for catching logic bugs in a sender's metadata. If sender<S>
  // and sender_in<S, Ctx> are both true, then they had better report the same metadata. This
  // completion signatures wrapper enforces that at compile time.
  template <class _CvSender, class... _Env>
  auto __checked_completion_signatures(_CvSender &&__sndr, _Env &&...__env) noexcept {
    using __completions_t = __completion_signatures_of_t<_CvSender, _Env...>;
    STDEXEC::__debug_sender(static_cast<_CvSender &&>(__sndr), __env...);
    return __completions_t{};
  }

  template <class _CvSender, class... _Env>
    requires sender_in<_CvSender, _Env...>
  using completion_signatures_of_t = decltype(STDEXEC::__checked_completion_signatures(
    __declval<_CvSender>(),
    __declval<_Env>()...));
#else
  template <class _CvSender, class... _Env>
    requires sender_in<_CvSender, _Env...>
  using completion_signatures_of_t = __completion_signatures_of_t<_CvSender, _Env...>;
#endif
} // namespace STDEXEC
