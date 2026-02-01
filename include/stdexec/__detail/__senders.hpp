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
#include "__connect.hpp"                   // IWYU pragma: export
#include "__get_completion_signatures.hpp" // IWYU pragma: export
#include "__transform_completion_signatures.hpp"

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [exec.snd]
  namespace __detail {
    template <class _Sig>
    extern __undefined<_Sig> __tag_of_sig_v;

    template <class _Tag, class... _Args>
    extern _Tag __tag_of_sig_v<_Tag(_Args...)>;

    template <class _Tag, class... _Args>
    extern _Tag __tag_of_sig_v<_Tag (*)(_Args...)>;

    template <class _Sig>
    using __tag_of_sig_t = decltype(__tag_of_sig_v<_Sig>);
  } // namespace __detail

  template <class _Sender, class _SetSig, class _Env = env<>>
  concept sender_of = sender_in<_Sender, _Env>
                   && __std::same_as<
                        __mlist<_SetSig>,
                        __gather_completions_of_t<
                          __detail::__tag_of_sig_t<_SetSig>,
                          _Sender,
                          _Env,
                          __mcompose<__qq<__mlist>, __qf<__detail::__tag_of_sig_t<_SetSig>>>,
                          __mconcat<__qq<__mlist>>
                        >
                   >;

  template <class _Error>
    requires false
  using __nofail_t = _Error;

  template <class _Sender, class _Env = env<>>
  concept __nofail_sender = sender_in<_Sender, _Env> && requires {
    typename __gather_completion_signatures_t<
      __completion_signatures_of_t<_Sender, _Env>,
      set_error_t,
      __nofail_t,
      __cmplsigs::__default_completion,
      __mlist
    >;
  };

  /////////////////////////////////////////////////////////////////////////////
  // early sender type-checking
  template <class _Sender>
  concept __well_formed_sender = sender_in<_Sender> || dependent_sender<_Sender>;
} // namespace STDEXEC
