/*
 * Copyright (c) 2021-2022 NVIDIA Corporation
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

#include "__meta.hpp"

namespace stdexec {
  namespace __detail {
    template <class _Ty>
    extern __q<__midentity> __name_of_v;

    template <class _Ty>
    using __name_of_fn = decltype(__name_of_v<_Ty>);

    template <class _Ty>
    using __name_of = __minvoke<__name_of_fn<_Ty>, _Ty>;
  }

  // A utility for pretty-printing type names in diagnostics
  template <class _Ty>
  using __name_of = __detail::__name_of<_Ty>;

  namespace __errs {
    inline constexpr __mstring __unrecognized_sender_type_diagnostic =
      "The given type cannot be used as a sender with the given environment "
      "because the attempt to compute the completion signatures failed."_mstr;

    template <class _Sender>
    struct _WITH_SENDER_;

    template <class... _Senders>
    struct _WITH_SENDERS_;
  } // namespace __errs

  template <__mstring _Diagnostic = __errs::__unrecognized_sender_type_diagnostic>
  struct _UNRECOGNIZED_SENDER_TYPE_;

  template <class _Sender>
  using _WITH_SENDER_ = __errs::_WITH_SENDER_<__name_of<_Sender>>;

  template <class... _Senders>
  using _WITH_SENDERS_ = __errs::_WITH_SENDERS_<__name_of<_Senders>...>;

  template <class _Env>
  struct _WITH_ENVIRONMENT_;

  template <class _Ty>
  struct _WITH_TYPE_;

  template <class _Receiver>
  struct _WITH_RECEIVER_;

  template <class _Sig>
  struct _MISSING_COMPLETION_SIGNAL_;
}
