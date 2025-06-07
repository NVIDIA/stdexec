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
#include "__diagnostics.hpp"
#include "__meta.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __sigs {
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;

    template <class>
    inline constexpr bool __is_completion_signatures = false;
    template <class... _Sigs>
    inline constexpr bool __is_completion_signatures<completion_signatures<_Sigs...>> = true;
  } // namespace __sigs

  template <class... _Sigs>
  struct completion_signatures { };

  template <class _Completions>
  concept __valid_completion_signatures = __same_as<__ok_t<_Completions>, __msuccess>
                                       && __sigs::__is_completion_signatures<_Completions>;

  template <class _Sender, class... _Env>
  using __unrecognized_sender_error =
    __mexception<_UNRECOGNIZED_SENDER_TYPE_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>...>;
} // namespace stdexec
