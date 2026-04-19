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

#include "__execution_fwd.hpp"

// IWYU pragma: begin_keep
#include "__completion_behavior.hpp"
#include "__completion_signatures.hpp"
#include "__meta.hpp"
#include "__static_vector.hpp"
#include "__typeinfo.hpp"

#include "../functional.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <span>
// IWYU pragma: end_keep

namespace STDEXEC
{
  template <class _Sig>
  constexpr _Sig *__signature = nullptr;

  struct __completion_info
  {
    using __behavior_t = __completion_behavior::__behavior;

    STDEXEC::__disposition __disposition = __invalid_disposition;
    __type_index           __signature   = __mtypeid<void>;
    __type_index           __domain      = __mtypeid<void>;
    __behavior_t           __behavior    = __completion_behavior::__unknown;

    __completion_info() = default;

    template <__completion_tag _Tag, class... _Args>
    constexpr __completion_info(_Tag (*)(_Args...),
                                __type_index __domain   = __mtypeid<void>,
                                __behavior_t __behavior = __completion_behavior::__unknown) noexcept
      : __disposition(_Tag::__disposition)
      , __signature(__mtypeid<_Tag(_Args...)>)
      , __domain(__domain)
      , __behavior(__behavior)
    {}

    template <class _Sender, class... _Env>
    constexpr auto __populate() noexcept -> __completion_info &
    {
      switch (__disposition)
      {
      case __disposition::__value:
        __domain   = __mtypeid<__completion_domain_t<set_value_t, env_of_t<_Sender>, _Env...>>;
        __behavior = STDEXEC::__get_completion_behavior<set_value_t, _Sender, _Env...>();
        break;
      case __disposition::__error:
        __domain   = __mtypeid<__completion_domain_t<set_error_t, env_of_t<_Sender>, _Env...>>;
        __behavior = STDEXEC::__get_completion_behavior<set_error_t, _Sender, _Env...>();
        break;
      case __disposition::__stopped:
        __domain   = __mtypeid<__completion_domain_t<set_stopped_t, env_of_t<_Sender>, _Env...>>;
        __behavior = STDEXEC::__get_completion_behavior<set_stopped_t, _Sender, _Env...>();
        break;
      }
      return *this;
    }

    [[nodiscard]]
    constexpr auto
    operator<=>(__completion_info const &) const noexcept -> std::strong_ordering = default;
  };

  namespace __cmplsigs
  {
    template <auto _GetComplInfo>
    constexpr auto __completion_info_from_v = []() noexcept
    {
      auto __cmpl_info = _GetComplInfo();
      STDEXEC_IF_OK(__cmpl_info)
      {
        constexpr auto __size = _GetComplInfo().size();
        auto           __arr  = __static_vector<__completion_info, __size>();
        std::ranges::sort(__cmpl_info);
        auto const __end = std::ranges::unique_copy(__cmpl_info, __arr.begin()).out;
        __arr.resize(__end - __arr.begin());
        return __arr;
      }
    }();

    template <class _GetComplInfo>
    consteval auto __completion_info_from(_GetComplInfo) noexcept -> auto const &
    {
      return __completion_info_from_v<(_GetComplInfo())>;
    }

    template <auto _GetComplInfo>
    constexpr auto __completion_sigs_from_v = []() noexcept
    {
      constexpr auto __completions = __completion_info_from_v<_GetComplInfo>;
      STDEXEC_IF_OK(__completions)
      {
        auto __signatures = __static_vector<__type_index, __completions.size()>();
        __signatures.resize(__completions.size());
        std::ranges::transform(__completions,
                               __signatures.begin(),
                               &__completion_info::__signature);
        std::ranges::sort(__signatures);
        auto const __end = std::ranges::unique(__signatures).begin();
        __signatures.erase(__end, __signatures.end());
        return __signatures;
      }
    }();

    template <class _GetComplInfo>
    consteval auto __completion_sigs_from(_GetComplInfo) noexcept
    {
      constexpr auto __sigs = __completion_sigs_from_v<(_GetComplInfo())>;
      STDEXEC_IF_OK(__sigs)
      {
        constexpr auto __fn = [=]<std::size_t... _Is>(__indices<_Is...>)
        {
          return completion_signatures<__msplice<__sigs[_Is]>...>();
        };
        return __fn(__make_indices<__sigs.size()>());
      }
    }

    template <class... _Sigs>
    [[nodiscard]]
    consteval auto __to_array(completion_signatures<_Sigs...>) noexcept
    {
      using __array_t = __static_vector<__completion_info, sizeof...(_Sigs)>;
      auto __compls   = __array_t{__completion_info(__signature<_Sigs>)...};
      std::ranges::sort(__compls);
      return __compls;
    }
  }  // namespace __cmplsigs
}  // namespace STDEXEC
