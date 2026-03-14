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
#include "__connect.hpp"
#include "__env.hpp"
#include "__get_completion_signatures.hpp"

namespace STDEXEC
{
  // A wrapper around a sender to be used when an adaptor with a sender transform wants to
  // query the transformed sender's attributes without actually transforming the sender.
  template <class _Sender>
  struct __sender_proxy
  {
    using sender_concept = _Sender::sender_concept;

    template <class _Self>
    using __sender_t = __copy_cvref_t<_Self, std::remove_cv_t<_Sender>>;

    template <class _Self, class... _Env>
    static consteval auto get_completion_signatures()  //
      -> __completion_signatures_of_t<__sender_t<_Self>, _Env...>
    {
      return STDEXEC::get_completion_signatures<__sender_t<_Self>, _Env...>();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> env_of_t<_Sender>
    {
      return STDEXEC::get_env(__sndr_);
    }

    _Sender& __sndr_;
  };

  template <class _Sender>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sender_proxy(_Sender&) -> __sender_proxy<_Sender>;

  // A reference wrapper around a multi-shot sender. Useful in adaptors like `repeat_n`
  // where we want to repeatedly connect to the same sender.
  template <class _Sender>
  struct __sender_ref
  {
    using sender_concept = _Sender::sender_concept;

    template <class, class... _Env>
    static consteval auto
    get_completion_signatures() -> __completion_signatures_of_t<_Sender&, _Env...>
    {
      return STDEXEC::get_completion_signatures<_Sender&, _Env...>();
    }

    template <class _Receiver>
    constexpr auto connect(_Receiver __rcvr) const
      noexcept(__nothrow_connectable<_Sender&, _Receiver>) -> connect_result_t<_Sender&, _Receiver>
    {
      return STDEXEC::connect(__sndr_, std::move(__rcvr));
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> env_of_t<_Sender>
    {
      return STDEXEC::get_env(__sndr_);
    }

    _Sender& __sndr_;
  };

  template <class _Sender>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE __sender_ref(_Sender&) -> __sender_ref<_Sender>;
}  // namespace STDEXEC
