/*
 * Copyright (c) 2021-2025 NVIDIA Corporation
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

#include "__get_completion_signatures.hpp"
#include "__meta.hpp"
#include "__sender_concepts.hpp"
#include "__tuple.hpp"
#include "__variant.hpp"

#include "__prologue.hpp"

namespace STDEXEC
{
  namespace __detail
  {
    struct __applier
    {
      template <class _Receiver, class _Tag, class... _Args>
      constexpr void operator()(_Receiver& __rcvr, _Tag, _Args&&... __args) noexcept
      {
        _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
      }
    };

    struct __visitor
    {
      template <class _Receiver, class _Tuple>
      constexpr void operator()(_Receiver& __rcvr, _Tuple&& __tupl) noexcept
      {
        __apply(__applier(), static_cast<_Tuple&&>(__tupl), __rcvr);
      }
    };

    // Add storage for an exception_ptr if the result datums are not all nothrow
    // decay-copyable.
    template <class _Tag, class... _Args>
    using __tuples_for_t =
      __if<__nothrow_decay_copyable_t<_Args...>,
           __mlist<__decayed_tuple<_Tag, _Args...>>,
           __mlist<__decayed_tuple<_Tag, _Args...>, __tuple<set_error_t, std::exception_ptr>>>;
  }  // namespace __detail

  // A variant type that is capable of storing the result datums of the specified
  // completion signatures.
  template <class... _Signatures>
  struct __results_storage
    : __mcall<__mconcat<__qq<__uniqued_variant>>,
              __mapply_q<__detail::__tuples_for_t, _Signatures>...>
  {
    constexpr __results_storage() noexcept
      : __results_storage::__variant(__no_init)
    {}

    template <class _Self, class _Receiver>
    STDEXEC_ATTRIBUTE(host, device)
    constexpr STDEXEC_EXPLICIT_THIS_BEGIN(void __complete)(this _Self&& __self,
                                                           _Receiver&   __rcvr) noexcept
    {
      __visit(__detail::__visitor(), static_cast<_Self&&>(__self), __rcvr);
    }
    STDEXEC_EXPLICIT_THIS_END(__complete)
  };

  template <class _CvSender, class _Env>
    requires sender_in<_CvSender, _Env>
  using __storage_for_t =
    __mapply_q<__results_storage, __completion_signatures_of_t<_CvSender, _Env>>;
}  // namespace STDEXEC

#include "__epilogue.hpp"
