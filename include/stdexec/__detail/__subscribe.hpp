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

#include "__config.hpp"

#if STDEXEC_USE_MODULES() && !defined(STDEXEC_IN_MODULE_PURVIEW)

import stdexec;

#else

#  include "__execution_fwd.hpp"

#  include "__concepts.hpp"
#  include "__sender_concepts.hpp"
#  include "__type_traits.hpp"
#  include "__utility.hpp"

#  include "__prologue.hpp"

// NOLINTNEXTLINE(bugprone-std-namespace-modification, modernize-concat-nested-namespaces)
namespace STDEXEC
{
  // Base of exec::sequence_sender_tag, so that senders in stdexec can tell
  // whether a sender is a sequence sender.
  STDEXEC_MODULE_EXPORT_AUTHORING
  struct __sequence_sender_tag : sender_tag
  {};

  template <class _Sender>
  concept __sequence_sender_tagged = requires {
    typename __decay_t<_Sender>::sender_concept;
  } && __std::derived_from<typename __decay_t<_Sender>::sender_concept, __sequence_sender_tag>;

  namespace __subscribe_
  {
    template <class _Sender, class _Receiver>
    concept __has_static_memfn = requires(_Sender && (*__sndr)(), _Receiver && (*__rcvr)()) {
      STDEXEC_REMOVE_REFERENCE(_Sender)::__static_subscribe(__sndr(), __rcvr());
    };

    template <class _Sender, class _Receiver>
    concept __has_memfn = requires(_Sender && (*__sndr)(), _Receiver && (*__rcvr)()) {
      __sndr().subscribe(__rcvr());
    };

    template <class _Sender, class _Receiver>
    consteval auto __nothrow_subscribe() noexcept -> bool
    {
      if constexpr (__has_static_memfn<_Sender, _Receiver>)
      {
        return noexcept(
          __declval<_Sender>().__static_subscribe(__declval<_Sender>(), __declval<_Receiver>()));
      }
      else
      {
        return noexcept(__declval<_Sender>().subscribe(__declval<_Receiver>()));
      }
    }

    // The customization point behind exec::subscribe. A sender hooks it with a
    // static __static_subscribe(sndr, rcvr) member or a subscribe(rcvr) member
    // function, the static one taking precedence. Senders defined in stdexec can
    // use this to behave as sequence senders without depending on exec.
    struct __subscribe_t
    {
      template <class _Sender, class _Receiver>
        requires __has_static_memfn<_Sender, _Receiver> || __has_memfn<_Sender, _Receiver>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(_Sender&& __sndr, _Receiver __rcvr) const
        noexcept(__nothrow_subscribe<_Sender, _Receiver>()) -> decltype(auto)
      {
        if constexpr (__has_static_memfn<_Sender, _Receiver>)
        {
          return __sndr.__static_subscribe(static_cast<_Sender&&>(__sndr),
                                           static_cast<_Receiver&&>(__rcvr));
        }
        else
        {
          return static_cast<_Sender&&>(__sndr).subscribe(static_cast<_Receiver&&>(__rcvr));
        }
      }
    };
  }  // namespace __subscribe_

  STDEXEC_MODULE_EXPORT_AUTHORING
  inline constexpr __subscribe_::__subscribe_t __subscribe{};

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sender, class _Receiver>
  using __subscribe_result_t = __call_result_t<__subscribe_::__subscribe_t, _Sender, _Receiver>;

  STDEXEC_MODULE_EXPORT_AUTHORING
  template <class _Sender, class _Receiver>
  concept __subscribable = __callable<__subscribe_::__subscribe_t, _Sender, _Receiver>;
}  // namespace STDEXEC

#  include "__epilogue.hpp"
#endif  // !STDEXEC_USE_MODULES() || defined(STDEXEC_IN_MODULE_PURVIEW)
