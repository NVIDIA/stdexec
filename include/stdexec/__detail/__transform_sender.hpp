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
#include "__domain.hpp"
#include "__env.hpp"
#include "__meta.hpp"
#include "__type_traits.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(type_qualifiers_ignored_on_reference)

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  namespace __detail {
    struct __transform_sender_1 {
      template <class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      static constexpr auto __is_nothrow() noexcept -> bool {
        if constexpr (__detail::__has_transform_sender<_Domain, _Sender, _Env...>) {
          return noexcept(__declval<_Domain&>()
                            .transform_sender(__declval<_Sender>(), __declval<const _Env&>()...));
        } else {
          return noexcept(default_domain()
                            .transform_sender(__declval<_Sender>(), __declval<const _Env&>()...));
        }
      }

      template <class _Domain, class _Sender, class... _Env>
      STDEXEC_ATTRIBUTE(always_inline)
      auto operator()(_Domain __dom, _Sender&& __sndr, const _Env&... __env) const
        noexcept(__is_nothrow<_Domain, _Sender, const _Env&...>()) -> decltype(auto) {
        if constexpr (__detail::__has_transform_sender<_Domain, _Sender, _Env...>) {
          return __dom.transform_sender(static_cast<_Sender&&>(__sndr), __env...);
        } else {
          return default_domain().transform_sender(static_cast<_Sender&&>(__sndr), __env...);
        }
      }
    };

    template <class _Ty, class _Uy>
    concept __decay_same_as = same_as<__decay_t<_Ty>, __decay_t<_Uy>>;

  } // namespace __detail

  struct transform_sender_t {
    template <class _Self = transform_sender_t, class _Domain, class _Sender, class... _Env>
    STDEXEC_ATTRIBUTE(always_inline)
    auto operator()(_Domain __dom, _Sender&& __sndr, const _Env&... __env) const
      noexcept(__nothrow_callable<__detail::__transform_sender_1, _Domain, _Sender, const _Env&...>)
        -> decltype(auto) {
      using _Sender2 = __call_result_t<__detail::__transform_sender_1, _Domain, _Sender, const _Env&...>;
      // If the transformation doesn't change the sender's type, then do not
      // apply the transform recursively.
      if constexpr (__detail::__decay_same_as<_Sender, _Sender2>) {
        return __detail::__transform_sender_1()(__dom, static_cast<_Sender&&>(__sndr), __env...);
      } else {
        // We transformed the sender and got back a different sender. Transform that one too.
        return _Self()(
          __dom,
          __detail::__transform_sender_1()(__dom, static_cast<_Sender&&>(__sndr), __env...),
          __env...);
      }
    }
  };

  /////////////////////////////////////////////////////////////////////////////
  // [execution.transform_sender]
  inline constexpr transform_sender_t transform_sender{};

  struct _CHILD_SENDERS_WITH_DIFFERENT_DOMAINS_ { };

  /////////////////////////////////////////////////////////////////////////////
  template <class _Tag, class _Domain, class _Sender, class... _Args>
  concept __has_implementation_for =
    __detail::__has_apply_sender<_Domain, _Tag, _Sender, _Args...>
    || __detail::__has_apply_sender<default_domain, _Tag, _Sender, _Args...>;

  /////////////////////////////////////////////////////////////////////////////
  // [execution.apply_sender]
  inline constexpr struct apply_sender_t {
    template <class _Domain, class _Tag, class _Sender, class... _Args>
      requires __has_implementation_for<_Tag, _Domain, _Sender, _Args...>
    STDEXEC_ATTRIBUTE(always_inline)
    auto
      operator()(_Domain __dom, _Tag, _Sender&& __sndr, _Args&&... __args) const -> decltype(auto) {
      if constexpr (__detail::__has_apply_sender<_Domain, _Tag, _Sender, _Args...>) {
        return __dom
          .apply_sender(_Tag(), static_cast<_Sender&&>(__sndr), static_cast<_Args&&>(__args)...);
      } else {
        return default_domain()
          .apply_sender(_Tag(), static_cast<_Sender&&>(__sndr), static_cast<_Args&&>(__args)...);
      }
    }
  } apply_sender{};

  template <class _Domain, class _Tag, class _Sender, class... _Args>
  using apply_sender_result_t = __call_result_t<apply_sender_t, _Domain, _Tag, _Sender, _Args...>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Tag = set_value_t>
  concept __completes_on =
    __decays_to<__call_result_t<get_completion_scheduler_t<_Tag>, env_of_t<_Sender>>, _Scheduler>;

  /////////////////////////////////////////////////////////////////////////////
  template <class _Sender, class _Scheduler, class _Env>
  concept __starts_on = __decays_to<__call_result_t<get_scheduler_t, _Env>, _Scheduler>;
} // namespace stdexec

STDEXEC_PRAGMA_POP()
