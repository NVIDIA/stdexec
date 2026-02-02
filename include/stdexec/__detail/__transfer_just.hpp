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
#include "__basic_sender.hpp"
#include "__concepts.hpp"
#include "__continues_on.hpp"
#include "__just.hpp"
#include "__schedulers.hpp"
#include "__tuple.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_GNU("-Wmissing-braces")

namespace STDEXEC {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.transfer_just]
  namespace __transfer_just {
    inline constexpr auto __make_transform_fn() {
      return []<class _Scheduler, __decay_copyable... _Values>(
               _Scheduler&& __sched, _Values&&... __vals) {
        return continues_on(
          just(static_cast<_Values&&>(__vals)...), static_cast<_Scheduler&&>(__sched));
      };
    }

    inline constexpr auto __transform_sender_fn() {
      return []<class _Data>(__ignore, _Data&& __data) {
        return STDEXEC::__apply(__make_transform_fn(), static_cast<_Data&&>(__data));
      };
    }

    struct transfer_just_t {
      template <scheduler _Scheduler, __movable_value... _Values>
      constexpr auto
        operator()(_Scheduler&& __sched, _Values&&... __vals) const -> __well_formed_sender auto {
        return __make_sexpr<transfer_just_t>(
          __tuple{static_cast<_Scheduler&&>(__sched), static_cast<_Values&&>(__vals)...});
      }

      template <class _Sender>
      static auto transform_sender(set_value_t, _Sender&& __sndr, __ignore) {
        if constexpr (__decay_copyable<_Sender>) {
          return __apply(__transform_sender_fn(), static_cast<_Sender&&>(__sndr));
        } else {
          return __not_a_sender<
            _SENDER_TYPE_IS_NOT_DECAY_COPYABLE_,
            _WITH_PRETTY_SENDER_<_Sender>
          >();
        }
      }
    };

    inline constexpr auto __make_attrs_fn() noexcept {
      return []<class _Scheduler>(const _Scheduler& __sched, const auto&...) noexcept {
        static_assert(scheduler<_Scheduler>, "transfer_just requires a scheduler");
        return __sched_attrs{std::cref(__sched)};
      };
    }

    struct __transfer_just_impl : __sexpr_defaults {
      static constexpr auto get_attrs = [](__ignore, const auto& __data) noexcept {
        return STDEXEC::__apply(__make_attrs_fn(), __data);
      };

      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        using __sndr_t =
          __detail::__transform_sender_result_t<transfer_just_t, set_value_t, _Sender, env<>>;
        return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
      };
    };
  } // namespace __transfer_just

  using __transfer_just::transfer_just_t;
  inline constexpr transfer_just_t transfer_just{};

  template <>
  struct __sexpr_impl<transfer_just_t> : __transfer_just::__transfer_just_impl { };
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
