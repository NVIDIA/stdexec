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

#include "__execution_fwd.hpp" // IWYU pragma: keep

#include "__meta.hpp"
#include "__env.hpp"
#include "__receivers.hpp"
#include "__senders.hpp"
#include "__submit.hpp"
#include "__transform_sender.hpp"
#include "__type_traits.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // [execution.senders.consumer.start_detached]
  namespace __start_detached {
    template <class _EnvId>
    struct __detached_receiver {
      using _Env = stdexec::__t<_EnvId>;

      struct __t {
        using receiver_concept = receiver_t;
        using __id = __detached_receiver;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Env __env_;

        template <class... _As>
        void set_value(_As&&...) noexcept {
        }

        template <class _Error>
        [[noreturn]]
        void set_error(_Error&&) noexcept {
          std::terminate();
        }

        void set_stopped() noexcept {
        }

        auto get_env() const noexcept -> const _Env& {
          // BUGBUG NOT TO SPEC
          return __env_;
        }
      };
    };

    template <class _Env = empty_env>
    using __detached_receiver_t = __t<__detached_receiver<__id<__decay_t<_Env>>>>;

    struct start_detached_t {
      template <sender_in<__root_env> _Sender>
        requires __callable<apply_sender_t, __early_domain_of_t<_Sender>, start_detached_t, _Sender>
      void operator()(_Sender&& __sndr) const {
        auto __domain = __get_early_domain(__sndr);
        stdexec::apply_sender(__domain, *this, static_cast<_Sender&&>(__sndr));
      }

      template <class _Env, sender_in<__as_root_env_t<_Env>> _Sender>
        requires __callable<
          apply_sender_t,
          __late_domain_of_t<_Sender, __as_root_env_t<_Env>>,
          start_detached_t,
          _Sender,
          __as_root_env_t<_Env>>
      void operator()(_Sender&& __sndr, _Env&& __env) const {
        auto __domain = __get_late_domain(__sndr, __env);
        stdexec::apply_sender(
          __domain,
          *this,
          static_cast<_Sender&&>(__sndr),
          __as_root_env(static_cast<_Env&&>(__env)));
      }

      using _Sender = __0;
      using __legacy_customizations_t = __types<
        tag_invoke_t(
          start_detached_t,
          get_completion_scheduler_t<set_value_t>(get_env_t(const _Sender&)),
          _Sender),
        tag_invoke_t(start_detached_t, _Sender)>;

      template <class _Sender, class _Env = __root_env>
        requires sender_to<_Sender, __detached_receiver_t<_Env>>
      void apply_sender(_Sender&& __sndr, _Env&& __env = {}) const {
        __submit(
          static_cast<_Sender&&>(__sndr), __detached_receiver_t<_Env>{static_cast<_Env&&>(__env)});
      }
    };
  } // namespace __start_detached

  using __start_detached::start_detached_t;
  inline constexpr start_detached_t start_detached{};
} // namespace stdexec
