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

// clang-format Language: Cpp

#pragma once

#include <stdexec/execution.hpp>
#include <test_common/type_helpers.hpp>

namespace ex = STDEXEC;

namespace {

  enum class scope_t {
    scheduler
  };

  template <scope_t Scope>
  struct cpo_t {
    using sender_concept = STDEXEC::sender_t;
    static constexpr scope_t scope = Scope;

    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()
    >;

    template <class Receiver>
    struct operation_state_t {
      using sender_t = cpo_t<Scope>;
      using receiver_t = Receiver;

      receiver_t receiver_;

      void start() & noexcept {
        ex::set_value(std::move(receiver_));
      }
    };

    template <ex::receiver Receiver>
    auto connect(Receiver r) const noexcept -> operation_state_t<Receiver> {
      return operation_state_t<Receiver>{r};
    }
  };

  struct cpo_scheduler_domain {
    template <class Sender, class Env>
    static auto transform_sender(STDEXEC::set_value_t, Sender &&, const Env &) noexcept {
      return cpo_t<scope_t::scheduler>{};
    }
  };

  template <class CPO, class... CompletionSignals>
  struct cpo_test_env_t;

  template <class CPO, class... CompletionSignals>
  struct cpo_test_scheduler_t {
    struct sender_t {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = ex::completion_signatures<
        ex::set_value_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_stopped_t()
      >;

      auto get_env() const noexcept -> cpo_test_env_t<CPO, CompletionSignals...> {
        return {};
      }
    };

    template <STDEXEC::__one_of<ex::set_value_t, CompletionSignals...> Tag>
    auto query(ex::get_completion_scheduler_t<Tag>) const noexcept -> cpo_test_scheduler_t {
      return {};
    }

    template <STDEXEC::__one_of<ex::set_value_t, CompletionSignals...> Tag>
    auto query(ex::get_completion_domain_t<Tag>) const noexcept -> cpo_scheduler_domain {
      return {};
    }

    auto schedule() const noexcept -> sender_t {
      return sender_t{};
    }

    auto operator==(const cpo_test_scheduler_t &) const noexcept -> bool = default;
  };

  template <class CPO, class... CompletionSignals>
  struct cpo_test_env_t {
    template <STDEXEC::__one_of<ex::set_value_t, CompletionSignals...> Tag>
    auto query(ex::get_completion_scheduler_t<Tag>) const noexcept
      -> cpo_test_scheduler_t<CPO, CompletionSignals...> {
      return {};
    }

    template <STDEXEC::__one_of<ex::set_value_t, CompletionSignals...> Tag>
    auto query(ex::get_completion_domain_t<Tag>) const noexcept -> cpo_scheduler_domain {
      return {};
    }
  };

} // namespace
