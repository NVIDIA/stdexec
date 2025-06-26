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

namespace ex = stdexec;

namespace {

  enum class scope_t {
    free_standing,
    scheduler
  };

  template <scope_t Scope>
  struct cpo_t {
    using sender_concept = stdexec::sender_t;
    static constexpr scope_t scope = Scope;

    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()
    >;
  };

  struct cpo_sender_domain {
    template <class Sender>
    static auto transform_sender(Sender&&) noexcept {
      return cpo_t<scope_t::free_standing>{};
    }
  };

  struct cpo_sender_attrs_t {
    [[nodiscard]]
    auto query(ex::get_domain_t) const noexcept {
      return cpo_sender_domain{};
    }

    [[nodiscard]]
    auto query(ex::get_domain_override_t) const noexcept {
      return cpo_sender_domain{};
    }
  };

  template <class CPO>
  struct cpo_test_sender_t {
    using sender_concept = stdexec::sender_t;
    using __id = cpo_test_sender_t;
    using __t = cpo_test_sender_t;
    using completion_signatures = ex::completion_signatures<
      ex::set_value_t(),
      ex::set_error_t(std::exception_ptr),
      ex::set_stopped_t()
    >;

    auto get_env() const noexcept {
      return cpo_sender_attrs_t{};
    }
  };

  struct cpo_scheduler_domain {
    template <class Sender>
    static auto transform_sender(Sender&&) noexcept {
      return cpo_t<scope_t::scheduler>{};
    }
  };

  template <class CPO, class... CompletionSignals>
  struct cpo_test_scheduler_t {
    using __id = cpo_test_scheduler_t;
    using __t = cpo_test_scheduler_t;

    struct env_t {
      template <stdexec::__one_of<ex::set_value_t, CompletionSignals...> Tag>
      auto query(ex::get_completion_scheduler_t<Tag>) const noexcept -> cpo_test_scheduler_t {
        return {};
      }
    };

    struct sender_t {
      using sender_concept = stdexec::sender_t;
      using __id = sender_t;
      using __t = sender_t;
      using completion_signatures = ex::completion_signatures<
        ex::set_value_t(),
        ex::set_error_t(std::exception_ptr),
        ex::set_stopped_t()
      >;

      auto get_env() const noexcept -> env_t {
        return {};
      }
    };

    auto query(ex::get_domain_t) const noexcept {
      return cpo_scheduler_domain{};
    }

    auto schedule() const noexcept -> sender_t {
      return sender_t{};
    }

    auto operator==(const cpo_test_scheduler_t&) const noexcept -> bool = default;
  };
} // namespace
