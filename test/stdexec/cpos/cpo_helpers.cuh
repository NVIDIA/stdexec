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
    constexpr static scope_t scope = Scope;

    using completion_signatures = ex::completion_signatures< //
      ex::set_value_t(),                                     //
      ex::set_error_t(std::exception_ptr),                   //
      ex::set_stopped_t()>;
  };

  template <class CPO>
  struct free_standing_sender_t {
    using sender_concept = stdexec::sender_t;
    using __id = free_standing_sender_t;
    using __t = free_standing_sender_t;
    using completion_signatures = ex::completion_signatures< //
      ex::set_value_t(),                                     //
      ex::set_error_t(std::exception_ptr),                   //
      ex::set_stopped_t()>;

    template <class... Ts>
    friend auto tag_invoke(CPO, const free_standing_sender_t&, Ts&&...) noexcept {
      return cpo_t<scope_t::free_standing>{};
    }
  };

  template <class CPO, class... CompletionSignals>
  struct scheduler_t {
    using __id = scheduler_t;
    using __t = scheduler_t;

    struct env_t {
      template <stdexec::__one_of<ex::set_value_t, CompletionSignals...> Tag>
      scheduler_t query(ex::get_completion_scheduler_t<Tag>) const noexcept {
        return {};
      }
    };

    struct sender_t {
      using sender_concept = stdexec::sender_t;
      using __id = sender_t;
      using __t = sender_t;
      using completion_signatures = ex::completion_signatures< //
        ex::set_value_t(),                                     //
        ex::set_error_t(std::exception_ptr),                   //
        ex::set_stopped_t()>;

      env_t get_env() const noexcept {
        return {};
      }
    };

    template <class... Ts>
    friend auto tag_invoke(CPO, const scheduler_t&, Ts&&...) noexcept {
      return cpo_t<scope_t::scheduler>{};
    }

    sender_t schedule() const noexcept {
      return sender_t{};
    }

    bool operator==(const scheduler_t&) const noexcept = default;
  };
} // namespace
