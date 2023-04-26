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

enum class scope_t {
  free_standing,
  scheduler
};

template <scope_t Scope>
struct cpo_t {
  using is_sender = void;
  constexpr static scope_t scope = Scope;

  using completion_signatures = ex::completion_signatures< //
    ex::set_value_t(),                                     //
    ex::set_error_t(std::exception_ptr),                   //
    ex::set_stopped_t()>;

  friend empty_env tag_invoke(ex::get_env_t, const cpo_t&) noexcept {
    return {};
  }
};

template <class CPO>
struct free_standing_sender_t {
  using is_sender = void;
  using __id = free_standing_sender_t;
  using __t = free_standing_sender_t;
  using completion_signatures = ex::completion_signatures< //
    ex::set_value_t(),                                     //
    ex::set_error_t(std::exception_ptr),                   //
    ex::set_stopped_t()>;

  template <class... Ts>
  friend auto tag_invoke(CPO, const free_standing_sender_t& self, Ts&&...) noexcept {
    return cpo_t<scope_t::free_standing>{};
  }

  friend empty_env tag_invoke(ex::get_env_t, const free_standing_sender_t&) noexcept {
    return {};
  }
};

template <class CPO, class... CompletionSignals>
struct scheduler_t {
  using __id = scheduler_t;
  using __t = scheduler_t;

  struct env_t {
    template <stdexec::__one_of<ex::set_value_t, CompletionSignals...> Tag>
    friend scheduler_t tag_invoke(ex::get_completion_scheduler_t<Tag>, const env_t&) noexcept {
      return {};
    }
  };

  struct sender_t {
    using is_sender = void;
    using __id = sender_t;
    using __t = sender_t;
    using completion_signatures = ex::completion_signatures< //
      ex::set_value_t(),                                     //
      ex::set_error_t(std::exception_ptr),                   //
      ex::set_stopped_t()>;

    friend env_t tag_invoke(ex::get_env_t, const sender_t&) noexcept {
      return {};
    }
  };

  template <class... Ts>
  friend auto tag_invoke(CPO, const scheduler_t&, Ts&&...) noexcept {
    return cpo_t<scope_t::scheduler>{};
  }

  friend sender_t tag_invoke(ex::schedule_t, scheduler_t) {
    return sender_t{};
  }

  friend bool operator==(scheduler_t, scheduler_t) noexcept {
    return true;
  }

  friend bool operator!=(scheduler_t, scheduler_t) noexcept {
    return false;
  }
};
