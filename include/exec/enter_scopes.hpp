/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *                         Copyright (c) 2025 Robert Leahy. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "elide.hpp"
#include "enter_scope_sender.hpp"
#include "storage_for_completion_signatures.hpp"
#include "../stdexec/execution.hpp"

#include <iostream>

namespace experimental::execution {

namespace detail::enter_scopes {

struct t {
  template<::exec::enter_scope_sender... Senders>
  constexpr ::exec::enter_scope_sender auto operator()(Senders&&... senders)
    const noexcept(
      (std::is_nothrow_constructible_v<
        std::remove_cvref_t<Senders>,
        Senders> && ...))
  {
    return ::STDEXEC::__make_sexpr<t>(
      ::STDEXEC::__(),
      (Senders&&)senders...);
  }
  template<::exec::enter_scope_sender Sender>
  constexpr ::exec::enter_scope_sender auto operator()(Sender&& sender) const
    noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<Sender>,
        Sender>)
  {
    return (Sender&&)sender;
  }
  constexpr ::exec::enter_scope_sender auto operator()() const noexcept {
    return ::STDEXEC::just(::STDEXEC::just());
  }
};

template<
  ::exec::enter_scope_sender Sender,
  typename Receiver,
  typename RollbackReceiver>
class enter_scope_sender_state {
  using env_type_ = ::STDEXEC::env_of_t<Receiver>;
  using exit_scope_sender_type_ = ::exec::exit_scope_sender_of_t<
    Sender,
    env_type_>;
  struct receiver_type_ : Receiver {
    //  Note the parameter is by value which means that we can't possibly have a
    //  reference back into the operation state, this is exception safe because
    //  exit senders must be nothrow decay-copyable
    constexpr void set_value(exit_scope_sender_type_ sender) && noexcept {
      auto base = (Receiver&&)*this;
      auto&& self = self_;
      //  This destroys *this
      self.storage_.template emplace<exit_scope_sender_type_>(
        std::move(sender));
      ::STDEXEC::set_value(std::move(base));
    }
    enter_scope_sender_state& self_;
  };
  using operation_state_type_ = ::STDEXEC::connect_result_t<
    Sender,
    receiver_type_>;
  using rollback_operation_state_type_ = ::STDEXEC::connect_result_t<
    exit_scope_sender_type_,
    RollbackReceiver>;
  std::variant<
    operation_state_type_,
    exit_scope_sender_type_,
    rollback_operation_state_type_> storage_;
  static constexpr bool nothrow_constructible_ =
    ::STDEXEC::__nothrow_connectable<Sender, Receiver>;
public:
  constexpr explicit enter_scope_sender_state(Sender&& sender, Receiver r)
    noexcept(nothrow_constructible_)
    : storage_(
        std::in_place_type<operation_state_type_>,
        ::exec::elide([&]() noexcept(nothrow_constructible_) {
          return ::STDEXEC::connect(
            (Sender&&)sender,
            receiver_type_{
              {(Receiver&&)r},
              *this});
        }))
  {}
  constexpr void start() & noexcept {
    const auto ptr = std::get_if<operation_state_type_>(&storage_);
    STDEXEC_ASSERT(ptr);
    ::STDEXEC::start(*ptr);
  }
  [[nodiscard]]
  constexpr bool connect_rollback(RollbackReceiver r) & noexcept {
    const auto ptr = std::get_if<exit_scope_sender_type_>(&storage_);
    if (!ptr) {
      return false;
    }
    auto sender = std::move(*ptr);
    storage_.template emplace<rollback_operation_state_type_>(
      ::exec::elide([&]() noexcept {
        return ::STDEXEC::connect(
          std::move(sender),
          std::move(r));
      }));
    return true;
  }
  constexpr void start_rollback() & noexcept {
    if (
      const auto ptr = std::get_if<rollback_operation_state_type_>(&storage_);
      ptr)
    {
      ::STDEXEC::start(*ptr);
    }
  }
  constexpr exit_scope_sender_type_&& complete() && noexcept {
    const auto ptr = std::get_if<exit_scope_sender_type_>(&storage_);
    STDEXEC_ASSERT(ptr);
    return std::move(*ptr);
  }
};

template<typename...>
using remove_set_value_t = ::STDEXEC::completion_signatures<>;

template<typename... Senders>
using exit_sender_t = decltype(
  ::STDEXEC::when_all(
    std::declval<Senders>()...));

template<typename... Env>
struct storage_for_completion_signatures {
  template<typename... Senders>
  using __f = ::exec::storage_for_completion_signatures<
    ::STDEXEC::transform_completion_signatures<
      ::STDEXEC::__concat_completion_signatures_t<
        ::STDEXEC::completion_signatures_of_t<
          Senders,
          Env...>...>,
      ::STDEXEC::completion_signatures<>,
      remove_set_value_t>>;
};

template<typename Receiver, typename... Senders>
class state {
  using env_type_ = ::STDEXEC::env_of_t<Receiver>;
  struct receiver_base_ {
    using receiver_concept = ::STDEXEC::receiver_t;
    constexpr env_type_ get_env() const noexcept;
    state& self_;
  };
  struct receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value() && noexcept;
    template<typename... Args>
    constexpr void set_error(Args&&...) && noexcept;
    template<typename... Args>
    constexpr void set_stopped(Args&&...) && noexcept;
  };
  struct rollback_receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value() && noexcept;
  };
  template<typename S>
  using state_type_ = enter_scope_sender_state<
    S,
    receiver_type_,
    rollback_receiver_type_>;
  Receiver r_;
  std::tuple<state_type_<Senders>...> states_;
  //  TODO: There's no need for these two members if no child can fail or stop
  typename storage_for_completion_signatures<
    ::STDEXEC::env_of_t<Receiver>>::template __f<Senders...> storage_;
  std::atomic<bool> stored_{false};
  std::atomic<std::size_t> outstanding_{sizeof...(Senders)};
  constexpr bool is_complete_() noexcept {
    return outstanding_.fetch_sub(1, std::memory_order_acq_rel) == 1;
  }
  constexpr void complete_() noexcept {
    if (!is_complete_()) {
      return;
    }
    std::apply(
      [&](auto&... states) noexcept {
        if (stored_.load(std::memory_order_relaxed)) {
          //  Starting with one prevents the child operations from finalizing
          //  the operation out from under us, thereby preventing UB (it does
          //  require that we check to see if we have to finalize as we
          //  complete, see below)
          outstanding_.store(1, std::memory_order_relaxed);
          const auto impl = [&](auto& state) noexcept {
            if (state.connect_rollback(rollback_receiver_type_{{*this}})) {
              outstanding_.fetch_add(1, std::memory_order_relaxed);
            }
          };
          (impl(states), ...);
          (states.start_rollback(), ...);
          rollback_complete_();
        } else {
          ::STDEXEC::set_value(
            std::move(r_),
            ::STDEXEC::when_all(std::move(states).complete()...));
        }
      },
      states_);
  }
  template<typename... Args>
  constexpr void fail_(Args&&... args) noexcept {
    if (!stored_.exchange(true, std::memory_order_relaxed)) {
      storage_.arrive((Args&&)args...);
    }
    complete_();
  }
  constexpr void rollback_complete_() noexcept {
    if (is_complete_()) {
      std::move(storage_).complete(std::move(r_));
    }
  }
public:
  explicit constexpr state(
    Receiver r,
    Senders&&... senders) noexcept(
      (std::is_nothrow_constructible_v<
        state_type_<Senders>,
        Senders,
        receiver_type_> && ...))
    : r_((Receiver&&)r),
      states_(
        ::exec::elide([&]() noexcept(
          std::is_nothrow_constructible_v<
            state_type_<Senders>,
            Senders,
            receiver_type_>)
        {
          return state_type_<Senders>(
            (Senders&&)senders,
            receiver_type_{{*this}});
        })...)
  {}
  constexpr void start() & noexcept {
    std::apply(
      [](auto&&... states) noexcept {
        (states.start(), ...);
      },
      states_);
  }
};

template<typename Receiver, typename... Senders>
constexpr auto state<Receiver, Senders...>::receiver_base_::get_env() const
  noexcept -> env_type_
{
  return ::STDEXEC::get_env(self_.r_);
}

template<typename Receiver, typename... Senders>
constexpr void state<Receiver, Senders...>::receiver_type_::set_value() &&
  noexcept
{
  self_.complete_();
}

template<typename Receiver, typename... Senders>
template<typename... Args>
constexpr void state<Receiver, Senders...>::receiver_type_::set_error(
  Args&&... args) && noexcept
{
  self_.fail_(::STDEXEC::set_error, (Args&&)args...);
}

template<typename Receiver, typename... Senders>
template<typename... Args>
constexpr void state<Receiver, Senders...>::receiver_type_::set_stopped(
  Args&&... args) && noexcept
{
  self_.fail_(::STDEXEC::set_stopped, (Args&&)args...);
}

template<typename Receiver, typename... Senders>
constexpr void state<Receiver, Senders...>::rollback_receiver_type_::set_value()
  && noexcept
{
  self_.rollback_complete_();
}

template<typename... Env>
struct completions;

template<typename Env>
struct completions<Env> {
  template<typename... Children>
  using __f = ::STDEXEC::transform_completion_signatures<
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(
        exit_sender_t<::exec::exit_scope_sender_of_t<Children, Env>...>)>,
    typename storage_for_completion_signatures<Env>::
      template __f<Children...>::completion_signatures>;
};

//  TODO: We should be able to get a better "message" than this
struct FAILED_TO_FORM_COMPLETION_SIGNATURES {};

template<typename Receiver>
struct connect_result {
  template<typename... Children>
  using __f = state<Receiver, Children...>;
};

template<typename Receiver>
struct nothrow_connect {
  template<typename... Children>
  using __f = std::is_nothrow_constructible<
    state<Receiver, Children...>,
    Receiver,
    Children...>;
};

class impl : public ::STDEXEC::__sexpr_defaults {
  template<typename Sender, typename... Env>
  using completions_ = ::STDEXEC::__children_of<Sender, completions<Env...>>;
public:
  template<typename Sender, typename... Env>
  static consteval auto __get_completion_signatures() {
    if constexpr (::STDEXEC::__minvocable_q<completions_, Sender, Env...>) {
      return completions_<Sender, Env...>{};
    } else if constexpr (sizeof...(Env) == 0) {
      return STDEXEC::__dependent_sender<Sender>();
    } else {
      return ::STDEXEC::__throw_compile_time_error<
        FAILED_TO_FORM_COMPLETION_SIGNATURES,
        ::STDEXEC::_WITH_PRETTY_SENDER_<Sender>>();
    }
  }
  static constexpr auto __connect = []<typename Sender, typename Receiver>(
    Sender&& sender, Receiver r) noexcept(
      ::STDEXEC::__children_of<Sender, nothrow_connect<Receiver>>::value)
        -> ::STDEXEC::__children_of<Sender, connect_result<Receiver>>
  {
    return ::STDEXEC::__apply(
      [&](auto&&, auto&&, auto&&... children) noexcept(
        ::STDEXEC::__children_of<Sender, nothrow_connect<Receiver>>::value)
      {
        return ::STDEXEC::__children_of<Sender, connect_result<Receiver>>(
          (Receiver&&)r,
          (decltype(children)&&)children...);
      },
      (Sender&&)sender);
  };
};

}

using enter_scopes_t = detail::enter_scopes::t;
inline constexpr enter_scopes_t enter_scopes;

}  // namespace exec

namespace STDEXEC {

template<>
struct __sexpr_impl<::exec::enter_scopes_t> : ::exec::detail::enter_scopes::impl
{};

}
