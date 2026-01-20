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

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "completion_signatures.hpp"
#include "elide.hpp"
#include "enter_scope_sender.hpp"
#include "storage_for_completion_signatures.hpp"
#include "variant_sender.hpp"
#include "../stdexec/execution.hpp"

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
    return ::STDEXEC::__sexpr{t(),
      ::STDEXEC::__(),
      (Senders&&)senders...};
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

template<typename...>
using remove_set_value_t = ::STDEXEC::completion_signatures<>;

template<typename... Senders>
using exit_sender_t = decltype(
  ::STDEXEC::when_all(
    std::declval<Senders>()...));

template<typename Sender, typename Env>
using exit_sender_storage_t = std::optional<
  ::exec::exit_scope_sender_of_t<Sender, Env>>;

template<typename Storage>
struct store_exit_sender {
  template<typename Sender>
  constexpr void operator()(Sender&& sender) const noexcept {
    storage.emplace((Sender&&)sender);
  }

  Storage& storage;
};

template<typename Env, typename... Senders>
using enter_sender_t = decltype(
  ::STDEXEC::when_all(
    ::STDEXEC::then(
      std::declval<Senders>(),
      std::declval<store_exit_sender<
        exit_sender_storage_t<Senders, Env>>>())...));

template<typename Receiver, typename Indices, typename... Senders>
class state_impl;

template<typename Receiver, std::size_t... Indices, typename... Senders>
class state_impl<Receiver, std::index_sequence<Indices...>, Senders...> {
  using env_type_ = ::STDEXEC::env_of_t<Receiver>;

  template<typename Sender>
  using exit_scope_sender_type_ = ::exec::exit_scope_sender_of_t<
    Sender,
    env_type_>;

  using exit_senders_type_ = std::tuple<
    exit_sender_storage_t<Senders, env_type_>...>;

  struct receiver_base_ {
    using receiver_concept = ::STDEXEC::receiver_t;
    constexpr env_type_ get_env() const noexcept {
      return ::STDEXEC::get_env(self_.r_);
    }
    state_impl& self_;
  };

  struct enter_receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value() && noexcept {
      self_.succeed_();
    }
    template<typename... Args>
    constexpr void set_error(Args&&... args) && noexcept {
      self_.rollback_(::STDEXEC::set_error, (Args&&)args...);
    }
    template<typename... Args>
    constexpr void set_stopped(Args&&... args) && noexcept {
      self_.rollback_(::STDEXEC::set_stopped, (Args&&)args...);
    }
  };

  struct rollback_receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value() && noexcept {
      std::move(self_.storage_).complete(std::move(self_.r_));
    }
  };

  using enter_sender_type_ = enter_sender_t<env_type_, Senders...>;

  using enter_operation_state_type_ = ::STDEXEC::connect_result_t<
    enter_sender_type_,
    enter_receiver_type_>;

  using empty_exit_sender_type_ = decltype(::STDEXEC::just());

  template<typename Sender>
  using rollback_child_sender_type_ = ::exec::variant_sender<
      empty_exit_sender_type_,
      exit_scope_sender_type_<Sender>>;

  using rollback_sender_type_ = decltype(
    ::STDEXEC::when_all(
      std::declval<rollback_child_sender_type_<Senders>>()...));

  using rollback_operation_state_type_ = ::STDEXEC::connect_result_t<
    rollback_sender_type_,
    rollback_receiver_type_>;

  using storage_type_ = ::exec::storage_for_completion_signatures<
    ::STDEXEC::transform_completion_signatures<
      ::STDEXEC::completion_signatures_of_t<
        enter_sender_type_,
        env_type_>,
      ::STDEXEC::completion_signatures<>,
      remove_set_value_t>>;

  static constexpr bool nothrow_make_enter_sender_ = noexcept(
    ::STDEXEC::when_all(
      ::STDEXEC::then(
        std::declval<Senders>(),
        std::declval<store_exit_sender<
          exit_sender_storage_t<Senders, env_type_>>>())...));

  static constexpr bool nothrow_connect_enter_sender_ =
    ::STDEXEC::__nothrow_connectable<
      enter_sender_type_,
      enter_receiver_type_>;

  static constexpr bool nothrow_construct_ =
    std::is_nothrow_constructible_v<Receiver, Receiver> &&
    nothrow_make_enter_sender_ &&
    nothrow_connect_enter_sender_;

  Receiver r_;
  exit_senders_type_ exit_senders_;
  storage_type_ storage_;
  std::variant<
    enter_operation_state_type_,
    rollback_operation_state_type_> operation_;

  constexpr enter_sender_type_ make_enter_sender_(Senders&&... senders)
    noexcept(nothrow_make_enter_sender_)
  {
    return ::STDEXEC::when_all(
      ::STDEXEC::then(
        (Senders&&)senders,
        store_exit_sender<exit_sender_storage_t<Senders, env_type_>>{
          std::get<Indices>(exit_senders_)})...);
  }

  template<std::size_t Index, typename Sender>
  constexpr auto make_rollback_child_() noexcept
    -> rollback_child_sender_type_<Sender>
  {
    auto& sender = std::get<Index>(exit_senders_);
    if (sender) {
      return rollback_child_sender_type_<Sender>(
        std::move(*sender));
    }
    return rollback_child_sender_type_<Sender>(
      ::STDEXEC::just());
  }

  constexpr rollback_sender_type_ make_rollback_sender_() noexcept {
    return ::STDEXEC::when_all(
      make_rollback_child_<Indices, Senders>()...);
  }

  constexpr void succeed_() noexcept {
    ::STDEXEC::set_value(
      std::move(r_),
      ::STDEXEC::when_all(
        std::move(*std::get<Indices>(exit_senders_))...));
  }

  template<typename... Args>
  constexpr void rollback_(Args&&... args) noexcept {
    storage_.arrive((Args&&)args...);
    auto&& operation = operation_.template emplace<
      rollback_operation_state_type_>(
        ::exec::elide([&]() noexcept {
          return ::STDEXEC::connect(
            make_rollback_sender_(),
            rollback_receiver_type_{{*this}});
        }));
    ::STDEXEC::start(operation);
  }

public:
  explicit constexpr state_impl(
    Receiver r,
    Senders&&... senders) noexcept(nothrow_construct_)
    : r_((Receiver&&)r),
      operation_(
        std::in_place_type<enter_operation_state_type_>,
        ::exec::elide([&]() noexcept(
          nothrow_make_enter_sender_ &&
          nothrow_connect_enter_sender_)
        {
          return ::STDEXEC::connect(
            make_enter_sender_((Senders&&)senders...),
            enter_receiver_type_{{*this}});
        }))
  {}

  constexpr void start() & noexcept {
    const auto operation = std::get_if<enter_operation_state_type_>(
      &operation_);
    STDEXEC_ASSERT(operation);
    ::STDEXEC::start(*operation);
  }
};

template<typename Receiver, typename... Senders>
using state = state_impl<
  Receiver,
  std::index_sequence_for<Senders...>,
  Senders...>;

template<typename Env, typename... Children>
consteval auto compute_completion_signatures() {
  auto enter_completions = ::STDEXEC::get_completion_signatures<
    enter_sender_t<Env, Children...>,
    Env>();
  auto stored_input_completions = ::exec::transform_completion_signatures(
    enter_completions,
    ::exec::ignore_completion{},
    ::exec::keep_completion<::STDEXEC::set_error_t>{},
    ::exec::keep_completion<::STDEXEC::set_stopped_t>{});
  using storage = ::exec::storage_for_completion_signatures<
    decltype(stored_input_completions)>;
  auto stored_completions = storage::get_completion_signatures();
  return ::exec::transform_completion_signatures(
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(
        exit_sender_t<::exec::exit_scope_sender_of_t<Children, Env>...>)>{},
    ::exec::keep_completion<::STDEXEC::set_value_t>{},
    ::exec::keep_completion<::STDEXEC::set_error_t>{},
    ::exec::keep_completion<::STDEXEC::set_stopped_t>{},
    stored_completions);
}

template<typename Env, typename Children>
struct compute_completion_signatures_for_children;

template<typename Env, typename... Children>
struct compute_completion_signatures_for_children<
  Env,
  ::STDEXEC::__mlist<Children...>>
{
  static consteval auto get() {
    return enter_scopes::compute_completion_signatures<Env, Children...>();
  }
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
public:
  template<typename Sender, typename... Env>
  static consteval auto __get_completion_signatures() {
    if constexpr (sizeof...(Env) == 0) {
      return STDEXEC::__throw_dependent_sender_error<Sender>();
    } else {
      return compute_completion_signatures_for_children<
        Env...,
        ::STDEXEC::__children_of<Sender>>::get();
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
