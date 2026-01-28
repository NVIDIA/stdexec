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

#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "elide.hpp"
#include "enter_scope_sender.hpp"
#include "like_t.hpp"
#include "storage_for_completion_signatures.hpp"
#include "../stdexec/execution.hpp"

namespace exec {

namespace detail::within {

struct tag {};

struct t {
  template<::exec::enter_scope_sender Scope, ::STDEXEC::sender Sender>
  constexpr ::STDEXEC::sender auto operator()(Scope&& scope, Sender&& sender)
    const noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<Scope>,
        Scope> &&
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<Sender>,
        Sender>)
  {
    return ::STDEXEC::__make_sexpr<t>(
      ::STDEXEC::__(),
      (Scope&&)scope,
      (Sender&&)sender);
  }
  template<typename Tag, typename Sender, typename Env>
  static constexpr auto transform_sender(
    Tag,
    Sender&& sender,
    const Env&) noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<Sender>,
        Sender>)
  {
    auto&& [_, data, scope, inner] = (Sender&&)sender;
    (void)data;
    return ::STDEXEC::__make_sexpr<tag>(
      //  TODO: Transform the children?
      std::tuple((decltype(scope)&&)scope, (decltype(inner)&&)inner));
  }
};

template<typename...>
using remove_set_value_t = ::STDEXEC::completion_signatures<>;

template<typename Sender, typename Env>
inline constexpr bool nothrow_connect =
  ::STDEXEC::__nothrow_connectable<
    Sender,
    ::STDEXEC::__receiver_archetype<Env>> &&
  //  Because we move the sender out of storage so we can reuse it
  std::is_nothrow_move_constructible_v<Sender>;

template<typename Sender, typename Env>
using completion_signatures_of_sender_t =
  ::STDEXEC::transform_completion_signatures<
    ::STDEXEC::completion_signatures_of_t<Sender, Env>,
    std::conditional_t<
      nothrow_connect<Sender, Env>,
      ::STDEXEC::completion_signatures<>,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_error_t(std::exception_ptr)>>>;

template<typename Scope, typename Env>
using completion_signatures_of_scope_t =
  ::STDEXEC::transform_completion_signatures<
    ::STDEXEC::completion_signatures_of_t<Scope, Env>,
    ::STDEXEC::completion_signatures<>,
    remove_set_value_t>;

template<typename Sender, typename Env>
using storage_for_completion_signatures_t =
  ::exec::storage_for_completion_signatures<
    completion_signatures_of_sender_t<Sender, Env>>;

template<typename, typename, typename>
struct completions;
template<typename Self, typename Scope, typename Sender, typename Env>
struct completions<Self, std::tuple<Scope, Sender>, Env> {
  using type = ::STDEXEC::transform_completion_signatures<
    typename storage_for_completion_signatures_t<
      Sender,
      Env>::completion_signatures,
    completion_signatures_of_scope_t<
      //  Note the use of like_t for the scope but not the inner sender, this is
      //  because we're going to decay-copy the inner sender whereas we connect
      //  the scope with the value category of the incoming sender
      ::exec::like_t<Self, Scope>,
      Env>>;
};

template<typename Scope, typename Sender, typename Receiver>
class state {
  using env_type_ = ::STDEXEC::env_of_t<Receiver>;
  struct receiver_base_ {
    using receiver_concept = ::STDEXEC::receiver_t;
    constexpr env_type_ get_env() const noexcept;
    state& self_;
  };
  using exit_scope_sender_type_ =
    ::exec::exit_scope_sender_of_t<Scope, env_type_>;
  struct enter_receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value(exit_scope_sender_type_) && noexcept;
    template<typename... Args>
    constexpr void set_error(Args&&...) && noexcept;
    template<typename... Args>
    constexpr void set_stopped(Args&&...) && noexcept;
  };
  using enter_operation_state_type_ = ::STDEXEC::connect_result_t<
    Scope,
    enter_receiver_type_>;
  struct exit_receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value() && noexcept;
  };
  using exit_operation_state_type_ = ::STDEXEC::connect_result_t<
    exit_scope_sender_type_,
    exit_receiver_type_>;
  struct receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    template<typename... Args>
    constexpr void set_value(Args&&...) && noexcept;
    template<typename... Args>
    constexpr void set_error(Args&&...) && noexcept;
    template<typename... Args>
    constexpr void set_stopped(Args&&...) && noexcept;
  };
  using operation_state_type_ = ::STDEXEC::connect_result_t<
    Sender,
    receiver_type_>;
  Receiver r_;
  storage_for_completion_signatures_t<Sender, env_type_> storage_;
  struct enter_state_type_ {
    enter_operation_state_type_ op;
    Sender s;
  };
  struct state_type_ {
    operation_state_type_ op;
    exit_scope_sender_type_ s;
  };
  std::variant<
    enter_state_type_,
    state_type_,
    exit_operation_state_type_> state_;
  constexpr void exit_() noexcept {
    const auto ptr = std::get_if<state_type_>(&state_);
    auto sender = std::move(ptr->s);
    auto&& op = state_.template emplace<exit_operation_state_type_>(
      ::exec::elide([&]() noexcept {
        return ::STDEXEC::connect(
          std::move(sender),
          exit_receiver_type_{{*this}});
      }));
    ::STDEXEC::start(op);
  }
  template<typename... Args>
  constexpr void complete_(Args&&... args) noexcept {
    storage_.arrive((Args&&)args...);
    exit_();
  }
public:
  template<typename S>
  explicit constexpr state(
    Scope&& scope,
    S&& sender,
    Receiver r) // noexcept
    : r_((Receiver&&)r),
      state_(
        std::in_place_type<enter_state_type_>,
        ::exec::elide([&]() /* noexcept */ {
          return enter_state_type_{
            ::STDEXEC::connect(
              (Scope&&)scope,
              enter_receiver_type_{{*this}}),
            (S&&)sender};
        }))
  {
  }
  constexpr void start() & noexcept {
    const auto ptr = std::get_if<enter_state_type_>(&state_);
    STDEXEC_ASSERT(ptr);
    ::STDEXEC::start(ptr->op);
  }
};

template<typename Scope, typename Sender, typename Receiver>
constexpr auto state<Scope, Sender, Receiver>::receiver_base_::get_env() const
  noexcept -> env_type_
{
  return ::STDEXEC::get_env(self_.r_);
}

template<typename Scope, typename Sender, typename Receiver>
constexpr void state<Scope, Sender, Receiver>::enter_receiver_type_::set_value(
  exit_scope_sender_type_ exit) && noexcept
{
  //  Because we're going to destroy the operation state and therefore
  //  transitively *this
  auto&& self = self_;
  constexpr auto nothrow = nothrow_connect<Sender, env_type_>;
  const auto ptr = std::get_if<enter_state_type_>(&self.state_);
  STDEXEC_ASSERT(ptr);
  try {
    auto sender = (Sender&&)ptr->s;
    auto&& state = self.state_.template emplace<state_type_>(
      ::exec::elide([&]() noexcept(nothrow) {
        return state_type_{
          ::STDEXEC::connect(
            (Sender&&)sender,
            receiver_type_{{self}}),
          std::move(exit)};
        }));
    ::STDEXEC::start(state.op);
  } catch (...) {
    if constexpr (nothrow) {
      STDEXEC_UNREACHABLE();
    } else {
      ::STDEXEC::set_error((Receiver&&)self.r_, std::current_exception());
    }
  }
}

template<typename Scope, typename Sender, typename Receiver>
template<typename... Args>
constexpr void state<Scope, Sender, Receiver>::enter_receiver_type_::set_error(
  Args&&... args) && noexcept
{
  ::STDEXEC::set_error((Receiver&&)self_.r_, (Args&&)args...);
}

template<typename Scope, typename Sender, typename Receiver>
template<typename... Args>
constexpr void state<Scope, Sender, Receiver>::enter_receiver_type_::
  set_stopped(Args&&... args) && noexcept
{
  ::STDEXEC::set_stopped((Receiver&&)self_.r_, (Args&&)args...);
}

template<typename Scope, typename Sender, typename Receiver>
constexpr void state<Scope, Sender, Receiver>::exit_receiver_type_::set_value()
  && noexcept
{
  std::move(self_.storage_).complete((Receiver&&)self_.r_);
}

template<typename Scope, typename Sender, typename Receiver>
template<typename... Args>
constexpr void state<Scope, Sender, Receiver>::receiver_type_::set_value(
  Args&&... args) && noexcept
{
  self_.complete_(::STDEXEC::set_value, (Args&&)args...);
}

template<typename Scope, typename Sender, typename Receiver>
template<typename... Args>
constexpr void state<Scope, Sender, Receiver>::receiver_type_::set_error(
  Args&&... args) && noexcept
{
  self_.complete_(::STDEXEC::set_error, (Args&&)args...);
}

template<typename Scope, typename Sender, typename Receiver>
template<typename... Args>
constexpr void state<Scope, Sender, Receiver>::receiver_type_::set_stopped(
  Args&&... args) && noexcept
{
  self_.complete_(::STDEXEC::set_stopped, (Args&&)args...);
}

//  TODO: We should be able to get a better "message" than this
struct FAILED_TO_FORM_COMPLETION_SIGNATURES {};

template<typename Tuple, typename, typename = std::remove_cvref_t<Tuple>>
struct get_state_result;
template<typename Tuple, typename Receiver, typename Scope, typename Sender>
struct get_state_result<Tuple, Receiver, std::tuple<Scope, Sender>> {
  using type = state<
    ::exec::like_t<Tuple, Scope>,
    Sender,
    Receiver>;
};

template<typename Tuple, typename, typename = std::remove_cvref_t<Tuple>>
struct nothrow_get_state;
template<typename Tuple, typename Receiver, typename Scope, typename Sender>
struct nothrow_get_state<Tuple, Receiver, std::tuple<Scope, Sender>> {
  static constexpr bool value = std::is_nothrow_constructible_v<
    typename get_state_result<Tuple, Receiver>::type,
    ::exec::like_t<Tuple, Scope>,
    ::exec::like_t<Tuple, Sender>,
    Receiver>;
};

class impl : public ::STDEXEC::__sexpr_defaults {
  template<typename Self, typename Tuple, typename... Env>
  using completions_ = completions<Self, Tuple, Env...>::type;
public:
  template<typename Self, typename... Env>
  static consteval auto get_completion_signatures() {
    using tuple = std::remove_cvref_t<
      ::STDEXEC::__data_of<Self>>;
    if constexpr (sizeof...(Env) == 0) {
      return ::STDEXEC::__dependent_sender_error_t<Self>();
    } else if constexpr (::STDEXEC::__mvalid<completions_, Self, tuple, Env...>)
    {
      return completions_<Self, tuple, Env...>{};
    } else {
      return ::STDEXEC::__throw_compile_time_error<
        FAILED_TO_FORM_COMPLETION_SIGNATURES,
        ::STDEXEC::_WITH_PRETTY_SENDER_<Self>>();
    }
  }
  static constexpr auto get_state = []<typename Sender, typename Receiver>(
    Sender&& sender, Receiver r) noexcept(
      nothrow_get_state<::STDEXEC::__data_of<Sender>, Receiver>::value)
        -> get_state_result<::STDEXEC::__data_of<Sender>, Receiver>::type
  {
    auto&& [_, tuple] = (Sender&&)sender;
    auto&& [scope, inner] = (decltype(tuple)&&)tuple;
    //  See note on completion signature computation for why we remove_cvref_t
    //  here
    using inner_type = std::remove_cvref_t<decltype(inner)>;
    static_assert(
      ::exec::enter_scope_sender_in<
        decltype(scope),
        ::STDEXEC::env_of_t<Receiver>>);
    static_assert(
      std::is_constructible_v<
        inner_type,
        decltype(inner)>);
    return
      typename get_state_result<::STDEXEC::__data_of<Sender>, Receiver>::type(
        (decltype(scope)&&)scope,
        (decltype(inner)&&)inner,
        (Receiver&&)r);
  };
  static constexpr auto start = [](auto& state) noexcept {
    state.start();
  };
};

}

using within_t = detail::within::t;
inline constexpr within_t within;

}  // namespace exec

namespace STDEXEC {

template<>
struct __sexpr_impl<::exec::detail::within::tag>
  : ::exec::detail::within::impl {};

template<>
struct __sexpr_impl<::exec::within_t> : ::STDEXEC::__sexpr_defaults {
  template <class Sender, class... Env>
  static consteval auto get_completion_signatures() {
    static_assert(::STDEXEC::sender_expr_for<Sender, ::exec::within_t>);
    using type = decltype(
      ::STDEXEC::transform_sender(
        std::declval<Sender>(),
        std::declval<const Env&>()...));
    static_assert(!std::is_same_v<
      std::remove_cvref_t<type>,
      std::remove_cvref_t<Sender>>);
    return ::STDEXEC::get_completion_signatures<type, Env...>();
  }
};

}
