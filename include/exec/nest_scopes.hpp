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
#include <exception>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#include "elide.hpp"
#include "enter_scope_sender.hpp"
#include "sequence.hpp"
#include "storage_for_completion_signatures.hpp"
#include "../stdexec/execution.hpp"

namespace experimental::execution {

namespace detail::nest_scopes {

template<typename... Senders>
constexpr auto make_exit_scope_sender_impl(std::tuple<Senders...> tuple)
  noexcept(noexcept(::exec::sequence(std::declval<Senders>()...)))
{
  return std::apply(
    [](auto&&... senders) noexcept(
      noexcept(::exec::sequence((decltype(senders)&&)senders...)))
    {
      return ::exec::sequence((decltype(senders)&&)senders...);
    },
    std::move(tuple));
}

template<typename... Senders, typename T, typename... Rest>
constexpr auto make_exit_scope_sender_impl(
  std::tuple<Senders...> tuple,
  T&& t,
  Rest&&... rest) noexcept(
    //  Assumes exec::sequence just decay-copies its arguments
    (std::is_nothrow_constructible_v<
      std::remove_cvref_t<Senders>,
      Senders...> && ...) &&
    std::is_nothrow_constructible_v<
      std::remove_cvref_t<T>,
      T> &&
    (std::is_nothrow_constructible_v<
      std::remove_cvref_t<Rest>,
      Rest> && ...))
{
  return std::apply(
    [&](auto&&... senders) noexcept(
      noexcept(nest_scopes::make_exit_scope_sender_impl(
        std::declval<std::tuple<Senders..., T&&>>(),
        std::declval<Rest>()...)))
    {
      return nest_scopes::make_exit_scope_sender_impl(
        std::forward_as_tuple(
          (T&&)t,
          (decltype(senders)&&)senders...),
        (Rest&&)rest...);
    },
    std::move(tuple));
}

//  Obtains the composed exit sender by reversing the order and wrapping in
//  exec::sequence
template<typename... Senders>
constexpr auto make_exit_scope_sender(Senders&&... senders) noexcept(
  noexcept(
    nest_scopes::make_exit_scope_sender_impl(
      std::tuple<>{},
      std::declval<Senders>()...)))
{
  return nest_scopes::make_exit_scope_sender_impl(
    std::tuple<>{},
    (Senders&&)senders...);
}

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

template<typename Sender, typename Env>
inline constexpr bool nothrow_connect_sender =
  ::STDEXEC::__nothrow_connectable<
    Sender,
    ::STDEXEC::__receiver_archetype<Env>> &&
  //  Because we move the sender out of storage so we can reuse it
  std::is_nothrow_move_constructible_v<Sender>;

template<typename Env, typename... Senders>
using extra_set_error_signatures_t = std::conditional_t<
  (nothrow_connect_sender<Senders, Env> || ...),
  ::STDEXEC::completion_signatures<>,
  ::STDEXEC::completion_signatures<
    ::STDEXEC::set_error_t(std::exception_ptr)>>;

template<typename...>
using remove_set_value_t = ::STDEXEC::completion_signatures<>;

template<typename Env, typename First, typename... Rest>
using storage_for_completion_signatures_t =
  ::exec::storage_for_completion_signatures<
    ::STDEXEC::transform_completion_signatures<
      ::STDEXEC::__concat_completion_signatures_t<
        //  Because the first sender can be connected in the constructor it is
        //  handled differently
        ::STDEXEC::completion_signatures_of_t<First, Env>,
        ::STDEXEC::completion_signatures_of_t<
          std::remove_cvref_t<Rest>,
          Env>...>,
      //  Same reason here as above, since we don't connect the first sender as
      //  part of the asynchronous part of the operation we don't need to check
      //  it here
      extra_set_error_signatures_t<Env, Rest...>,
      remove_set_value_t>>;

template<typename Env, typename First, typename... Rest>
using exit_scope_sender_t = decltype(
  nest_scopes::make_exit_scope_sender(
    std::declval<::exec::exit_scope_sender_of_t<First, Env>>(),
    std::declval<::exec::exit_scope_sender_of_t<
      std::remove_cvref_t<Rest>,
      Env>>()...));

template<typename Env>
struct completions {
  template<typename First, typename... Rest>
  using __f = ::STDEXEC::__concat_completion_signatures_t<
    typename storage_for_completion_signatures_t<Env, First, Rest...>::
      completion_signatures,
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_value_t(
        exit_scope_sender_t<Env, First, Rest...>)>>;
};

template<typename, typename, typename...>
class state_impl;
template<typename Receiver, std::size_t... Ns, typename First, typename... Rest>
class state_impl<Receiver, std::index_sequence<Ns...>, First, Rest...> {
  using env_type_ = ::STDEXEC::env_of_t<Receiver>;
  struct receiver_base_ {
    using receiver_concept = ::STDEXEC::receiver_t;
    state_impl& self_;
    constexpr env_type_ get_env() const noexcept {
      return ::STDEXEC::get_env(self_.r_);
    }
  };
  template<std::size_t N>
  using exit_scope_sender_type_ = ::exec::exit_scope_sender_of_t<
    std::tuple_element_t<N, std::tuple<First, Rest...>>,
    env_type_>;
  template<std::size_t N>
  struct receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value(exit_scope_sender_type_<N> exit) && noexcept {
      if constexpr (N) {
        auto&& v = std::get<N>(self_.storage_);
        v.template emplace<exit_scope_sender_type_<N>>(std::move(exit));
      } else {
        auto&& o = std::get<N>(self_.storage_);
        o.emplace(std::move(exit));
      }
      if constexpr (N == sizeof...(Rest)) {
        std::apply(
          [&](auto&& o, auto&&... vs) noexcept {
            ::STDEXEC::set_value(
              std::move(self_.r_),
              nest_scopes::make_exit_scope_sender(
                *(decltype(o)&&)o,
                std::move(*std::get_if<1>(&vs))...));
          },
          std::move(self_.storage_));
      } else {
        auto&& v = std::get<N + 1>(self_.storage_);
        const auto ptr = std::get_if<0>(&v);
        STDEXEC_ASSERT(ptr);
        using sender_type = std::remove_cvref_t<decltype(*ptr)>;
        using receiver_type = receiver_type_<N + 1>;
        constexpr bool nothrow = ::STDEXEC::__nothrow_connectable<
          sender_type,
          receiver_type>;
        //  Because we're about to destroy *this
        auto&& self = self_;
        try {
          auto&& op = self.op_.template emplace<N + 1>(
            ::exec::elide([&]() noexcept(nothrow) {
              return ::STDEXEC::connect(
                std::move(*ptr),
                receiver_type{{self}});
            }));
          ::STDEXEC::start(op);
        } catch (...) {
          if constexpr (nothrow) {
            STDEXEC_UNREACHABLE();
          } else {
            //  It's important we use self and not self_ even here because
            //  despite an exception being thrown *this may still have been
            //  destroyed
            self.completion_.arrive(
              ::STDEXEC::set_error,
              std::current_exception());
            //  We successfully completed, the failure is because of the next
            //  operation, so we add one to ensure that we get rolled back
            self.template rollback_<N + 1>();
          }
        }
      }
    }
    template<typename... Args>
    constexpr void set_error([[maybe_unused]] Args&&... args) && noexcept {
      self_.completion_.arrive(::STDEXEC::set_error, (Args&&)args...);
      self_.template rollback_<N>();
    }
    template<typename... Args>
    constexpr void set_stopped([[maybe_unused]] Args&&... args) && noexcept {
      self_.completion_.arrive(::STDEXEC::set_stopped, (Args&&)args...);
      self_.template rollback_<N>();
    }
  };
  template<std::size_t N>
  struct rollback_receiver_type_ : receiver_base_ {
    using receiver_base_::self_;
    constexpr void set_value() && noexcept {
      self_.template rollback_<N>();
    }
  };
  template<std::size_t N>
  constexpr void rollback_() noexcept {
    if constexpr (N) {
      auto&& sender = [&]() noexcept -> decltype(auto) {
        if constexpr (N - 1) {
          auto&& v = std::get<N - 1>(storage_);
          const auto ptr = std::get_if<1>(&v);
          STDEXEC_ASSERT(ptr);
          return *ptr;
        } else {
          auto&& o = std::get<0>(storage_);
          STDEXEC_ASSERT(o);
          return *o;
        }
      }();
      auto&& op = op_.template emplace<sizeof...(Rest) + N>(
        ::exec::elide([&]() noexcept {
          return ::STDEXEC::connect(
            std::move(sender),
            rollback_receiver_type_<N - 1>{{*this}});
        }));
      ::STDEXEC::start(op);
    } else {
      std::move(completion_).complete(std::move(r_));
    }
  }
  using first_operation_state_type_ = ::STDEXEC::connect_result_t<
    First,
    receiver_type_<0>>;
  Receiver r_;
  std::variant<
    first_operation_state_type_,
    ::STDEXEC::connect_result_t<
      Rest,
      receiver_type_<Ns + 1>>...,
    ::STDEXEC::connect_result_t<
      exit_scope_sender_type_<0>,
      rollback_receiver_type_<0>>,
    ::STDEXEC::connect_result_t<
      exit_scope_sender_type_<Ns + 1>,
      rollback_receiver_type_<Ns + 1>>...> op_;
  std::tuple<
    std::optional<exit_scope_sender_type_<0>>,
    std::variant<
      Rest,
      exit_scope_sender_type_<Ns + 1>>...> storage_;
  storage_for_completion_signatures_t<env_type_, First, Rest...> completion_;
public:
  template<typename... Ts>
  explicit constexpr state_impl(
    Receiver r,
    First&& first,
    Ts&&... rest) noexcept(
      ::STDEXEC::__nothrow_connectable<First, receiver_type_<0>> &&
      (std::is_nothrow_constructible_v<
        Rest,
        Ts> && ...))
    : r_((Receiver&&)r),
      op_(
        std::in_place_type<first_operation_state_type_>,
        ::exec::elide([&]() noexcept(
          ::STDEXEC::__nothrow_connectable<First, receiver_type_<0>>)
        {
          return ::STDEXEC::connect(
            (First&&)first,
            receiver_type_<0>{{*this}});
        })),
      storage_(
        std::nullopt,
        (Ts&&)rest...)
  {}
  constexpr void start() & noexcept {
    const auto ptr = std::get_if<first_operation_state_type_>(&op_);
    STDEXEC_ASSERT(ptr);
    ::STDEXEC::start(*ptr);
  }
};

template<typename Receiver, typename First, typename... Rest>
class state : public state_impl<
  Receiver,
  std::index_sequence_for<Rest...>,
  First,
  Rest...>
{
  using base_ = state_impl<
    Receiver,
    std::index_sequence_for<Rest...>,
    First,
    Rest...>;
public:
  using base_::base_;
};

//  TODO: We should be able to get a better "message" than this
struct FAILED_TO_FORM_COMPLETION_SIGNATURES {};

template<typename Receiver>
struct connect_result {
  template<typename First, typename... Rest>
  using __f = state<Receiver, First, std::remove_cvref_t<Rest>...>;
};

template<typename Receiver>
struct nothrow_connect {
  template<typename First, typename... Rest>
  using __f = std::is_nothrow_constructible<
    typename connect_result<Receiver>::template __f<First, Rest...>,
    Receiver,
    First,
    Rest...>;
};

class impl : public ::STDEXEC::__sexpr_defaults {
  template<typename Sender, typename... Env>
  using completions_ = ::STDEXEC::__children_of<Sender, completions<Env...>>;
public:
  template<typename Sender, typename... Env>
  static consteval auto __get_completion_signatures() {
    if constexpr (sizeof...(Env) == 0) {
      return STDEXEC::__dependent_sender<Sender>();
    } else if constexpr (::STDEXEC::__minvocable_q<completions_, Sender, Env...>) {
      return completions_<Sender, Env...>{};
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
          -> ::STDEXEC::__children_of<Sender, connect_result<Receiver>>
      {
        return ::STDEXEC::__children_of<Sender, connect_result<Receiver>>(
          (Receiver&&)r, 
          (decltype(children)&&)children...);
      },
      (Sender&&)sender);
  };
};

}

using nest_scopes_t = detail::nest_scopes::t;
inline constexpr nest_scopes_t nest_scopes;

}  // namespace exec

namespace STDEXEC {

template<>
struct __sexpr_impl<::exec::nest_scopes_t> : ::exec::detail::nest_scopes::impl
{};

}
