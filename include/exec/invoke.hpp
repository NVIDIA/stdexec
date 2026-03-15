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

#include <exception>
#include <functional>
#include <tuple>
#include <type_traits>

#include "elide.hpp"
#include "../stdexec/execution.hpp"

namespace experimental::execution {

namespace detail::invoke {

struct tag {};

struct t {
  template<::STDEXEC::sender Sender, typename F>
  constexpr ::STDEXEC::sender auto operator()(Sender&& sender, F&& f) const
    noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<Sender>,
        Sender> &&
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<F>,
        F>)
  {
    return ::STDEXEC::__make_sexpr<t>(
      (F&&)f,
      (Sender&&)sender);
  }
  template<typename F>
  constexpr auto operator()(F&& f) const noexcept(
    std::is_nothrow_constructible_v<
      std::remove_cvref_t<F>,
      F>)
  {
    return ::STDEXEC::__closure(*this, (F&&)f);
  }
  template<typename Sender, typename Env>
  static constexpr auto transform_sender(
    ::STDEXEC::set_value_t,
    Sender&& sender,
    const Env&) noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<Sender>,
        Sender>)
  {
    auto&& [_, f, predecessor] = (Sender&&)sender;
    static_assert(::STDEXEC::sender<decltype(predecessor)>);
    return ::STDEXEC::__make_sexpr<tag>(
      std::tuple((decltype(f)&&)f, (decltype(predecessor)&&)predecessor));
  }
};

template<typename Sender, typename... Env>
struct sender_check;
template<typename Sender>
struct sender_check<Sender> {
  static constexpr bool value = ::STDEXEC::sender<Sender>;
};
template<typename Sender, typename Env>
struct sender_check<Sender, Env> {
  static constexpr bool value = ::STDEXEC::sender_in<Sender, Env>;
};

template<typename F, typename... Env>
class transform_set_value {
  template<typename... Args>
  class impl_ {
    static_assert(std::is_invocable_v<F, Args...>);
    using sender_ = std::invoke_result_t<F, Args...>;
    static_assert(sender_check<sender_, Env...>::value);
    static constexpr bool nothrow_invoke_ = std::is_nothrow_invocable_v<
      F,
      Args...>;
    static constexpr bool nothrow_connect_ = ::STDEXEC::__nothrow_connectable<
      sender_,
      ::STDEXEC::__receiver_archetype<Env...>>;
  public:
    using type = ::STDEXEC::transform_completion_signatures<
      ::STDEXEC::completion_signatures_of_t<
        sender_,
        Env...>,
      std::conditional_t<
        nothrow_invoke_ && nothrow_connect_,
        ::STDEXEC::completion_signatures<>,
        ::STDEXEC::completion_signatures<
          ::STDEXEC::set_error_t(std::exception_ptr)>>>;
  };
public:
  template<typename... Args>
  using fn = impl_<Args...>::type;
};

template<typename Sender, typename F, typename... Env>
using completions = ::STDEXEC::transform_completion_signatures<
  ::STDEXEC::completion_signatures_of_t<Sender, Env...>,
  ::STDEXEC::completion_signatures<>,
  transform_set_value<F, Env...>::template fn>;

//  TODO: We should be able to get a better "message" than this
struct FAILED_TO_FORM_COMPLETION_SIGNATURES {};

template<::STDEXEC::receiver Receiver>
struct receiver_ref {
  using receiver_concept = ::STDEXEC::receiver_t;
  Receiver& r_;
  constexpr ::STDEXEC::env_of_t<Receiver> get_env() const noexcept {
    return ::STDEXEC::get_env(r_);
  }
  template<typename... Args>
    requires ::STDEXEC::receiver_of<
      Receiver,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_value_t(Args...)>>
  constexpr void set_value(Args&&... args) && noexcept {
    ::STDEXEC::set_value((Receiver&&)r_, (Args&&)args...);
  }
  template<typename T>
    requires ::STDEXEC::receiver_of<
      Receiver,
      ::STDEXEC::completion_signatures<
        ::STDEXEC::set_error_t(T)>>
  constexpr void set_error(T&& t) && noexcept {
    ::STDEXEC::set_error((Receiver&&)r_, (T&&)t);
  }
  constexpr void set_stopped() && noexcept requires ::STDEXEC::receiver_of<
    Receiver,
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_stopped_t()>>
  {
    ::STDEXEC::set_stopped((Receiver&&)r_);
  }
};

template<typename F, typename Receiver, typename Additional, typename Signatures, typename InProgress = std::tuple<>>
struct variant_for_operation_states;

template<typename F, typename Receiver, typename Additional, typename... Args, typename... Signatures, typename... States>
struct variant_for_operation_states<
  F,
  Receiver,
  Additional,
  ::STDEXEC::completion_signatures<
    ::STDEXEC::set_value_t(Args...),
    Signatures...>,
  std::tuple<States...>>
{
  using type = variant_for_operation_states<
    F,
    Receiver,
    Additional,
    ::STDEXEC::completion_signatures<Signatures...>,
    std::tuple<
      ::STDEXEC::connect_result_t<
        std::invoke_result_t<
          F,
          Args...>,
        receiver_ref<Receiver>>,
      States...>>::type;
};

template<typename F, typename Receiver, typename Additional, typename Signature, typename... Signatures, typename States>
struct variant_for_operation_states<
  F,
  Receiver,
  Additional,
  ::STDEXEC::completion_signatures<
    Signature,
    Signatures...>,
  States>
{
  using type = variant_for_operation_states<
    F,
    Receiver,
    Additional,
    ::STDEXEC::completion_signatures<Signatures...>,
    States>::type;
};

template<typename F, typename Receiver, typename... Additional, typename... States>
struct variant_for_operation_states<
  F,
  Receiver,
  std::tuple<Additional...>,
  ::STDEXEC::completion_signatures<>,
  std::tuple<States...>>
{
  using type = ::STDEXEC::__munique<
    ::STDEXEC::__qq<std::variant>>::__f<Additional..., States...>;
};

template<typename Sender, typename F, typename Receiver>
class state {
  using env_type_ = ::STDEXEC::env_of_t<Receiver>;
  struct receiver_type_ {
    using receiver_concept = ::STDEXEC::receiver_t;
    state& self_;
    constexpr env_type_ get_env() const noexcept;
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
  using operation_states_type_ = variant_for_operation_states<
    F,
    Receiver,
    std::tuple<operation_state_type_>,
    ::STDEXEC::completion_signatures_of_t<Sender, receiver_type_>>::type;
  Receiver r_;
  F f_;
  operation_states_type_ ops_;
  static constexpr bool nothrow_connect_ = ::STDEXEC::__nothrow_connectable<
    Sender,
    receiver_type_>;
public:
  template<typename T>
  explicit constexpr state(Sender&& s, T&& t, Receiver r) noexcept(
    std::is_nothrow_constructible_v<F, T> && nothrow_connect_)
    : r_((Receiver&&)r),
      f_((T&&)t),
      ops_(
        std::in_place_type<operation_state_type_>,
        ::exec::elide([&]() noexcept(nothrow_connect_) {
          return ::STDEXEC::connect(
            (Sender&&)s,
            receiver_type_{*this});
        }))
  {}
  constexpr void start() & noexcept {
    const auto ptr = std::get_if<operation_state_type_>(&ops_);
    STDEXEC_ASSERT(ptr);
    ::STDEXEC::start(*ptr);
  }
};

template<typename Sender, typename F, typename Receiver>
constexpr auto state<Sender, F, Receiver>::receiver_type_::get_env() const
  noexcept -> env_type_
{
  return ::STDEXEC::get_env(self_.r_);
}

template<typename Sender, typename F, typename Receiver>
template<typename... Args>
constexpr void state<Sender, F, Receiver>::receiver_type_::set_value(
  Args&&... args) && noexcept
{
  constexpr auto nothrow_connect = ::STDEXEC::__nothrow_connectable<
    std::invoke_result_t<F, Args...>,
    receiver_ref<Receiver>>;
  //  We store this locally because we're going to destroy the operation state
  //  and therefore, transitively, *this
  auto&& self = self_;
  try {
    //  It's important we use auto&& here not auto because the invocable might
    //  return a reference to a sender
    auto&& sender = std::invoke((F&&)self.f_, (Args&&)args...);
    //  receiver_ref is important here, imagine we didn't use receiver_ref and
    //  instead directly moved the final receiver into connect, then if connect
    //  threw we'd already potentially have "consumed" the receiver and wouldn't
    //  be able to send the error completion anywhere
    using op_state_type = ::STDEXEC::connect_result_t<
      decltype(sender),
      receiver_ref<Receiver>>;
    auto&& op = self.ops_.template emplace<op_state_type>(
      ::exec::elide([&]() noexcept(nothrow_connect) {
        return ::STDEXEC::connect(
          (decltype(sender)&&)sender,
          receiver_ref<Receiver>{self.r_});
      }));
    ::STDEXEC::start(op);
  } catch (...) {
    if constexpr (std::is_nothrow_invocable_v<F, Args...> && nothrow_connect) {
      STDEXEC_UNREACHABLE();
    } else {
      ::STDEXEC::set_error((Receiver&&)self.r_, std::current_exception());
    }
  }
}

template<typename Sender, typename F, typename Receiver>
template<typename... Args>
constexpr void state<Sender, F, Receiver>::receiver_type_::set_error(
  Args&&... args) && noexcept
{
  ::STDEXEC::set_error((Receiver&&)self_.r_, (Args&&)args...);
}

template<typename Sender, typename F, typename Receiver>
template<typename... Args>
constexpr void state<Sender, F, Receiver>::receiver_type_::set_stopped(
  Args&&... args) && noexcept
{
  ::STDEXEC::set_stopped((Receiver&&)self_.r_, (Args&&)args...);
}

template<typename Sender, typename Receiver>
class nothrow_get_state {
  using tuple_type_ = std::remove_cvref_t<::STDEXEC::__data_of<Sender>>;
  using sender_type_ = ::STDEXEC::__copy_cvref_t<
    Sender,
    std::tuple_element_t<1, tuple_type_>>;
  using invocable_type_ = std::tuple_element_t<0, tuple_type_>;
public:
  static constexpr bool value = std::is_nothrow_constructible_v<
    state<sender_type_, invocable_type_, Receiver>,
    sender_type_,
    ::STDEXEC::__copy_cvref_t<Sender, invocable_type_>,
    Receiver>;
};

template<typename Tuple, typename, typename = std::remove_cvref_t<Tuple>>
struct get_state_result;
template<typename Tuple, typename Receiver, typename F, typename Sender>
struct get_state_result<Tuple, Receiver, std::tuple<F, Sender>> {
  using type = state<
    ::STDEXEC::__copy_cvref_t<Tuple, Sender>,
    F,
    Receiver>;
};

struct impl : public ::STDEXEC::__sexpr_defaults {
  template<typename Self, typename... Env>
  static consteval auto __get_completion_signatures() {
    using tuple = std::remove_cvref_t<
      ::STDEXEC::__data_of<Self>>;
    using f = std::tuple_element_t<0, tuple>;
    using sender = ::STDEXEC::__copy_cvref_t<Self, std::tuple_element_t<1, tuple>>;
    static_assert(::STDEXEC::sender<sender>);
    if constexpr (sizeof...(Env)) {
      static_assert(::STDEXEC::sender_in<sender, Env...>);
    }
    if constexpr (::STDEXEC::__minvocable_q<completions, sender, f, Env...>) {
      return completions<sender, f, Env...>{};
    } else {
      return ::STDEXEC::__throw_compile_time_error<
        FAILED_TO_FORM_COMPLETION_SIGNATURES,
        ::STDEXEC::_WITH_PRETTY_SENDER_<Self>>();
    }
  }
  static constexpr auto __get_state = []<typename Sender, typename Receiver>(
    Sender&& sender, Receiver r) noexcept(
      nothrow_get_state<Sender, Receiver>::value)
        -> get_state_result<::STDEXEC::__data_of<Sender>, Receiver>::type
  {
    auto&& [_, tuple] = (Sender&&)sender;
    auto&& [f, inner] = (decltype(tuple)&&)tuple;
    return
      typename get_state_result<::STDEXEC::__data_of<Sender>, Receiver>::type(
        (decltype(inner)&&)inner,
        (decltype(f)&&)f,
        (Receiver&&)r);
  };
  static constexpr auto __start = [](auto& state) noexcept {
    state.start();
  };
};

}

using invoke_t = detail::invoke::t;
inline constexpr invoke_t invoke;

}  // namespace exec

namespace STDEXEC {

template<>
struct __sexpr_impl<::exec::detail::invoke::tag>
  : ::exec::detail::invoke::impl {};

template<>
struct __sexpr_impl<::exec::invoke_t> : ::STDEXEC::__sexpr_defaults {
  template <class Sender, class... Env>
  static consteval auto get_completion_signatures() {
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
