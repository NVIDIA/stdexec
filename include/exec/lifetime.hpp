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
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "enter_scope_sender.hpp"
#include "enter_scopes.hpp"
#include "invoke.hpp"
#include "object.hpp"
#include "within.hpp"
#include "../stdexec/execution.hpp"

namespace experimental::execution {

namespace detail::lifetime {

template<typename F, typename... Objects>
concept function = ::STDEXEC::sender<
  std::invoke_result_t<
    F,
    ::exec::type_of_object_t<Objects>&...>>;

template<typename SenderFactory, typename... Objects>
concept sender_factory = ::exec::enter_scope_sender<
  std::invoke_result_t<
    SenderFactory,
    ::exec::enter_scope_sender_of_object_t<Objects>...>>;

struct t {
  template<typename F, ::exec::object... Objects>
    requires function<F, Objects...>
  constexpr ::STDEXEC::sender auto operator()(F&& f, Objects&&... objects) const
    noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<F>,
        F> &&
      (std::is_nothrow_constructible_v<
        std::remove_cvref_t<Objects>,
        Objects> && ...))
  {
    return (*this)(
      ::exec::enter_scopes,
      (F&&)f,
      (Objects&&)objects...);
  }
  template<typename SenderFactory, typename F, ::exec::object... Objects>
    requires
      sender_factory<SenderFactory, Objects...> &&
      function<F, Objects...>
  constexpr ::STDEXEC::sender auto operator()(
    SenderFactory&& sender_factory,
    F&& f,
    Objects&&... objects) const noexcept(
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<SenderFactory>,
        SenderFactory> &&
      std::is_nothrow_constructible_v<
        std::remove_cvref_t<F>,
        F> &&
      (std::is_nothrow_constructible_v<
        std::remove_cvref_t<Objects>,
        Objects> && ...))
  {
    return ::STDEXEC::__make_sexpr<t>(
      std::tuple(
        (SenderFactory&&)sender_factory,
        (F&&)f,
        (Objects&&)objects...));
  }
};

template<typename T>
class storage_for_object {
  union type_ {
    char c;
    T t;
    constexpr type_() noexcept : c() {}
    constexpr ~type_() noexcept {}
  };
  type_ storage_;
public:
  constexpr T* get_storage() noexcept {
    return std::addressof(storage_.t);
  }
  constexpr T& get_object() noexcept {
    return *std::launder(get_storage());
  }
};

template<typename Object>
using storage_for_object_t = storage_for_object<
  ::exec::type_of_object_t<Object>>;

template<typename... Objects>
using storage_for_objects_t = std::tuple<
  storage_for_object_t<Objects>...>;

template<typename Tuple, typename = std::remove_cvref_t<Tuple>>
class make_sender;

template<
  typename Tuple,
  typename SenderFactory,
  typename F,
  typename... Objects>
class make_sender<Tuple, std::tuple<SenderFactory, F, Objects...>> {
  static constexpr auto impl_(
    ::STDEXEC::__copy_cvref_t<Tuple, F>&& f,
    storage_for_object_t<Objects>&... storage) noexcept(
      std::is_nothrow_constructible_v<
        F,
        ::STDEXEC::__copy_cvref_t<Tuple, F>>)
  {
    return [f = (::STDEXEC::__copy_cvref_t<Tuple, F>&&)f, &storage...]() mutable noexcept(
      std::is_nothrow_invocable_v<
        F,
        ::exec::type_of_object_t<Objects>&...>)
    {
      return std::invoke(
        std::move(f),
        storage.get_object()...);
    };
  }
public:
  static constexpr bool nothrow = noexcept(
    ::exec::within(
      std::invoke(
        std::declval<::STDEXEC::__copy_cvref_t<Tuple, SenderFactory>>(),
        std::invoke(
          std::declval<::STDEXEC::__copy_cvref_t<Tuple, Objects>>(),
          std::declval<::exec::type_of_object_t<Objects>*>())...),
      ::STDEXEC::just() | ::exec::invoke(
        impl_(
          std::declval<::STDEXEC::__copy_cvref_t<Tuple, F>>(),
          std::declval<storage_for_object_t<Objects>&>()...))));
  using storage_type = storage_for_objects_t<Objects...>;
  static constexpr ::STDEXEC::sender auto impl(
    Tuple&& t,
    storage_type& storage) noexcept(nothrow)
  {
    return std::apply(
      [&](auto&& sender_factory, auto&& f, auto&&... objects) noexcept(nothrow)
      {
        return std::apply(
          [&](auto&... storage_for_object) noexcept(nothrow) {
            return ::exec::within(
              std::invoke(
                (decltype(sender_factory)&&)sender_factory,
                std::invoke(
                  (decltype(objects)&&)objects,
                  storage_for_object.get_storage())...),
              ::STDEXEC::just() | ::exec::invoke(
                impl_((decltype(f)&&)f, storage_for_object...)));
          },
          storage);
      },
      (Tuple&&)t);
  }
  using sender_type = decltype(
    impl(
      std::declval<Tuple>(),
      std::declval<storage_for_objects_t<Objects...>&>()));
};

template<typename Tuple, typename Receiver>
class state {
  using impl_ = make_sender<Tuple>;
  typename impl_::storage_type storage_;
  ::STDEXEC::connect_result_t<
    typename impl_::sender_type,
    Receiver> op_;
public:
  explicit constexpr state(Tuple&& t, Receiver r) noexcept(impl_::nothrow)
    : op_(
      ::STDEXEC::connect(
        impl_::impl((Tuple&&)t, storage_),
        (Receiver&&)r))
  {}
  constexpr void start() & noexcept {
    ::STDEXEC::start(op_);
  }
};

//  TODO: We should be able to get a better "message" than this
struct FAILED_TO_FORM_COMPLETION_SIGNATURES {};

class impl : public ::STDEXEC::__sexpr_defaults {
  template<typename Self, typename... Env>
  using completions_ = ::STDEXEC::completion_signatures_of_t<
    typename make_sender<::STDEXEC::__data_of<Self>>::sender_type,
    Env...>;
public:
  template<typename Self, typename... Env>
  static consteval auto __get_completion_signatures() {
    if constexpr (::STDEXEC::__minvocable_q<completions_, Self, Env...>) {
      return completions_<Self, Env...>{};
    } else if constexpr (sizeof...(Env) == 0) {
      return STDEXEC::__dependent_sender<Self>();
    } else {
      return ::STDEXEC::__throw_compile_time_error<
        FAILED_TO_FORM_COMPLETION_SIGNATURES,
        ::STDEXEC::_WITH_PRETTY_SENDER_<Self>>();
    }
  }
  static constexpr auto __get_state = []<typename Sender, typename Receiver>(
    Sender&& sender, Receiver r) noexcept(
      std::is_nothrow_constructible_v<
        state<::STDEXEC::__data_of<Sender>, Receiver>,
        ::STDEXEC::__data_of<Sender>,
        Receiver>) -> state<::STDEXEC::__data_of<Sender>, Receiver>
  {
    auto&& [_, tuple] = (Sender&&)sender;
    return state<decltype(tuple), Receiver>(
      (decltype(tuple)&&)tuple,
      (Receiver&&)r);
  };
  static constexpr auto __start = [](auto& state) noexcept {
    state.start();
  };
};

}

using lifetime_t = detail::lifetime::t;
inline constexpr lifetime_t lifetime;

}  // namespace exec

namespace STDEXEC {

template<>
struct __sexpr_impl<::exec::lifetime_t> : ::exec::detail::lifetime::impl {};

}
