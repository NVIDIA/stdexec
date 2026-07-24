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

#include "matching_completion_signature.hpp"

#include "../stdexec/execution.hpp"

#include <exception>
#include <functional>
#include <tuple>
#include <type_traits>
#include <variant>

namespace experimental::execution {

namespace detail::storage_for_completion_signatures {

template<typename T>
struct decay {
  using type = std::decay_t<T>;
};
template<typename T>
struct decay<T&> {
  using type = T&;
};
template<typename T>
struct decay<T&&> {
  using type = T&&;
};

template<typename>
struct tuple_for_signature;

template<typename Tag, typename... Args>
struct tuple_for_signature<Tag(Args...)> {
  using type = std::tuple<Tag, typename decay<Args>::type...>;
};

template<typename Signatures, typename Tag, typename... Args>
using tuple_for_arrival = typename tuple_for_signature<
  ::exec::matching_completion_signature_t<Signatures, Tag, Args...>>::type;

template<typename>
struct variant_for_signatures;

template<typename... Signatures>
struct variant_for_signatures<
  ::STDEXEC::completion_signatures<Signatures...>>
{
  using type = std::variant<
    std::monostate,
    typename tuple_for_signature<Signatures>::type...>;
};

template<typename>
struct signature;

template<typename Tag, typename... Args>
struct signature<Tag(Args...)> {
  using type = Tag(typename decay<Args>::type...);
};

template<typename, typename>
struct nothrow_visitable;

template<typename Visitor, typename Tag, typename... Args>
struct nothrow_visitable<Visitor, Tag(Args...)> {
  static constexpr bool value = std::is_nothrow_invocable_v<
    Visitor,
    Tag,
    typename decay<Args>::type...>;
};

template<typename>
struct nothrow_storable;

template<typename Tag, typename... Args>
struct nothrow_storable<Tag(Args...)> {
  static constexpr bool value = (
    std::is_nothrow_constructible_v<
      typename decay<Args>::type,
      Args> && ...);
};

template<typename, typename...>
struct arrival_storable;

template<typename Tag, typename... StoredArgs, typename... Args>
struct arrival_storable<std::tuple<Tag, StoredArgs...>, Args...> {
  static constexpr bool value =
    sizeof...(StoredArgs) == sizeof...(Args)
    && ((std::is_reference_v<StoredArgs>
         || !std::is_lvalue_reference_v<Args>) && ...);
};

}

enum class storage_for_completion_signatures_error_policy
{
  __internalize,
  __propagate
};

template<
  typename,
  storage_for_completion_signatures_error_policy =
    storage_for_completion_signatures_error_policy::__internalize>
class storage_for_completion_signatures;

template<>
class storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<>,
  storage_for_completion_signatures_error_policy::__internalize>
{
public:
  using completion_signatures = ::STDEXEC::completion_signatures<>;
  static constexpr bool nothrow_arrive = true;

  template<typename Visitor>
  constexpr bool visit(Visitor&&) && noexcept {
    return false;
  }

  template<::STDEXEC::receiver Receiver>
  constexpr bool complete(Receiver&&) && noexcept {
    return false;
  }
};

template<>
class storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<>,
  storage_for_completion_signatures_error_policy::__propagate>
  : public storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<>,
      storage_for_completion_signatures_error_policy::__internalize>
{};

template<typename... Signatures>
class storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<Signatures...>,
  storage_for_completion_signatures_error_policy::__internalize>
{
  using base_signatures_ = ::STDEXEC::completion_signatures<
    typename detail::storage_for_completion_signatures::signature<
      Signatures>::type...>;
  static constexpr auto noexcept_ =
    (detail::storage_for_completion_signatures::nothrow_storable<
      Signatures>::value && ...);
  using maybe_throwing_signature_ = std::conditional_t<
    noexcept_,
    ::STDEXEC::completion_signatures<>,
    ::STDEXEC::completion_signatures<
      ::STDEXEC::set_error_t(std::exception_ptr)>>;
public:
  using completion_signatures = ::STDEXEC::transform_completion_signatures<
    base_signatures_,
    maybe_throwing_signature_>;
  static constexpr bool nothrow_arrive = true;

private:
  template<typename Tag, typename... Args>
  using tuple_type_ =
    detail::storage_for_completion_signatures::tuple_for_arrival<
      ::STDEXEC::completion_signatures<Signatures...>,
      Tag,
      Args...>;
  using storage_type_ =
    typename detail::storage_for_completion_signatures::variant_for_signatures<
      completion_signatures>::type;
  storage_type_ storage_;
  template<typename Visitor>
  static constexpr bool nothrow_visitable_ = (
    detail::storage_for_completion_signatures::nothrow_visitable<
      Visitor,
      Signatures>::value && ...);
public:
  template<typename Tag, typename... Args>
    requires std::is_constructible_v<
      storage_type_,
      std::in_place_type_t<tuple_type_<Tag, Args...>>,
      Tag,
      Args...>
      && detail::storage_for_completion_signatures::arrival_storable<
        tuple_type_<Tag, Args...>,
        Args...>::value
  constexpr void arrive(Tag t, Args&&... args) noexcept {
    STDEXEC_ASSERT(std::holds_alternative<std::monostate>(storage_));
    constexpr auto nothrow = std::is_nothrow_constructible_v<
      tuple_type_<Tag, Args...>,
      Tag,
      Args...>;
    const auto impl = [&]() noexcept(nothrow) {
      storage_.template emplace<tuple_type_<Tag, Args...>>(
        (Tag&&)t,
        (Args&&)args...);
    };
    if constexpr (nothrow) {
      impl();
    } else {
      try {
        impl();
      } catch (...) {
        storage_.template emplace<
          std::tuple<
            ::STDEXEC::set_error_t,
            std::exception_ptr>>(
              ::STDEXEC::set_error,
              std::current_exception());
      }
    }
  }
  template<typename Visitor>
  constexpr bool visit(Visitor&& visitor) && noexcept(nothrow_visitable_<Visitor>)
  {
    return std::visit(
      [&](auto&& tuple_or_monostate) noexcept(nothrow_visitable_<Visitor>) {
        if constexpr (std::is_same_v<
          std::monostate,
          std::remove_cvref_t<decltype(tuple_or_monostate)>>)
        {
          return false;
        } else {
          std::apply(
            (Visitor&&)visitor,
            (decltype(tuple_or_monostate)&&)tuple_or_monostate);
          return true;
        }
      },
      (storage_type_&&)storage_);
  }
  template<::STDEXEC::receiver_of<completion_signatures> Receiver>
  constexpr bool complete(Receiver&& r) && noexcept {
    return std::visit(
      [&](auto&& tuple_or_monostate) noexcept {
        if constexpr (std::is_same_v<
          std::monostate,
          std::remove_cvref_t<decltype(tuple_or_monostate)>>)
        {
          return false;
        } else {
          using tuple_type =
            std::remove_cvref_t<decltype(tuple_or_monostate)>;
          constexpr auto nothrow =
            std::is_nothrow_constructible_v<tuple_type, tuple_type&&>;
          const auto complete = [&]() noexcept(nothrow) {
            std::apply(
              [&](const auto tag, auto&&... args) noexcept {
                tag((Receiver&&)r, (decltype(args)&&)args...);
              },
              //  Odds are this is inside an operation state, which means that
              //  sending the completion signal may end our lifetime, which means
              //  we shouldn't send references into ourselves, therefore we move
              //  all the non-references onto the stack
              tuple_type(std::move(tuple_or_monostate)));
          };
          if constexpr (nothrow) {
            complete();
          } else {
            try {
              complete();
            } catch (...) {
              ::STDEXEC::set_error((Receiver&&)r, std::current_exception());
            }
          }
          return true;
        }
      },
      (storage_type_&&)storage_);
  }
};

template<typename... Signatures>
class storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<Signatures...>,
  storage_for_completion_signatures_error_policy::__propagate>
{
  using base_signatures_ = ::STDEXEC::completion_signatures<
    typename detail::storage_for_completion_signatures::signature<
      Signatures>::type...>;
public:
  using completion_signatures = base_signatures_;
  static constexpr bool nothrow_arrive =
    (detail::storage_for_completion_signatures::nothrow_storable<
      Signatures>::value && ...);

private:
  template<typename Tag, typename... Args>
  using tuple_type_ =
    detail::storage_for_completion_signatures::tuple_for_arrival<
      ::STDEXEC::completion_signatures<Signatures...>,
      Tag,
      Args...>;
  using storage_type_ =
    typename detail::storage_for_completion_signatures::variant_for_signatures<
      completion_signatures>::type;
  storage_type_ storage_;
  template<typename Visitor>
  static constexpr bool nothrow_visitable_ = (
    detail::storage_for_completion_signatures::nothrow_visitable<
      Visitor,
      Signatures>::value && ...);

public:
  template<typename Tag, typename... Args>
    requires std::is_constructible_v<
      storage_type_,
      std::in_place_type_t<tuple_type_<Tag, Args...>>,
      Tag,
      Args...>
      && detail::storage_for_completion_signatures::arrival_storable<
        tuple_type_<Tag, Args...>,
        Args...>::value
  constexpr void arrive(Tag t, Args&&... args) noexcept(
    std::is_nothrow_constructible_v<
      tuple_type_<Tag, Args...>,
      Tag,
      Args...>)
  {
    STDEXEC_ASSERT(std::holds_alternative<std::monostate>(storage_));
    try {
      storage_.template emplace<tuple_type_<Tag, Args...>>(
        (Tag&&)t,
        (Args&&)args...);
    } catch (...) {
      storage_.template emplace<std::monostate>();
      throw;
    }
  }

  template<typename Visitor>
  constexpr bool visit(Visitor&& visitor) && noexcept(nothrow_visitable_<Visitor>)
  {
    return std::visit(
      [&](auto&& tuple_or_monostate) noexcept(nothrow_visitable_<Visitor>) {
        if constexpr (std::is_same_v<
          std::monostate,
          std::remove_cvref_t<decltype(tuple_or_monostate)>>)
        {
          return false;
        } else {
          std::apply(
            (Visitor&&)visitor,
            (decltype(tuple_or_monostate)&&)tuple_or_monostate);
          return true;
        }
      },
      (storage_type_&&)storage_);
  }

  template<::STDEXEC::receiver_of<completion_signatures> Receiver>
  constexpr bool complete(Receiver&& r) && noexcept(nothrow_arrive) {
    return std::visit(
      [&](auto&& tuple_or_monostate) noexcept(nothrow_arrive) {
        if constexpr (std::is_same_v<
          std::monostate,
          std::remove_cvref_t<decltype(tuple_or_monostate)>>)
        {
          return false;
        } else {
          using tuple_type =
            std::remove_cvref_t<decltype(tuple_or_monostate)>;
          std::apply(
            [&](const auto tag, auto&&... args) noexcept(
              noexcept(tag((Receiver&&)r, (decltype(args)&&)args...))) {
              tag((Receiver&&)r, (decltype(args)&&)args...);
            },
            tuple_type(std::move(tuple_or_monostate)));
          return true;
        }
      },
      (storage_type_&&)storage_);
  }
};

} // namespace exec
