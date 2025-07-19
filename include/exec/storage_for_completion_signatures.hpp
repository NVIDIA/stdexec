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
#include <utility>
#include <variant>

namespace experimental::execution {

template<typename>
struct storage_for_completion_signature;

template<typename Tag, typename... Args>
struct storage_for_completion_signature<Tag(Args...)> {
  using tag_type = Tag;
  using signature_type = Tag(Args...);
  using normalized_signature_type_ = Tag(Args&&...);
  using tuple_type = std::tuple<Args...>;

  template<typename... OtherArgs>
    requires(sizeof...(OtherArgs) == sizeof...(Args))
         && std::is_constructible_v<tuple_type, OtherArgs...>
  constexpr explicit storage_for_completion_signature(Tag, OtherArgs&&... args)
    noexcept(std::is_nothrow_constructible_v<tuple_type, OtherArgs...>)
    : args_(static_cast<OtherArgs&&>(args)...)
  {}

  static constexpr auto tag() noexcept -> Tag {
    return {};
  }

  constexpr tuple_type& arguments() & noexcept {
    return args_;
  }

  constexpr const tuple_type& arguments() const& noexcept {
    return args_;
  }

  constexpr tuple_type& arguments() && noexcept {
    return args_;
  }

  constexpr const tuple_type& arguments() const&& noexcept {
    return args_;
  }

  constexpr auto forward_arguments() & noexcept {
    return forward_arguments_(*this);
  }

  constexpr auto forward_arguments() const& noexcept {
    return forward_arguments_(*this);
  }

  constexpr auto forward_arguments() && noexcept {
    return forward_arguments_(static_cast<storage_for_completion_signature&&>(*this));
  }

  constexpr auto forward_arguments() const&& noexcept {
    return forward_arguments_(static_cast<const storage_for_completion_signature&&>(*this));
  }

private:
  tuple_type args_;

  template<typename Self>
  static constexpr auto forward_arguments_(Self&& self) noexcept {
    return std::apply(
      []<typename... Ts>(Ts&&... args) noexcept {
        return std::forward_as_tuple(static_cast<Ts&&>(args)...);
      },
      static_cast<Self&&>(self).args_);
  }
};

namespace detail::storage_for_completion_signatures {

template<typename Signatures, typename Tag, typename... Args>
using storage_for_arrival =
  ::exec::storage_for_completion_signature<
    ::exec::matching_completion_signature_t<Signatures, Tag, Args...>>;

template<typename>
struct variant_for_signatures;

template<typename... Signatures>
struct variant_for_signatures<
  ::STDEXEC::completion_signatures<Signatures...>>
{
  using type = std::variant<
    std::monostate,
    ::exec::storage_for_completion_signature<Signatures>...>;
};

template<typename>
struct nothrow_storable;

template<typename Tag, typename... Args>
struct nothrow_storable<Tag(Args...)> {
  static constexpr bool value = (
    std::is_nothrow_constructible_v<
      Args,
      Args> && ...);
};

template<typename, typename...>
struct arrival_storable;

template<typename Tag, typename... StoredArgs, typename... Args>
struct arrival_storable<
  ::exec::storage_for_completion_signature<Tag(StoredArgs...)>,
  Args...> {
  static constexpr bool value =
    sizeof...(StoredArgs) == sizeof...(Args)
    && ((std::is_reference_v<StoredArgs>
         || !std::is_lvalue_reference_v<Args>) && ...);
};

struct ambiguous_normalized_completion_signatures;
struct non_persistable_completion_signature;

template<typename>
struct persistable_completion_signature;

template<typename Tag, typename... Args>
struct persistable_completion_signature<Tag(Args...)> {
  static constexpr bool value =
    ((std::is_reference_v<Args> || std::is_move_constructible_v<Args>) && ...);
};

template<typename>
struct persistable_completion_signatures;

template<typename... Signatures>
struct persistable_completion_signatures<
  ::STDEXEC::completion_signatures<Signatures...>>
{
  static constexpr bool value =
    (persistable_completion_signature<Signatures>::value && ...);
};

template<typename>
struct normalized_completion_signatures;

template<typename... Signatures>
struct normalized_completion_signatures<
  ::STDEXEC::completion_signatures<Signatures...>>
{
  using type = ::STDEXEC::transform_completion_signatures<
    ::STDEXEC::completion_signatures<
      typename ::exec::storage_for_completion_signature<
        Signatures>::normalized_signature_type_...>>;
};

template<typename Signatures>
using normalized_completion_signatures_t =
  typename normalized_completion_signatures<Signatures>::type;

template<typename Signatures>
inline constexpr bool has_unique_normalized_completion_signatures =
  ::STDEXEC::__mapply<::STDEXEC::__msize, Signatures>::value
  == ::STDEXEC::__mapply<
       ::STDEXEC::__msize,
       normalized_completion_signatures_t<Signatures>>::value;

template<typename InputSignatures, typename OutputSignatures>
consteval auto get_completion_signatures() {
  if constexpr (!has_unique_normalized_completion_signatures<InputSignatures>) {
    return ::STDEXEC::__throw_compile_time_error<
      ambiguous_normalized_completion_signatures>();
  } else if constexpr (!persistable_completion_signatures<InputSignatures>::value) {
    return ::STDEXEC::__throw_compile_time_error<
      non_persistable_completion_signature>();
  } else {
    return OutputSignatures{};
  }
}

template<typename Storage>
using variant_for_storage =
  decltype(std::declval<Storage>().get_variant_());

template<typename Visitor, typename Alternatives, typename... Variants>
struct nothrow_visit_stored_completions;

template<typename Visitor, typename... Alternatives>
struct nothrow_visit_stored_completions<
  Visitor,
  std::tuple<Alternatives...>>
{
  static constexpr bool value =
    std::is_nothrow_invocable_v<Visitor, Alternatives...>;
};

template<
  typename Visitor,
  typename... Alternatives,
  typename Variant,
  typename... Variants>
struct nothrow_visit_stored_completions<
  Visitor,
  std::tuple<Alternatives...>,
  Variant,
  Variants...>
{
private:
  template<std::size_t... Indices>
  static constexpr bool value_(std::index_sequence<Indices...>) {
    return (
      nothrow_visit_stored_completions<
        Visitor,
        std::tuple<
          Alternatives...,
          decltype(std::get<Indices + 1>(std::declval<Variant>()))>,
        Variants...>::value && ...);
  }

public:
  static constexpr bool value = value_(
    std::make_index_sequence<
      std::variant_size_v<std::remove_cvref_t<Variant>> - 1>{});
};

template<typename Visitor, typename... Variants>
constexpr bool nothrow_visit_stored_completions_for_variants_v =
  std::is_nothrow_invocable_v<Visitor>
  && nothrow_visit_stored_completions<
    Visitor,
    std::tuple<>,
    Variants...>::value;

template<typename Visitor, typename... Storages>
constexpr bool nothrow_visit_stored_completions_v =
  nothrow_visit_stored_completions_for_variants_v<
    Visitor,
    variant_for_storage<Storages>...>;

template<typename Result, typename Visitor, typename Alternatives, typename... Variants>
struct nothrow_visit_stored_completions_r;

template<typename Result, typename Visitor, typename... Alternatives>
struct nothrow_visit_stored_completions_r<
  Result,
  Visitor,
  std::tuple<Alternatives...>>
{
  static constexpr bool value =
    std::is_nothrow_invocable_r_v<Result, Visitor, Alternatives...>;
};

template<
  typename Result,
  typename Visitor,
  typename... Alternatives,
  typename Variant,
  typename... Variants>
struct nothrow_visit_stored_completions_r<
  Result,
  Visitor,
  std::tuple<Alternatives...>,
  Variant,
  Variants...>
{
private:
  template<std::size_t... Indices>
  static constexpr bool value_(std::index_sequence<Indices...>) {
    return (
      nothrow_visit_stored_completions_r<
        Result,
        Visitor,
        std::tuple<
          Alternatives...,
          decltype(std::get<Indices + 1>(std::declval<Variant>()))>,
        Variants...>::value && ...);
  }

public:
  static constexpr bool value = value_(
    std::make_index_sequence<
      std::variant_size_v<std::remove_cvref_t<Variant>> - 1>{});
};

template<typename Result, typename Visitor, typename... Variants>
constexpr bool nothrow_visit_stored_completions_r_for_variants_v =
  std::is_nothrow_invocable_r_v<Result, Visitor>
  && nothrow_visit_stored_completions_r<
    Result,
    Visitor,
    std::tuple<>,
    Variants...>::value;

template<typename Result, typename Visitor, typename... Storages>
constexpr bool nothrow_visit_stored_completions_r_v =
  nothrow_visit_stored_completions_r_for_variants_v<
    Result,
    Visitor,
    variant_for_storage<Storages>...>;

template<std::size_t Index = 1, typename Visitor, typename Variant>
constexpr decltype(auto) visit_variant_completion(
  Visitor&& visitor,
  Variant&& variant) {
  using variant_t = std::remove_cvref_t<Variant>;
  if constexpr (Index < std::variant_size_v<variant_t>) {
    if (variant.index() == Index) {
      return std::invoke(
        static_cast<Visitor&&>(visitor),
        std::get<Index>(static_cast<Variant&&>(variant)));
    }
    if constexpr (Index + 1 < std::variant_size_v<variant_t>) {
      return storage_for_completion_signatures::visit_variant_completion<Index + 1>(
        static_cast<Visitor&&>(visitor),
        static_cast<Variant&&>(variant));
    } else {
      STDEXEC_ASSERT(false);
      STDEXEC_UNREACHABLE();
    }
  } else {
    STDEXEC_ASSERT(false);
    STDEXEC_UNREACHABLE();
  }
}

template<typename Visitor, typename Storage>
constexpr decltype(auto) visit_present_stored_completions(
  Visitor&& visitor,
  Storage&& storage) {
  return storage_for_completion_signatures::visit_variant_completion(
    static_cast<Visitor&&>(visitor),
    static_cast<Storage&&>(storage).get_variant_());
}

template<typename Visitor, typename Storage, typename... Storages>
constexpr decltype(auto) visit_present_stored_completions(
  Visitor&& visitor,
  Storage&& storage,
  Storages&&... storages) {
  return storage_for_completion_signatures::visit_variant_completion(
    [&](auto&& completion) -> decltype(auto) {
      return storage_for_completion_signatures::visit_present_stored_completions(
        [&](auto&&... completions) -> decltype(auto) {
          return std::invoke(
            static_cast<Visitor&&>(visitor),
            static_cast<decltype(completion)&&>(completion),
            static_cast<decltype(completions)&&>(completions)...);
        },
        static_cast<Storages&&>(storages)...);
    },
    static_cast<Storage&&>(storage).get_variant_());
}

}

template<typename Visitor, typename Storage, typename... Storages>
constexpr decltype(auto) visit_stored_completion(
  Visitor&& visitor,
  Storage&& storage,
  Storages&&... storages) noexcept(
    detail::storage_for_completion_signatures::
      nothrow_visit_stored_completions_v<
        Visitor,
        Storage,
        Storages...>)
{
  if (!(storage.has_completion() && ... && storages.has_completion())) {
    return std::invoke(static_cast<Visitor&&>(visitor));
  }
  return detail::storage_for_completion_signatures::visit_present_stored_completions(
    static_cast<Visitor&&>(visitor),
    static_cast<Storage&&>(storage),
    static_cast<Storages&&>(storages)...);
}

template<typename Result, typename Visitor, typename Storage, typename... Storages>
constexpr Result visit_stored_completion(
  Visitor&& visitor,
  Storage&& storage,
  Storages&&... storages) noexcept(
    detail::storage_for_completion_signatures::
      nothrow_visit_stored_completions_r_v<
        Result,
        Visitor,
        Storage,
        Storages...>)
{
  if (!(storage.has_completion() && ... && storages.has_completion())) {
    if constexpr (std::is_void_v<Result>) {
      std::invoke(static_cast<Visitor&&>(visitor));
      return;
    } else {
      return std::invoke(static_cast<Visitor&&>(visitor));
    }
  }
  if constexpr (std::is_void_v<Result>) {
    detail::storage_for_completion_signatures::visit_present_stored_completions(
      static_cast<Visitor&&>(visitor),
      static_cast<Storage&&>(storage),
      static_cast<Storages&&>(storages)...);
    return;
  } else {
    return detail::storage_for_completion_signatures::visit_present_stored_completions(
      static_cast<Visitor&&>(visitor),
      static_cast<Storage&&>(storage),
      static_cast<Storages&&>(storages)...);
  }
}

enum class storage_for_completion_signatures_error_policy
{
  internalize,
  propagate
};

template<
  typename,
  storage_for_completion_signatures_error_policy =
    storage_for_completion_signatures_error_policy::internalize>
class storage_for_completion_signatures;

template<>
class storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<>,
  storage_for_completion_signatures_error_policy::internalize>
{
public:
  using completion_signatures = ::STDEXEC::completion_signatures<>;
  static constexpr bool nothrow_arrive = true;

  static consteval auto get_completion_signatures() noexcept {
    return completion_signatures{};
  }

  static constexpr std::variant<std::monostate> get_variant_() noexcept {
    return {};
  }

  static constexpr bool has_completion() noexcept {
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
  storage_for_completion_signatures_error_policy::propagate>
  : public storage_for_completion_signatures<
      ::STDEXEC::completion_signatures<>,
      storage_for_completion_signatures_error_policy::internalize>
{};

template<typename... Signatures>
class storage_for_completion_signatures<
  ::STDEXEC::completion_signatures<Signatures...>,
  storage_for_completion_signatures_error_policy::internalize>
{
  using input_completion_signatures_ =
    ::STDEXEC::transform_completion_signatures<
      ::STDEXEC::completion_signatures<Signatures...>>;
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
    input_completion_signatures_,
    maybe_throwing_signature_>;
  static constexpr bool nothrow_arrive = true;

  static consteval auto get_completion_signatures() {
    return detail::storage_for_completion_signatures::get_completion_signatures<
      input_completion_signatures_,
      completion_signatures>();
  }

private:
  template<typename Tag, typename... Args>
  using storage_type_for_arrival_ =
    detail::storage_for_completion_signatures::storage_for_arrival<
      input_completion_signatures_,
      Tag,
      Args...>;
  using storage_type_ =
    typename detail::storage_for_completion_signatures::variant_for_signatures<
      completion_signatures>::type;
  storage_type_ storage_;
public:
  template<typename Tag, typename... Args>
    requires std::is_constructible_v<
      storage_type_,
      std::in_place_type_t<storage_type_for_arrival_<Tag, Args...>>,
      Tag,
      Args...>
      && detail::storage_for_completion_signatures::arrival_storable<
        storage_type_for_arrival_<Tag, Args...>,
        Args...>::value
  constexpr void arrive(Tag t, Args&&... args) noexcept {
    STDEXEC_ASSERT(std::holds_alternative<std::monostate>(storage_));
    constexpr auto nothrow = std::is_nothrow_constructible_v<
      storage_type_for_arrival_<Tag, Args...>,
      Tag,
      Args...>;
    const auto impl = [&]() noexcept(nothrow) {
      storage_.template emplace<storage_type_for_arrival_<Tag, Args...>>(
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
          storage_for_completion_signature<
            ::STDEXEC::set_error_t(std::exception_ptr)>>(
              ::STDEXEC::set_error,
              std::current_exception());
      }
    }
  }
  constexpr bool has_completion() const noexcept {
    return !std::holds_alternative<std::monostate>(storage_);
  }

  constexpr storage_type_& get_variant_() & noexcept {
    return storage_;
  }

  constexpr const storage_type_& get_variant_() const& noexcept {
    return storage_;
  }

  constexpr storage_type_&& get_variant_() && noexcept {
    return static_cast<storage_type_&&>(storage_);
  }

  constexpr const storage_type_&& get_variant_() const&& noexcept {
    return static_cast<const storage_type_&&>(storage_);
  }

  template<::STDEXEC::receiver_of<completion_signatures> Receiver>
  constexpr bool complete(Receiver&& r) && noexcept {
    return std::visit(
      [&](auto&& active_alternative) noexcept {
        if constexpr (std::is_same_v<
          std::monostate,
          std::remove_cvref_t<decltype(active_alternative)>>)
        {
          return false;
        } else {
          using completion_type =
            std::remove_cvref_t<decltype(active_alternative)>;
          constexpr auto nothrow =
            std::is_nothrow_constructible_v<completion_type, completion_type&&>;
          const auto complete = [&]() noexcept(nothrow) {
            auto completion = completion_type(std::move(active_alternative));
            std::apply(
              [&](auto&&... args) noexcept {
                typename completion_type::tag_type{}(
                  (Receiver&&)r,
                  (decltype(args)&&)args...);
              },
              //  Odds are this is inside an operation state, which means that
              //  sending the completion signal may end our lifetime, which means
              //  we shouldn't send references into ourselves, therefore we move
              //  all the non-references onto the stack
              std::move(completion).forward_arguments());
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
  storage_for_completion_signatures_error_policy::propagate>
{
  using input_completion_signatures_ =
    ::STDEXEC::transform_completion_signatures<
      ::STDEXEC::completion_signatures<Signatures...>>;
public:
  using completion_signatures = input_completion_signatures_;
  static constexpr bool nothrow_arrive =
    (detail::storage_for_completion_signatures::nothrow_storable<
      Signatures>::value && ...);

  static consteval auto get_completion_signatures() {
    return detail::storage_for_completion_signatures::get_completion_signatures<
      input_completion_signatures_,
      completion_signatures>();
  }

private:
  template<typename Tag, typename... Args>
  using storage_type_for_arrival_ =
    detail::storage_for_completion_signatures::storage_for_arrival<
      input_completion_signatures_,
      Tag,
      Args...>;
  using storage_type_ =
    typename detail::storage_for_completion_signatures::variant_for_signatures<
      completion_signatures>::type;
  storage_type_ storage_;

public:
  template<typename Tag, typename... Args>
    requires std::is_constructible_v<
      storage_type_,
      std::in_place_type_t<storage_type_for_arrival_<Tag, Args...>>,
      Tag,
      Args...>
      && detail::storage_for_completion_signatures::arrival_storable<
        storage_type_for_arrival_<Tag, Args...>,
        Args...>::value
  constexpr void arrive(Tag t, Args&&... args) noexcept(
    std::is_nothrow_constructible_v<
      storage_type_for_arrival_<Tag, Args...>,
      Tag,
      Args...>)
  {
    STDEXEC_ASSERT(std::holds_alternative<std::monostate>(storage_));
    try {
      storage_.template emplace<storage_type_for_arrival_<Tag, Args...>>(
        (Tag&&)t,
        (Args&&)args...);
    } catch (...) {
      storage_.template emplace<std::monostate>();
      throw;
    }
  }

  constexpr bool has_completion() const noexcept {
    return !std::holds_alternative<std::monostate>(storage_);
  }

  constexpr storage_type_& get_variant_() & noexcept {
    return storage_;
  }

  constexpr const storage_type_& get_variant_() const& noexcept {
    return storage_;
  }

  constexpr storage_type_&& get_variant_() && noexcept {
    return static_cast<storage_type_&&>(storage_);
  }

  constexpr const storage_type_&& get_variant_() const&& noexcept {
    return static_cast<const storage_type_&&>(storage_);
  }

  template<::STDEXEC::receiver_of<completion_signatures> Receiver>
  constexpr bool complete(Receiver&& r) && noexcept(nothrow_arrive) {
    return std::visit(
      [&](auto&& active_alternative) noexcept(nothrow_arrive) {
        if constexpr (std::is_same_v<
          std::monostate,
          std::remove_cvref_t<decltype(active_alternative)>>)
        {
          return false;
        } else {
          using completion_type =
            std::remove_cvref_t<decltype(active_alternative)>;
          auto completion = completion_type(std::move(active_alternative));
          std::apply(
            [&](auto&&... args) noexcept(
              noexcept(typename completion_type::tag_type{}(
                (Receiver&&)r,
                (decltype(args)&&)args...))) {
              typename completion_type::tag_type{}(
                (Receiver&&)r,
                (decltype(args)&&)args...);
            },
            std::move(completion).forward_arguments());
          return true;
        }
      },
      (storage_type_&&)storage_);
  }
};

} // namespace exec
