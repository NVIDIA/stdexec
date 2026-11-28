/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Robert Leahy. All rights reserved.
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

#include "../stdexec/execution.hpp"

#include <cstddef>
#include <exception>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace experimental::execution
{
  namespace detail::started_by
  {
    struct default_base
    {};

    template <class _Base, class... _Args>
    struct BASE_MUST_BE_CONSTRUCTIBLE_FROM
    {};

    template <class _Initiation, class _State>
    struct INITIATION_MUST_BE_INVOCABLE_WITH_OPERATION_STATE
    {};

    template <class _Derived, class _Base, class _Signatures>
    struct state_base;

    template <class _Derived, class _Base>
    struct state_base<_Derived, _Base, ::STDEXEC::completion_signatures<>> : _Base
    {
      state_base() = default;

      using _Base::_Base;

      template <class... _Args>
        requires(sizeof...(_Args) != 0) && (!std::is_abstract_v<_Base>)
             && std::is_constructible_v<_Base, _Args...>
      constexpr explicit state_base(_Args&&... __args)
        noexcept(std::is_nothrow_constructible_v<_Base, _Args...>)
        : _Base(static_cast<_Args&&>(__args)...)
      {}
    };

    template <class _Derived, class _Base, class _Tag, class... _Args>
    struct state_base<_Derived, _Base, ::STDEXEC::completion_signatures<_Tag(_Args...)>> : _Base
    {
      state_base() = default;

      using _Base::_Base;

      template <class... _BaseArgs>
        requires(sizeof...(_BaseArgs) != 0) && (!std::is_abstract_v<_Base>)
             && std::is_constructible_v<_Base, _BaseArgs...>
      constexpr explicit state_base(_BaseArgs&&... __args)
        noexcept(std::is_nothrow_constructible_v<_Base, _BaseArgs...>)
        : _Base(static_cast<_BaseArgs&&>(__args)...)
      {}

      constexpr void complete(_Tag __tag, _Args&&... __args) noexcept
      {
        static_cast<_Derived&>(*this).__complete(__tag, static_cast<_Args&&>(__args)...);
      }
    };

    template <class _Derived,
              class _Base,
              class _Tag,
              class... _Args,
              class _NextSignature,
              class... _RemainingSignatures>
    struct state_base<
      _Derived,
      _Base,
      ::STDEXEC::completion_signatures<_Tag(_Args...), _NextSignature, _RemainingSignatures...>>
      : state_base<_Derived,
                   _Base,
                   ::STDEXEC::completion_signatures<_NextSignature, _RemainingSignatures...>>
    {
      using __base_t =
        state_base<_Derived,
                   _Base,
                   ::STDEXEC::completion_signatures<_NextSignature, _RemainingSignatures...>>;

      state_base() = default;

      using __base_t::__base_t;

      template <class... _BaseArgs>
        requires(sizeof...(_BaseArgs) != 0) && (!std::is_abstract_v<_Base>)
             && std::is_constructible_v<_Base, _BaseArgs...>
      constexpr explicit state_base(_BaseArgs&&... __args)
        noexcept(std::is_nothrow_constructible_v<_Base, _BaseArgs...>)
        : __base_t(static_cast<_BaseArgs&&>(__args)...)
      {}

      using __base_t::complete;

      constexpr void complete(_Tag __tag, _Args&&... __args) noexcept
      {
        static_cast<_Derived&>(*this).__complete(__tag, static_cast<_Args&&>(__args)...);
      }
    };

    template <class _Signatures, class _Base>
    struct base_probe : state_base<base_probe<_Signatures, _Base>, _Base, _Signatures>
    {
      using __base_t = state_base<base_probe, _Base, _Signatures>;
      using __base_t::__base_t;

      template <class _Tag, class... _Args>
      constexpr void __complete(_Tag, _Args&&...) noexcept
      {}
    };

    template <class _Signatures, class _Base, class _Initiation, class _Receiver>
    class state
      : public state_base<state<_Signatures, _Base, _Initiation, _Receiver>, _Base, _Signatures>
    {
      using __base_t =
        state_base<state<_Signatures, _Base, _Initiation, _Receiver>, _Base, _Signatures>;

     public:
      using operation_state_concept = ::STDEXEC::operation_state_tag;

      template <class... _Args>
      constexpr explicit state(_Initiation __init, _Receiver __receiver, _Args&&... __args)
        noexcept(std::is_nothrow_move_constructible_v<_Initiation>
                 && std::is_nothrow_constructible_v<__base_t, _Args...>)
        : __base_t(static_cast<_Args&&>(__args)...)
        , __receiver(static_cast<_Receiver&&>(__receiver))
        , __init(static_cast<_Initiation&&>(__init))
      {}

      STDEXEC_IMMOVABLE(state);

      constexpr void start() & noexcept
      {
        try
        {
          std::invoke(static_cast<_Initiation&&>(__init), *this);
        }
        catch (...)
        {
          if constexpr (!std::is_nothrow_invocable_v<_Initiation, state&>)
          {
            ::STDEXEC::set_error(static_cast<_Receiver&&>(__receiver), std::current_exception());
          }
        }
      }

      template <class _Tag, class... _Args>
      constexpr void __complete(_Tag __tag, _Args&&... __args) noexcept
      {
        __tag(static_cast<_Receiver&&>(__receiver), static_cast<_Args&&>(__args)...);
      }

     private:
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Receiver __receiver;
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Initiation __init;
    };

    template <class _Sender>
    using data_t = std::remove_cvref_t<::STDEXEC::__data_of<_Sender>>;

    template <class _Sender>
    using initiation_t = std::tuple_element_t<0, data_t<_Sender>>;

    template <class _Sender, std::size_t _Index>
    using data_element_t = decltype(std::get<_Index>(
      std::declval<::STDEXEC::__data_of<_Sender>>()));

    template <class _Signatures, class _Base>
    struct impl : ::STDEXEC::__sexpr_defaults
    {
      template <class _Sender, class _Receiver>
      using state_t = state<_Signatures, _Base, initiation_t<_Sender>, _Receiver>;

      template <class _Sender, std::size_t... _Indices>
      static consteval bool __base_constructible(std::index_sequence<_Indices...>)
      {
        return std::is_constructible_v<base_probe<_Signatures, _Base>,
                                       data_element_t<_Sender, _Indices + 1>...>;
      }

      template <class _Sender>
      static constexpr bool __base_constructible_v = __base_constructible<_Sender>(
        std::make_index_sequence<std::tuple_size_v<data_t<_Sender>> - 1>{});

      template <class _Sender, class _Receiver>
      static constexpr bool __connectable_v = []
      {
        if constexpr (!__base_constructible_v<_Sender>)
        {
          return false;
        }
        else
        {
          using __state_t = state_t<_Sender, _Receiver>;
          using __archetype_state_t =
            state_t<_Sender, ::STDEXEC::__receiver_archetype<::STDEXEC::env_of_t<_Receiver>>>;
          using __init_t = initiation_t<_Sender>;

          return std::is_invocable_v<__init_t, __state_t&>
              && std::is_invocable_v<__init_t, __archetype_state_t&>
              && (std::is_nothrow_invocable_v<__init_t, __state_t&>
                  == std::is_nothrow_invocable_v<__init_t, __archetype_state_t&>);
        }
      }();

      template <class _Sender, class _Receiver, std::size_t... _Indices>
      static constexpr bool __nothrow_connect(std::index_sequence<_Indices...>)
      {
        return std::is_nothrow_constructible_v<state_t<_Sender, _Receiver>,
                                               data_element_t<_Sender, 0>,
                                               _Receiver,
                                               data_element_t<_Sender, _Indices + 1>...>;
      }

      template <class _Sender, class _Receiver>
      static constexpr bool __nothrow_connect_v = __nothrow_connect<_Sender, _Receiver>(
        std::make_index_sequence<std::tuple_size_v<data_t<_Sender>> - 1>{});

      template <class _Sender, class _Receiver, std::size_t... _Indices>
      static constexpr auto
      __connect_impl(_Sender&& __sender, _Receiver __receiver, std::index_sequence<_Indices...>)
        noexcept(__nothrow_connect_v<_Sender, _Receiver>) -> state_t<_Sender, _Receiver>
      {
        auto& [__tag, __data] = __sender;
        return state_t<_Sender, _Receiver>(::STDEXEC::__forward_like<_Sender>(std::get<0>(__data)),
                                           static_cast<_Receiver&&>(__receiver),
                                           ::STDEXEC::__forward_like<_Sender>(
                                             std::get<_Indices + 1>(__data))...);
      }

      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        if constexpr (sizeof...(_Env) == 0)
        {
          return ::STDEXEC::transform_completion_signatures<
            _Signatures,
            ::STDEXEC::completion_signatures<::STDEXEC::set_error_t(std::exception_ptr)>>{};
        }
        else if constexpr (!__base_constructible_v<_Sender>)
        {
          return []<std::size_t... _Indices>(std::index_sequence<_Indices...>)
          {
            return ::STDEXEC::__throw_compile_time_error<
              BASE_MUST_BE_CONSTRUCTIBLE_FROM<_Base, data_element_t<_Sender, _Indices + 1>...>,
              ::STDEXEC::_WITH_SENDER_<_Sender>>();
          }(std::make_index_sequence<std::tuple_size_v<data_t<_Sender>> - 1>{});
        }
        else
        {
          using __archetype_state_t = state_t<_Sender, ::STDEXEC::__receiver_archetype<_Env...>>;
          using __init_t            = initiation_t<_Sender>;

          if constexpr (!std::is_invocable_v<__init_t, __archetype_state_t&>)
          {
            return ::STDEXEC::__throw_compile_time_error<
              INITIATION_MUST_BE_INVOCABLE_WITH_OPERATION_STATE<__init_t, __archetype_state_t>,
              ::STDEXEC::_WITH_SENDER_<_Sender>>();
          }
          else
          {
            using __additional_signatures_t = std::conditional_t<
              std::is_nothrow_invocable_v<__init_t, __archetype_state_t&>,
              ::STDEXEC::completion_signatures<>,
              ::STDEXEC::completion_signatures<::STDEXEC::set_error_t(std::exception_ptr)>>;
            return ::STDEXEC::transform_completion_signatures<_Signatures,
                                                              __additional_signatures_t>{};
          }
        }
      }

      static constexpr auto __connect =
        []<class _Sender, class _Receiver>(_Sender&& __sender, _Receiver __receiver) noexcept(
          __nothrow_connect_v<_Sender, _Receiver>)
        requires __connectable_v<_Sender, _Receiver>
      {
        return __connect_impl(static_cast<_Sender&&>(__sender),
                              static_cast<_Receiver&&>(__receiver),
                              std::make_index_sequence<std::tuple_size_v<data_t<_Sender>> - 1>{});
      };
    };
  }  // namespace detail::started_by

  template <::STDEXEC::__valid_completion_signatures _Signatures,
            class _Base = detail::started_by::default_base>
    requires std::is_class_v<_Base> && (!std::is_final_v<_Base>)
  struct started_by_t
  {
    template <::STDEXEC::__movable_value _Initiation, ::STDEXEC::__movable_value... _Args>
    [[nodiscard]]
    constexpr auto operator()(_Initiation&& __init, _Args&&... __args) const
      noexcept(::STDEXEC::__nothrow_decay_copyable<_Initiation, _Args...>)
        -> ::STDEXEC::__well_formed_sender auto
    {
      return ::STDEXEC::__sexpr{started_by_t{},
                                std::tuple(static_cast<_Initiation&&>(__init),
                                           static_cast<_Args&&>(__args)...)};
    }
  };

  template <::STDEXEC::__valid_completion_signatures _Signatures,
            class _Base = detail::started_by::default_base>
    requires std::is_class_v<_Base> && (!std::is_final_v<_Base>)
  inline constexpr started_by_t<_Signatures, _Base> started_by{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace STDEXEC
{
  template <class _Signatures, class _Base>
  struct __sexpr_impl<::exec::started_by_t<_Signatures, _Base>>
    : ::exec::detail::started_by::impl<_Signatures, _Base>
  {};
}  // namespace STDEXEC
