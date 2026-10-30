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

#include "../stdexec/__detail/__manual_lifetime.hpp"
#include "../stdexec/execution.hpp"

#include <atomic>
#include <concepts>
#include <cstdint>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace experimental::execution
{
  struct stopped_by_t;

  namespace detail::stopped_by
  {
    struct no_classifier
    {};

    template <class _Factory, class _Operation>
    struct STOP_FACTORY_MUST_BE_NOTHROW_INVOCABLE
    {};

    template <class _StopSender, class _Env>
    struct STOP_FACTORY_MUST_RETURN_A_SENDER_IN
    {};

    template <class _StopSender, class _Env>
    struct STOP_SENDER_MUST_BE_NOTHROW_CONNECTABLE
    {};

    template <class _StopSender, class _Signatures>
    struct STOP_SENDER_MUST_SEND_EXACTLY_NULLARY_SET_VALUE
    {};

    template <class _Classifier, class _Signatures>
    struct CLASSIFIER_MUST_BE_NOTHROW_INVOCABLE_FOR_EACH_COMPLETION
    {};

    template <class _Classifier, class _Signature>
    struct classifier_for_signature;

    template <class _Classifier, class _Tag, class... _Args>
    struct classifier_for_signature<_Classifier, _Tag(_Args...)>
      : std::bool_constant<
          std::same_as<_Tag, ::STDEXEC::set_stopped_t>
          || std::is_nothrow_invocable_r_v<
            bool,
            _Classifier&&,
            _Tag,
            std::add_lvalue_reference_t<std::add_const_t<std::remove_reference_t<_Args>>>...>>
    {};

    template <class _Classifier, class _Signatures>
    struct classifier_for;

    template <class _Classifier, class... _Signatures>
    struct classifier_for<_Classifier, ::STDEXEC::completion_signatures<_Signatures...>>
      : std::bool_constant<(classifier_for_signature<_Classifier, _Signatures>::value && ...)>
    {};

    template <class _Classifier, class _Signatures>
    inline constexpr bool classifier_for_v = classifier_for<_Classifier, _Signatures>::value;

    template <class _Sender>
    using data_t = std::remove_cvref_t<::STDEXEC::__data_of<_Sender>>;

    template <class _Sender>
    using factory_t = std::tuple_element_t<0, data_t<_Sender>>;

    template <class _Sender>
    using classifier_t = std::tuple_element_t<1, data_t<_Sender>>;

    template <class _Sender>
    inline constexpr bool has_classifier_v = !std::same_as<classifier_t<_Sender>, no_classifier>;

    template <class _Sender, class _ChildSignatures>
    using stored_signatures_t = std::conditional_t<
      has_classifier_v<_Sender>,
      ::STDEXEC::transform_completion_signatures<
        _ChildSignatures,
        ::STDEXEC::completion_signatures<::STDEXEC::set_stopped_t()>>,
      _ChildSignatures>;

    template <class _Sender, class _ChildSignatures>
    using storage_t = ::STDEXEC::storage_for_completion_signatures<
      stored_signatures_t<_Sender, _ChildSignatures>>;

    template <class _Env>
    inline constexpr bool stoppable_v =
      !::STDEXEC::unstoppable_token<::STDEXEC::stop_token_of_t<_Env>>;

    template <class _Child, class _Factory, class _Classifier, class _Receiver>
    class state
    {
      static constexpr bool has_classifier = !std::same_as<_Classifier, no_classifier>;

      struct receiver
      {
        using receiver_concept = ::STDEXEC::receiver_tag;

        template <class... _Values>
        constexpr void set_value(_Values&&... __values) && noexcept
        {
          __state.complete(::STDEXEC::set_value, static_cast<_Values&&>(__values)...);
        }

        template <class _Error>
        constexpr void set_error(_Error&& __error) && noexcept
        {
          __state.complete(::STDEXEC::set_error, static_cast<_Error&&>(__error));
        }

        constexpr void set_stopped() && noexcept
        {
          __state.complete(::STDEXEC::set_stopped);
        }

        constexpr auto get_env() const noexcept -> ::STDEXEC::env_of_t<_Receiver>
        {
          return ::STDEXEC::get_env(__state.__receiver);
        }

        state& __state;
      };

      struct stop_receiver
      {
        using receiver_concept = ::STDEXEC::receiver_tag;

        constexpr void set_value() && noexcept
        {
          // Save the back-pointer before destroying the operation state that owns *this.
          auto& __saved_state = __state;
          __saved_state.__stop_operation.__destroy();
          if (__saved_state.release_ref()) [[unlikely]]
          {
            __saved_state.complete_stored();
          }
        }

        constexpr auto get_env() const noexcept -> ::STDEXEC::env_of_t<_Receiver>
        {
          return ::STDEXEC::get_env(__state.__receiver);
        }

        state& __state;
      };

      using child_operation_t = ::STDEXEC::connect_result_t<_Child, receiver>;
      using stop_sender_t     = std::invoke_result_t<_Factory, child_operation_t&>;
      using stop_operation_t  = ::STDEXEC::connect_result_t<stop_sender_t, stop_receiver>;
      using child_signatures_t =
        ::STDEXEC::completion_signatures_of_t<_Child, ::STDEXEC::env_of_t<_Receiver>>;
      using stored_signatures_t =
        std::conditional_t<has_classifier,
                           ::STDEXEC::transform_completion_signatures<
                             child_signatures_t,
                             ::STDEXEC::completion_signatures<::STDEXEC::set_stopped_t()>>,
                           child_signatures_t>;
      using storage_t = ::STDEXEC::storage_for_completion_signatures<stored_signatures_t>;

      struct on_stop_request
      {
        constexpr void operator()() && noexcept
        {
          __state.request_stop();
        }

        state& __state;
      };

      using stop_token_t    = ::STDEXEC::stop_token_of_t<::STDEXEC::env_of_t<_Receiver>>;
      using stop_callback_t = ::STDEXEC::stop_callback_for_t<stop_token_t, on_stop_request>;

      constexpr bool release_ref() noexcept
      {
        // Each participant publishes its work before dropping its reference.
        return __outstanding.fetch_sub(1, std::memory_order_release) == 1;
      }

      constexpr void complete_stored() noexcept
      {
        // The last participant acquires those publications and delivers the stored
        // completion.
        std::atomic_thread_fence(std::memory_order_acquire);
        __stop_callback.__destroy();
        std::move(__storage).complete(static_cast<_Receiver&&>(__receiver));
      }

      constexpr void request_stop() noexcept
      {
        if (__outstanding.fetch_add(1, std::memory_order_relaxed) == 0) [[unlikely]]
        {
          // Completion has already been claimed. Destroying the callback synchronizes
          // with this invocation, so it is enough to let the invocation return.
          return;
        }

        auto& __operation = __stop_operation.__construct_from(
          ::STDEXEC::connect,
          std::invoke(static_cast<_Factory&&>(__factory), __child_operation),
          stop_receiver{*this});
        ::STDEXEC::start(__operation);
      }

      template <class _Tag, class... _Args>
      constexpr void complete_unclassified(_Tag __tag, _Args&&... __args) noexcept
      {
        if (__outstanding.load(std::memory_order_relaxed) == 1) [[likely]]
        {
          // Usually no stop was requested. Retire the callback and, after checking that
          // it did not race with us, forward the completion without storing it.
          std::atomic_thread_fence(std::memory_order_acquire);
          __stop_callback.__destroy();
          if (__outstanding.load(std::memory_order_acquire) == 1) [[likely]]
          {
            __tag(static_cast<_Receiver&&>(__receiver), static_cast<_Args&&>(__args)...);
            return;
          }
        }

        __storage.arrive(__tag, static_cast<_Args&&>(__args)...);
        if (release_ref()) [[unlikely]]
        {
          complete_stored();
        }
      }

     public:
      using operation_state_concept = ::STDEXEC::operation_state_tag;

      constexpr explicit state(_Child&&    __child,
                               _Factory    __factory,
                               _Classifier __classifier,
                               _Receiver   __receiver)
        noexcept(std::is_nothrow_move_constructible_v<_Factory>
                 && std::is_nothrow_move_constructible_v<_Classifier>
                 && ::STDEXEC::__nothrow_connectable<_Child, receiver>)
        : __receiver(static_cast<_Receiver&&>(__receiver))
        , __factory(static_cast<_Factory&&>(__factory))
        , __classifier(static_cast<_Classifier&&>(__classifier))
        , __child_operation(::STDEXEC::connect(static_cast<_Child&&>(__child), receiver{*this}))
      {}

      STDEXEC_IMMOVABLE(state);

      constexpr void start() & noexcept
      {
        ::STDEXEC::start(__child_operation);
        __stop_callback.__construct(::STDEXEC::get_stop_token(::STDEXEC::get_env(__receiver)),
                                    on_stop_request{*this});
        if (release_ref()) [[unlikely]]
        {
          complete_stored();
        }
      }

      template <class _Tag, class... _Args>
      constexpr void complete(_Tag __tag, _Args&&... __args) noexcept
      {
        if constexpr (has_classifier && !std::same_as<_Tag, ::STDEXEC::set_stopped_t>)
        {
          if (bool(std::invoke(static_cast<_Classifier&&>(__classifier),
                               _Tag{},
                               std::as_const(__args)...)))
          {
            __storage.arrive(::STDEXEC::set_stopped);
            if (release_ref()) [[likely]]
            {
              complete_stored();
            }
            return;
          }
        }

        complete_unclassified(__tag, static_cast<_Args&&>(__args)...);
      }

     private:
      _Receiver __receiver;
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Factory __factory;
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Classifier __classifier;
      ::STDEXEC::__manual_lifetime<stop_operation_t> __stop_operation;
      storage_t                                      __storage;
      ::STDEXEC::__manual_lifetime<stop_callback_t>  __stop_callback;
      child_operation_t                              __child_operation;
      // One reference for start() and one for the child operation. A stop callback
      // temporarily adds a third for the asynchronous stop operation. Keeping the
      // byte-sized counter in the low-alignment tail gives a compact child operation
      // the best chance of reducing the enclosing operation state's size.
      std::atomic<std::uint8_t> __outstanding{2};
    };

    template <class _Sender, class _Receiver>
    using state_t =
      state<::STDEXEC::__child_of<_Sender>, factory_t<_Sender>, classifier_t<_Sender>, _Receiver>;

    struct impl : ::STDEXEC::__sexpr_defaults
    {
      template <class _Sender, class... _Env>
      static consteval auto __get_completion_signatures()
      {
        if constexpr (sizeof...(_Env) == 0)
        {
          using __child_t = ::STDEXEC::__child_of<_Sender>;
          auto __child_signatures = ::STDEXEC::get_completion_signatures<__child_t>();
          using __child_signatures_t = decltype(__child_signatures);

          if constexpr (!::STDEXEC::__valid_completion_signatures<__child_signatures_t>)
          {
            return __child_signatures;
          }
          else
          {
            // Without an environment, conservatively assume that storage is needed.
            using __storage_t = storage_t<_Sender, __child_signatures_t>;
            return __storage_t::get_completion_signatures();
          }
        }
        else
        {
          using __env_t   = std::tuple_element_t<0, std::tuple<_Env...>>;
          using __child_t = ::STDEXEC::__child_of<_Sender>;
          auto __child_signatures =
            ::STDEXEC::get_completion_signatures<__child_t, _Env...>();
          using __child_signatures_t = decltype(__child_signatures);

          if constexpr (!::STDEXEC::__valid_completion_signatures<__child_signatures_t>)
          {
            return __child_signatures;
          }
          else if constexpr (!stoppable_v<__env_t>)
          {
            return __child_signatures_t{};
          }
          else
          {
            using __factory_t            = factory_t<_Sender>;
            using __classifier_t         = classifier_t<_Sender>;
            using __archetype_receiver_t = ::STDEXEC::__receiver_archetype<_Env...>;
            using __child_operation_t =
              ::STDEXEC::connect_result_t<__child_t, __archetype_receiver_t>;

            if constexpr (!std::is_nothrow_invocable_v<__factory_t, __child_operation_t&>)
            {
              return ::STDEXEC::__throw_compile_time_error<
                STOP_FACTORY_MUST_BE_NOTHROW_INVOCABLE<__factory_t, __child_operation_t>,
                ::STDEXEC::_WITH_SENDER_<_Sender>>();
            }
            else
            {
              using __stop_sender_t = std::invoke_result_t<__factory_t, __child_operation_t&>;

              if constexpr (!::STDEXEC::sender_in<__stop_sender_t, _Env...>)
              {
                return ::STDEXEC::__throw_compile_time_error<
                  STOP_FACTORY_MUST_RETURN_A_SENDER_IN<__stop_sender_t, __env_t>,
                  ::STDEXEC::_WITH_SENDER_<_Sender>>();
              }
              else if constexpr (!::STDEXEC::__nothrow_connectable<__stop_sender_t,
                                                                   __archetype_receiver_t>)
              {
                return ::STDEXEC::__throw_compile_time_error<
                  STOP_SENDER_MUST_BE_NOTHROW_CONNECTABLE<__stop_sender_t, __env_t>,
                  ::STDEXEC::_WITH_SENDER_<_Sender>>();
              }
              else
              {
                using __stop_signatures_t =
                  ::STDEXEC::completion_signatures_of_t<__stop_sender_t, _Env...>;

                if constexpr (!std::same_as<
                                __stop_signatures_t,
                                ::STDEXEC::completion_signatures<::STDEXEC::set_value_t()>>)
                {
                  return ::STDEXEC::__throw_compile_time_error<
                    STOP_SENDER_MUST_SEND_EXACTLY_NULLARY_SET_VALUE<__stop_sender_t,
                                                                    __stop_signatures_t>,
                    ::STDEXEC::_WITH_SENDER_<_Sender>>();
                }
                else if constexpr (has_classifier_v<_Sender>
                                   && !classifier_for_v<__classifier_t, __child_signatures_t>)
                {
                  return ::STDEXEC::__throw_compile_time_error<
                    CLASSIFIER_MUST_BE_NOTHROW_INVOCABLE_FOR_EACH_COMPLETION<__classifier_t,
                                                                             __child_signatures_t>,
                    ::STDEXEC::_WITH_SENDER_<_Sender>>();
                }
                else
                {
                  using __storage_t = storage_t<_Sender, __child_signatures_t>;
                  return __storage_t::get_completion_signatures();
                }
              }
            }
          }
        }
      }

      static constexpr auto __connect =
        []<class _Sender, class _Receiver>(_Sender&& __sender, _Receiver __receiver) noexcept(
          []() constexpr
          {
            if constexpr (stoppable_v<::STDEXEC::env_of_t<_Receiver>>)
            {
              return std::is_nothrow_constructible_v<state_t<_Sender, _Receiver>,
                                                     ::STDEXEC::__child_of<_Sender>,
                                                     factory_t<_Sender>,
                                                     classifier_t<_Sender>,
                                                     _Receiver>;
            }
            else
            {
              return ::STDEXEC::__nothrow_connectable<::STDEXEC::__child_of<_Sender>, _Receiver>;
            }
          }())
      {
        auto& [__tag, __data, __child] = __sender;
        if constexpr (stoppable_v<::STDEXEC::env_of_t<_Receiver>>)
        {
          auto& [__factory, __classifier] = __data;
          return state_t<_Sender, _Receiver>(::STDEXEC::__forward_like<_Sender>(__child),
                                             ::STDEXEC::__forward_like<_Sender>(__factory),
                                             ::STDEXEC::__forward_like<_Sender>(__classifier),
                                             static_cast<_Receiver&&>(__receiver));
        }
        else
        {
          return ::STDEXEC::connect(::STDEXEC::__forward_like<_Sender>(__child),
                                    static_cast<_Receiver&&>(__receiver));
        }
      };
    };
  }  // namespace detail::stopped_by

  struct stopped_by_t
  {
    template <::STDEXEC::sender          _Sender,
              ::STDEXEC::__movable_value _Factory,
              ::STDEXEC::__movable_value _Classifier>
    [[nodiscard]]
    constexpr auto
    operator()(_Sender&& __sender, _Factory&& __factory, _Classifier&& __classifier) const
      noexcept(::STDEXEC::__nothrow_decay_copyable<_Sender, _Factory, _Classifier>)
        -> ::STDEXEC::__well_formed_sender auto
    {
      return ::STDEXEC::__sexpr{stopped_by_t{},
                                std::tuple(static_cast<_Factory&&>(__factory),
                                           static_cast<_Classifier&&>(__classifier)),
                                static_cast<_Sender&&>(__sender)};
    }

    template <::STDEXEC::sender _Sender, ::STDEXEC::__movable_value _Factory>
    [[nodiscard]]
    constexpr auto operator()(_Sender&& __sender, _Factory&& __factory) const
      noexcept(::STDEXEC::__nothrow_decay_copyable<_Sender, _Factory>)
        -> ::STDEXEC::__well_formed_sender auto
    {
      return (*this)(static_cast<_Sender&&>(__sender),
                     static_cast<_Factory&&>(__factory),
                     detail::stopped_by::no_classifier{});
    }

    template <::STDEXEC::__movable_value _Factory>
    [[nodiscard]]
    constexpr auto
    operator()(_Factory&& __factory) const noexcept(::STDEXEC::__nothrow_decay_copyable<_Factory>)
    {
      return ::STDEXEC::__closure(*this, static_cast<_Factory&&>(__factory));
    }

    template <::STDEXEC::__movable_value _Factory, ::STDEXEC::__movable_value _Classifier>
      requires(!::STDEXEC::sender<std::remove_cvref_t<_Factory>>)
    [[nodiscard]]
    constexpr auto operator()(_Factory&& __factory, _Classifier&& __classifier) const
      noexcept(::STDEXEC::__nothrow_decay_copyable<_Factory, _Classifier>)
    {
      return ::STDEXEC::__closure(*this,
                                  static_cast<_Factory&&>(__factory),
                                  static_cast<_Classifier&&>(__classifier));
    }
  };

  inline constexpr stopped_by_t stopped_by{};
}  // namespace experimental::execution

namespace exec = experimental::execution;

namespace STDEXEC
{
  template <>
  struct __sexpr_impl<::exec::stopped_by_t> : ::exec::detail::stopped_by::impl
  {};
}  // namespace STDEXEC
