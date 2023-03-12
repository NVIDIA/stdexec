/*
 * Copyright (c) 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "../stdexec/execution.hpp"
#include "__detail/__manual_lifetime.hpp"
#include "stdexec/concepts.hpp"
#include <concepts>

namespace exec {
  namespace __repeat_effect_until {
    using namespace stdexec;

    template <typename _SourceId, typename _ReceiverId>
    struct __receiver {
      struct __t;
    };

    template <typename _SourceId, typename _ReceiverId>
    struct __operation {
      using _Source = stdexec::__t<_SourceId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : stdexec::__immovable {
        using __id = __operation;
        using __receiver_t = stdexec::__t<__receiver<_SourceId, _ReceiverId>>;
        using __source_op_t = stdexec::connect_result_t<_Source, __receiver_t>;

        [[no_unique_address]] _Source __source_;
        [[no_unique_address]] _Receiver __receiver_;
        __manual_lifetime<__source_op_t> __source_op_;

        template <typename _Source2, typename _Receiver2>
        __t(_Source2 &&__source, _Receiver2 &&__receiver) noexcept(
          std::is_nothrow_constructible_v<_Source, _Source2>
            &&std::is_nothrow_constructible_v<_Receiver, _Receiver2>
              &&__nothrow_connectable<_Source, __receiver_t>)
          : __source_((_Source2 &&) __source)
          , __receiver_((_Receiver2 &&) __receiver) {
          __source_op_.__construct_with([&] {
            return stdexec::connect((_Source2 &&) __source_, __receiver_t{this});
          });
        }

        ~__t() {
          __source_op_.__destruct();
        }

        friend void tag_invoke(stdexec::start_t, __t &__self) noexcept {
          stdexec::start(__self.__source_op_.__get());
        }
      };
    };

    template <typename _SourceId, typename _ReceiverId>
    struct __receiver<_SourceId, _ReceiverId>::__t {
      using __id = __receiver;
      using _Source = stdexec::__t<_SourceId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __op_t = stdexec::__t<__operation<_SourceId, _ReceiverId>>;

      __op_t *__op_;

      explicit __t(__op_t *op) noexcept
        : __op_(op) {
      }

      __t(__t &&__other) noexcept
        : __op_(std::exchange(__other.__op_, {})) {
      }

      template <__decays_to<__t> _Self, convertible_to<bool> _Arg>
      friend void tag_invoke(set_value_t, _Self &&__self, _Arg &&__flag) noexcept {
        __self.__op_->__source_op_.__destruct();

        if constexpr (__nothrow_connectable<_Source &, __t>) {
          // call __predicate and complete with void if it returns true
          if (__flag) {
            stdexec::set_value((_Receiver &&) __self.__op_->__receiver_);
            return;
          }

          auto &__source_op = __self.__op_->__source_op_.__construct_with([&]() {
            return stdexec::connect((_Source &&) __self.__op_->__source_, __t{__self.__op_});
          });
          stdexec::start(__source_op);
        } else {
          try {
            // call __predicate and complete with void if it returns true
            if (__flag) {
              stdexec::set_value((_Receiver &&) __self.__op_->__receiver_);
              return;
            }

            auto &__source_op = __self.__op_->__source_op_.__construct_with([&]() {
              return stdexec::connect((_Source &&) __self.__op_->__source_, __t{__self.__op_});
            });
            stdexec::start(__source_op);
          } catch (...) {
            stdexec::set_error((_Receiver &&) __self.__op_->__receiver_, std::current_exception());
          }
        }
      }

      template <__decays_to<__t> _Self>
      friend void tag_invoke(set_stopped_t, _Self &&__self) noexcept {
        stdexec::set_stopped((_Receiver &&) __self.__op_->__receiver_);
      }

      template <__decays_to<__t> _Self, typename _Error>
      friend void tag_invoke(set_error_t, _Self &&__self, _Error &&__error) noexcept {
        stdexec::set_error((_Receiver &&) __self.__op_->__receiver_, (_Error &&) __error);
      }

      friend env_of_t<const _Receiver &> tag_invoke(get_env_t, const __t &__self) noexcept {
        return get_env(__self.__op_->__receiver_);
      }
    };

    template <typename _SourceId>
    struct __sender {
      using _Source = stdexec::__t<_SourceId>;

      template <class _Receiver>
      using __op_t =
        stdexec::__t< __operation<_SourceId, stdexec::__id<std::remove_cvref_t<_Receiver>>>>;

      template <class _Receiver>
      using __receiver_t =
        stdexec::__t< __receiver<_SourceId, stdexec::__id<std::remove_cvref_t<_Receiver>>>>;

      struct __t {
        using __id = __sender;
        [[no_unique_address]] _Source __source_;

        template <class... Ts>
        using _Value = stdexec::completion_signatures<stdexec::set_value_t()>;

        template <class _Self, class _Env>
        using __completion_signatures = //
          stdexec::make_completion_signatures<
            __copy_cvref_t<_Self, _Source>,
            _Env,
            completion_signatures<set_error_t(std::exception_ptr)>,
            _Value >;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self &&, _Env)
          -> __completion_signatures<_Self, _Env> {
          return {};
        }

        template <typename _Source2>
        explicit __t(_Source2 &&__source) noexcept(
          std::is_nothrow_constructible_v<_Source, _Source2>)
          : __source_((_Source2 &&) __source) {
        }

        template <__decays_to<__t> _Sender, receiver _Receiver>
          requires sender_to<_Source, __receiver_t<_Receiver>>
        friend __op_t<_Receiver> tag_invoke(connect_t, _Sender &&__s, _Receiver &&__r) noexcept {
          return {((_Sender &&) __s).__source_, (_Receiver &&) __r};
        }

        friend env_of_t<const _Source &> tag_invoke(get_env_t, const __t &__self) noexcept {
          return get_env(__self.__source_);
        }
      };
    };

    template <typename _Source>
    using __sender_t = __t< __sender<stdexec::__id<std::remove_cvref_t<_Source>>>>;

    inline const struct repeat_effect_until_t {
      template <sender _Source>
      auto operator()(_Source &&__source) const
        noexcept(nothrow_tag_invocable<repeat_effect_until_t, _Source>)
          -> tag_invoke_result_t<repeat_effect_until_t, _Source> {
        return tag_invoke(*this, (_Source &&) __source);
      }

      template <sender _Source>
        requires(!tag_invocable<repeat_effect_until_t, _Source>)
      auto operator()(_Source &&__source) const
        noexcept(std::is_nothrow_constructible_v< __sender_t<_Source>, _Source>)
          -> __sender_t<_Source> {
        return __sender_t<_Source>{(_Source &&) __source};
      }

      constexpr auto operator()() const -> __binder_back<repeat_effect_until_t> {
        return {{}, {}, {}};
      }
    } repeat_effect_until{};

  } // namespace __repeat_effect

  inline constexpr __repeat_effect_until::repeat_effect_until_t repeat_effect_until{};
} // namespace exec