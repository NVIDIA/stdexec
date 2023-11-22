/*
 * Copyright (c) 2023 Runner-2019
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
#include "exec/trampoline_scheduler.hpp"
#include "exec/on.hpp"
#include "__detail/__manual_lifetime.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/concepts.hpp"
#include "stdexec/functional.hpp"
#include "trampoline_scheduler.hpp"
#include <concepts>

namespace exec {
  namespace __repeat_effect_until {
    using namespace stdexec;

    template <class _SourceId, class _ReceiverId>
    struct __receiver {
      struct __t;
    };

    template <class _SourceId, class _ReceiverId>
    struct __operation {
      using _Source = stdexec::__t<_SourceId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : stdexec::__immovable {
        using __id = __operation;
        using __receiver_t = stdexec::__t<__receiver<_SourceId, _ReceiverId>>;
        using __source_on_scheduler_sender =
          __call_result_t<stdexec::on_t, trampoline_scheduler, _Source &>;
        using __source_op_t = stdexec::connect_result_t<__source_on_scheduler_sender, __receiver_t>;

        STDEXEC_ATTRIBUTE((no_unique_address)) _Source __source_;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Receiver __rcvr_;
        __manual_lifetime<__source_op_t> __source_op_;
        trampoline_scheduler __sched_;

        template <class _Source2>
        __t(_Source2 &&__source, _Receiver __rcvr) noexcept(
          __nothrow_decay_copyable<_Source2> &&__nothrow_decay_copyable<_Receiver>
            &&nothrow_tag_invocable<stdexec::on_t, trampoline_scheduler, _Source>
              &&__nothrow_connectable<__source_on_scheduler_sender, __receiver_t>)
          : __source_((_Source2 &&) __source)
          , __rcvr_((_Receiver &&) __rcvr) {
          __source_op_.__construct_with([&] {
            return stdexec::connect(stdexec::on(__sched_, __source_), __receiver_t{this});
          });
        }

        friend void tag_invoke(stdexec::start_t, __t &__self) noexcept {
          stdexec::start(__self.__source_op_.__get());
        }
      };
    };

    template <class _SourceId, class _ReceiverId>
    struct __receiver<_SourceId, _ReceiverId>::__t {
      using receiver_concept = stdexec::receiver_t;
      using __id = __receiver;
      using _Source = stdexec::__t<_SourceId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __op_t = stdexec::__t<__operation<_SourceId, _ReceiverId>>;

      __op_t *__op_;

      explicit __t(__op_t *op) noexcept
        : __op_(op) {
        STDEXEC_ASSERT(__op_ != nullptr);
      }

      __t(__t &&__other) noexcept
        : __op_(std::exchange(__other.__op_, {})) {
        STDEXEC_ASSERT(__op_ != nullptr);
      }

      template <same_as<set_value_t> _Tag, same_as<__t> _Self, convertible_to<bool> _Done>
        requires __callable<set_value_t, _Receiver>
              && __callable<set_error_t, _Receiver, std::exception_ptr>
      friend void tag_invoke(_Tag, _Self &&__self, _Done &&__done_ish) noexcept {
        bool __done = static_cast<bool>(__done_ish); // BUGBUG potentially throwing.
        auto *__op = __self.__op_;

        // The following line causes the invalidation of __self.
        __op->__source_op_.__destroy();

        // If the sender completed with true, we're done
        if (__done) {
          stdexec::set_value((_Receiver &&) __op->__rcvr_);
        } else {
          try {
            auto &__source_op = __op->__source_op_.__construct_with([&]() {
              return stdexec::connect(stdexec::on(__op->__sched_, __op->__source_), __t{__op});
            });
            stdexec::start(__source_op);
          } catch (...) {
            stdexec::set_error((_Receiver &&) __op->__rcvr_, std::current_exception());
          }
        }
      }

      template <same_as<set_stopped_t> _Tag, same_as<__t> _Self>
        requires __callable<_Tag, _Receiver>
      friend void tag_invoke(_Tag, _Self &&__self) noexcept {
        auto *__op = __self.__op_;
        __op->__source_op_.__destroy();
        stdexec::set_stopped((_Receiver &&) __op->__rcvr_);
      }

      template <same_as<set_error_t> _Tag, same_as<__t> _Self, class _Error>
        requires __callable<_Tag, _Receiver, _Error>
      friend void tag_invoke(_Tag, _Self &&__self, _Error __error) noexcept {
        auto *__op = __self.__op_;
        __op->__source_op_.__destroy();
        stdexec::set_error((_Receiver &&) __op->__rcvr_, (_Error &&) __error);
      }

      friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t &__self) noexcept(
        __nothrow_callable<get_env_t, const _Receiver &>) {
        return get_env(__self.__op_->__rcvr_);
      }
    };

    template <class _SourceId>
    struct __sender {
      using _Source = stdexec::__t<_SourceId>;

      template <class _Receiver>
      using __op_t = stdexec::__t< __operation<_SourceId, stdexec::__id<_Receiver>>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t< __receiver<_SourceId, stdexec::__id<_Receiver>>>;

      struct __t {
        using sender_concept = stdexec::sender_t;
        using __id = __sender;
        STDEXEC_ATTRIBUTE((no_unique_address)) _Source __source_;

        template <class... Ts>
        using __value_t = stdexec::completion_signatures<>;

        template <class _Env>
        using __completion_signatures = //
          stdexec::make_completion_signatures<
            _Source &,
            _Env,
            stdexec::make_completion_signatures<
              stdexec::schedule_result_t<exec::trampoline_scheduler>,
              _Env,
              completion_signatures<set_error_t(std::exception_ptr), stdexec::set_value_t()>,
              __value_t>,
            __value_t>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self &&, _Env &&)
          -> __completion_signatures<_Env> {
          return {};
        }

        template <__decays_to<_Source> _Source2>
        explicit __t(_Source2 &&__source) noexcept(__nothrow_decay_copyable<_Source2>)
          : __source_((_Source2 &&) __source) {
        }

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sender_to<_Source &, __receiver_t<_Receiver>>
        friend __op_t<_Receiver> tag_invoke(connect_t, _Self &&__self, _Receiver __rcvr) noexcept(
          __nothrow_constructible_from<
            __op_t<_Receiver>,
            __copy_cvref_t<_Self, _Source>,
            _Receiver>) {
          return {((_Self &&) __self).__source_, (_Receiver &&) __rcvr};
        }

        friend auto tag_invoke(get_env_t, const __t &__self) //
          noexcept(__nothrow_callable<get_env_t, const _Source &>) -> env_of_t<const _Source &> {
          return get_env(__self.__source_);
        }
      };
    };

    template <class _Source>
    using __sender_t = __t< __sender<stdexec::__id<__decay_t<_Source>>>>;

    struct repeat_effect_until_t {
      template <sender _Source>
        requires tag_invocable<repeat_effect_until_t, _Source>
      auto operator()(_Source &&__source) const
        noexcept(nothrow_tag_invocable<repeat_effect_until_t, _Source>)
          -> tag_invoke_result_t<repeat_effect_until_t, _Source> {
        return tag_invoke(*this, (_Source &&) __source);
      }

      template <sender _Source>
      auto operator()(_Source &&__source) const
        noexcept(__nothrow_constructible_from< __sender_t<_Source>, _Source>)
          -> __sender_t<_Source> {
        return __sender_t<_Source>{(_Source &&) __source};
      }

      constexpr auto operator()() const -> __binder_back<repeat_effect_until_t> {
        return {{}, {}, {}};
      }
    };

  } // namespace __repeat_effect

  inline constexpr __repeat_effect_until::repeat_effect_until_t repeat_effect_until{};
} // namespace exec
