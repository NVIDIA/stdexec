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
#include "../stdexec/concepts.hpp"
#include "../stdexec/functional.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__basic_sender.hpp"

#include "on.hpp"
#include "trampoline_scheduler.hpp"
#include "__detail/__manual_lifetime.hpp"

#include <atomic>
#include <concepts>

namespace exec {
  namespace __repeat_effect_until {
    using namespace stdexec;

    template <class _Sender, class _Receiver>
    struct __repeat_effect_state;

    template <class _SenderId, class _ReceiverId>
    struct __receiver {
      using _Sender = stdexec::__t<_SenderId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        using receiver_concept = stdexec::receiver_t;
        __repeat_effect_state<_Sender, _Receiver> *__state_;

        template <__completion_tag _Tag, class... _Args>
        friend void tag_invoke(_Tag, __t &&__self, _Args &&...__args) noexcept {
          __self.__state_->__complete(_Tag(), (_Args &&) __args...);
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t &__self) noexcept {
          return get_env(__self.__state_->__receiver());
        }
      };
    };

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wtsan")

    template <class _Sender, class _Receiver>
    struct __repeat_effect_state : stdexec::__enable_receiver_from_this<_Sender, _Receiver> {
      using __child_t = __decay_t<__data_of<_Sender>>;
      using __receiver_t = stdexec::__t<__receiver<__id<_Sender>, __id<_Receiver>>>;
      using __child_on_sched_sender_t = __result_of<stdexec::on, trampoline_scheduler, __child_t &>;
      using __child_op_t = stdexec::connect_result_t<__child_on_sched_sender_t, __receiver_t>;

      __child_t __child_;
      std::atomic_flag __started_{};
      __manual_lifetime<__child_op_t> __child_op_;
      trampoline_scheduler __sched_;

      __repeat_effect_state(_Sender &&__sndr, _Receiver &)
        : __child_(__sexpr_apply((_Sender &&) __sndr, __detail::__get_data())) {
        __connect();
      }

      ~__repeat_effect_state() {
        if (!__started_.test(std::memory_order_acquire)) {
          std::atomic_thread_fence(std::memory_order_release);
          // TSan does not support std::atomic_thread_fence, so we
          // need to use the TSan-specific __tsan_release instead:
          STDEXEC_TSAN(__tsan_release(&__started_));
          __child_op_.__destroy();
        }
      }

      void __connect() {
        __child_op_.__construct_with([this] {
          return stdexec::connect(stdexec::on(__sched_, __child_), __receiver_t{this});
        });
      }

      void __start() noexcept {
        const bool __already_started [[maybe_unused]] =
          __started_.test_and_set(std::memory_order_relaxed);
        STDEXEC_ASSERT(!__already_started);
        stdexec::start(__child_op_.__get());
      }

      template <class _Tag, class... _Args>
      void __complete(_Tag, _Args &&...__args) noexcept {
        __child_op_.__destroy();
        if constexpr (same_as<_Tag, set_value_t>) {
          // If the sender completed with true, we're done
          try {
            const bool __done = (static_cast<bool>(__args) && ...);
            if (__done) {
              stdexec::set_value((_Receiver &&) this->__receiver());
            } else {
              __connect();
              stdexec::start(__child_op_.__get());
            }
          } catch (...) {
            stdexec::set_error((_Receiver &&) this->__receiver(), std::current_exception());
          }
        } else {
          _Tag()((_Receiver &&) this->__receiver(), (_Args &&) __args...);
        }
      }
    };

    STDEXEC_PRAGMA_POP()

    template <
      __mstring _Where = "In repeat_effect_until: "__csz,
      __mstring _What = "The input sender must send a single value that is convertible to bool"__csz>
    struct _INVALID_ARGUMENT_TO_REPEAT_EFFECT_UNTIL_ { };

    template <class _Sender, class... _Args>
    using __values_t = //
      // There's something funny going on with __if_c here. Use std::conditional_t instead. :-(
      std::conditional_t<
        ((sizeof...(_Args) == 1) && (convertible_to<_Args, bool> && ...)),
        completion_signatures<>,
        __mexception<_INVALID_ARGUMENT_TO_REPEAT_EFFECT_UNTIL_<>, _WITH_SENDER_<_Sender>>>;

    template <class _Sender, class _Env>
    using __completions_t = //
      stdexec::__try_make_completion_signatures<
        __decay_t<_Sender> &,
        _Env,
        stdexec::__try_make_completion_signatures<
          stdexec::schedule_result_t<exec::trampoline_scheduler>,
          _Env,
          __with_exception_ptr>,
        __mbind_front_q<__values_t, _Sender>>;

    struct __repeat_effect_until_tag { };

    struct __repeat_effect_until_impl : __sexpr_defaults {
      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender &&, _Env &&) noexcept {
          return __completions_t<__data_of<_Sender>, _Env>{};
        };

      static constexpr auto get_state = //
        []<class _Sender, class _Receiver>(_Sender &&__sndr, _Receiver &__rcvr) {
          return __repeat_effect_state{std::move(__sndr), __rcvr};
        };

      static constexpr auto start = //
        [](auto &__state, __ignore) noexcept -> void {
        __state.__start();
      };
    };

    struct repeat_effect_until_t {
      template <sender _Sender>
      auto operator()(_Sender &&__sndr) const {
        auto __domain = __get_early_domain(__sndr);
        return stdexec::transform_sender(
          __domain, __make_sexpr<repeat_effect_until_t>({}, (_Sender &&) __sndr));
      }

      constexpr auto operator()() const -> __binder_back<repeat_effect_until_t> {
        return {{}, {}, {}};
      }

      template <class _Sender>
      auto transform_sender(_Sender &&__sndr, __ignore) {
        return __sexpr_apply(
          (_Sender &&) __sndr, []<class _Child>(__ignore, __ignore, _Child __child) {
            return __make_sexpr<__repeat_effect_until_tag>(std::move(__child));
          });
      }
    };
  } // namespace __repeat_effect_until

  using __repeat_effect_until::repeat_effect_until_t;
  inline constexpr repeat_effect_until_t repeat_effect_until{};
} // namespace exec

namespace stdexec {
  template <>
  struct __sexpr_impl<exec::__repeat_effect_until::__repeat_effect_until_tag>
    : exec::__repeat_effect_until::__repeat_effect_until_impl { };
}
