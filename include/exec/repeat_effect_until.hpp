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

#include "../stdexec/__detail/__basic_sender.hpp"
#include "../stdexec/__detail/__meta.hpp"
#include "../stdexec/__detail/__optional.hpp"
#include "../stdexec/execution.hpp"

#include "sequence.hpp"
#include "trampoline_scheduler.hpp"

#include <exception>
#include <type_traits>

namespace exec {
  namespace __repeat_effect {
    using namespace STDEXEC;

    template <class _Sender, class _Receiver>
    struct __repeat_effect_state;

    template <class _SenderId, class _ReceiverId>
    struct __receiver {
      using _Sender = STDEXEC::__t<_SenderId>;
      using _Receiver = STDEXEC::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        using receiver_concept = STDEXEC::receiver_t;
        __repeat_effect_state<_Sender, _Receiver> *__state_;

        template <class... _Args>
        void set_value(_Args &&...__args) noexcept {
          __state_->__complete(set_value_t(), static_cast<_Args &&>(__args)...);
        }

        template <class _Error>
        void set_error(_Error &&__err) noexcept {
          __state_->__complete(set_error_t(), static_cast<_Error &&>(__err));
        }

        void set_stopped() noexcept {
          __state_->__complete(set_stopped_t());
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return STDEXEC::get_env(__state_->__rcvr_);
        }
      };
    };

    template <typename _T, bool _B>
    concept __compile_time_bool_of = std::remove_cvref_t<_T>::value == _B;

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wtsan")

    template <class _Sender, class _Receiver>
    struct __repeat_effect_state {
      using __child_t = __decay_t<__data_of<_Sender>>;
      using __receiver_t = STDEXEC::__t<__receiver<__id<_Sender>, __id<_Receiver>>>;
      using __child_on_sched_sender_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler &>, __child_t &>;
      using __child_op_t = STDEXEC::connect_result_t<__child_on_sched_sender_t, __receiver_t>;

      explicit __repeat_effect_state(_Sender &&__sndr, _Receiver &&__rcvr)
        : __rcvr_(static_cast<_Receiver &&>(__rcvr))
        , __child_(STDEXEC::__get<1>(static_cast<_Sender &&>(__sndr))) {
        __connect();
      }

      void __connect() {
        __child_op_.__emplace_from(
          STDEXEC::connect,
          exec::sequence(STDEXEC::schedule(__sched_), __child_),
          __receiver_t{this});
      }

      void __destroy() noexcept {
        __child_op_.reset();
      }

      void __start() noexcept {
        STDEXEC::start(*__child_op_);
      }

      template <class _Tag, class... _Args>
      void __complete(_Tag, _Args &&...__args) noexcept {
        if constexpr (__std::same_as<_Tag, set_value_t>) {
          // If the sender completed with true, we're done
          STDEXEC_TRY {
            if constexpr ((__compile_time_bool_of<_Args, true> && ...)) {
              STDEXEC::set_value(static_cast<_Receiver &&>(__rcvr_));
              return;
            } else if constexpr (!(__compile_time_bool_of<_Args, false> && ...)) {
              const bool __done = (static_cast<bool>(static_cast<_Args &&>(__args)) && ...);
              if (__done) {
                STDEXEC::set_value(static_cast<_Receiver &&>(__rcvr_));
                return;
              }
            }
            __destroy();
            STDEXEC_TRY {
              __connect();
            }
            STDEXEC_CATCH_ALL {
              STDEXEC::set_error(static_cast<_Receiver &&>(__rcvr_), std::current_exception());
              return;
            }
            STDEXEC::start(*__child_op_);
          }
          STDEXEC_CATCH_ALL {
            __destroy();
            STDEXEC::set_error(static_cast<_Receiver &&>(__rcvr_), std::current_exception());
          }
        } else {
          _Tag()(static_cast<_Receiver &&>(__rcvr_), static_cast<_Args &&>(__args)...);
        }
      }

      _Receiver __rcvr_;
      __child_t __child_;
      STDEXEC::__optional<__child_op_t> __child_op_;
      trampoline_scheduler __sched_;
    };

    template <class _Sender, class _Receiver>
    __repeat_effect_state(_Sender &&, _Receiver) -> __repeat_effect_state<_Sender, _Receiver>;

    STDEXEC_PRAGMA_POP()

    template <
      __mstring _Where = "In repeat_effect_until: "_mstr,
      __mstring _What = "The input sender must send a single value that is convertible to bool"_mstr
    >
    struct _INVALID_ARGUMENT_TO_REPEAT_EFFECT_UNTIL_ { };

    template <class _Sender, class... _Args>
    using __values_t =
      // There's something funny going on with __if_c here. Use std::conditional_t instead. :-(
      std::conditional_t<
        ((sizeof...(_Args) == 1) && (__std::convertible_to<_Args, bool> && ...)),
        std::conditional_t<
          (__compile_time_bool_of<_Args, false> && ...),
          completion_signatures<>,
          completion_signatures<set_value_t()>
        >,
        __mexception<_INVALID_ARGUMENT_TO_REPEAT_EFFECT_UNTIL_<>, _WITH_PRETTY_SENDER_<_Sender>>
      >;

    template <class...>
    using __delete_set_value_t = completion_signatures<>;

    template <class _Sender, class... _Env>
    using __completions_t = STDEXEC::transform_completion_signatures<
      __completion_signatures_of_t<__decay_t<_Sender> &, _Env...>,
      STDEXEC::transform_completion_signatures<
        __completion_signatures_of_t<STDEXEC::schedule_result_t<exec::trampoline_scheduler>, _Env...>,
        __eptr_completion,
        __delete_set_value_t
      >,
      __mbind_front_q<__values_t, _Sender>::template __f
    >;

    struct __repeat_effect_tag { };

    struct __repeat_effect_until_tag { };

    struct __repeat_effect_until_impl : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        // TODO: port this to use constant evaluation
        return __completions_t<__data_of<_Sender>, _Env...>{};
      };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender &&__sndr, _Receiver &&__rcvr) {
          return __repeat_effect_state{
            static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr)};
        };

      static constexpr auto start = [](auto &__state) noexcept -> void {
        __state.__start();
      };
    };

    struct repeat_effect_until_t {
      template <sender _Sender>
      auto operator()(_Sender &&__sndr) const {
        return __make_sexpr<repeat_effect_until_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const {
        return __closure(*this);
      }

      template <class _Sender>
      static auto transform_sender(STDEXEC::set_value_t, _Sender &&__sndr, __ignore) {
        return STDEXEC::__apply(
          []<class _Child>(__ignore, __ignore, _Child __child) {
            return __make_sexpr<__repeat_effect_until_tag>(std::move(__child));
          },
          static_cast<_Sender &&>(__sndr));
      }
    };

    struct repeat_effect_t {
      struct _never {
        template <class... _Args>
        STDEXEC_ATTRIBUTE(host, device, always_inline)
        constexpr std::false_type operator()(_Args &&...) const noexcept {
          return {};
        }
      };

      template <sender _Sender>
      auto operator()(_Sender &&__sndr) const {
        return __make_sexpr<repeat_effect_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const {
        return __closure(*this);
      }

      template <class _Sender>
      static auto transform_sender(STDEXEC::set_value_t, _Sender &&__sndr, __ignore) {
        return STDEXEC::__apply(
          [](__ignore, __ignore, auto __child) {
            return repeat_effect_until_t{}(STDEXEC::then(std::move(__child), _never{}));
          },
          static_cast<_Sender &&>(__sndr));
      }
    };
  } // namespace __repeat_effect

  using __repeat_effect::repeat_effect_until_t;
  inline constexpr repeat_effect_until_t repeat_effect_until{};

  using __repeat_effect::repeat_effect_t;
  inline constexpr repeat_effect_t repeat_effect{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::__repeat_effect::__repeat_effect_until_tag>
    : exec::__repeat_effect::__repeat_effect_until_impl { }; // namespace STDEXEC

  template <>
  struct __sexpr_impl<exec::repeat_effect_until_t> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      static_assert(sender_expr_for<_Sender, exec::repeat_effect_until_t>);
      using __sndr_t = __detail::__transform_sender_result_t<
        exec::repeat_effect_until_t,
        set_value_t,
        _Sender,
        env<>
      >;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
    }
  };
} // namespace STDEXEC
