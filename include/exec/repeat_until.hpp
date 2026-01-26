/*
 * Copyright (c) 2023 Runner-2019
 * Copyright (c) 2026 NVIDIA Corporation
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
  namespace __repeat {
    using namespace STDEXEC;

    template <class _Receiver>
    struct __repeat_state_base {
      constexpr explicit __repeat_state_base(_Receiver &&__rcvr)
        : __rcvr_{static_cast<_Receiver &&>(__rcvr)} {
      }

      virtual constexpr void __cleanup() noexcept = 0;
      virtual constexpr void __repeat() noexcept = 0;

      _Receiver __rcvr_;
      trampoline_scheduler __sched_;
    };

    template <class _Bool, bool _Expected>
    concept __bool_constant = __decay_t<_Bool>::value == _Expected;

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = STDEXEC::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        using receiver_concept = STDEXEC::receiver_t;

        template <class... _Bools>
        constexpr void set_value(_Bools &&...__bools) noexcept {
          if constexpr ((__bool_constant<_Bools, true> && ...)) {
            // Always done:
            __state_->__cleanup();
            STDEXEC::set_value(std::move(__state_->__rcvr_));
          } else if constexpr ((__bool_constant<_Bools, false> && ...)) {
            // Never done:
            __state_->__repeat();
          } else {
            // Mixed results:
            constexpr bool __is_nothrow = noexcept(
              (static_cast<bool>(static_cast<_Bools &&>(__bools)) && ...));
            STDEXEC_TRY {
              // If the child sender completed with true, we're done
              const bool __done = (static_cast<bool>(static_cast<_Bools &&>(__bools)) && ...);
              if (__done) {
                __state_->__cleanup();
                STDEXEC::set_value(std::move(__state_->__rcvr_));
              } else {
                __state_->__repeat();
              }
            }
            STDEXEC_CATCH_ALL {
              if constexpr (!__is_nothrow) {
                __state_->__cleanup();
                STDEXEC::set_error(
                  std::move(__state_->__rcvr_), std::current_exception());
              }
            }
          }
        }

        template <class _Error>
        constexpr void set_error(_Error &&__err) noexcept { // intentionally pass-by-value
          STDEXEC_TRY {
            auto __err_copy = static_cast<_Error &&>(__err); // make a local copy of the error...
            __state_->__cleanup(); // because this could potentially invalidate it.
            STDEXEC::set_error(std::move(__state_->__rcvr_), static_cast<_Error &&>(__err_copy));
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!__nothrow_decay_copyable<_Error>) {
              __state_->__cleanup();
              STDEXEC::set_error(
                std::move(__state_->__rcvr_), std::current_exception());
            }
          }
        }

        constexpr void set_stopped() noexcept {
          __state_->__cleanup();
          STDEXEC::set_stopped(std::move(__state_->__rcvr_));
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
          return STDEXEC::get_env(__state_->__rcvr_);
        }

        __repeat_state_base<_Receiver> *__state_;
      };
    };

    STDEXEC_PRAGMA_PUSH()
    STDEXEC_PRAGMA_IGNORE_GNU("-Wtsan")

    template <class _Sender, class _Receiver>
    struct __repeat_state : __repeat_state_base<_Receiver> {
      using __child_t = __decay_t<__data_of<_Sender>>;
      using __receiver_t = STDEXEC::__t<__receiver<__id<_Receiver>>>;
      using __child_on_sched_sender_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler>, __child_t &>;
      using __child_op_t = STDEXEC::connect_result_t<__child_on_sched_sender_t, __receiver_t>;

      constexpr explicit __repeat_state(_Sender &&__sndr, _Receiver &&__rcvr)
        : __repeat_state_base<_Receiver>(static_cast<_Receiver &&>(__rcvr))
        , __child_(STDEXEC::__get<1>(static_cast<_Sender &&>(__sndr))) {
        __connect();
      }

      constexpr auto __connect() -> __child_op_t & {
        return __child_op_.__emplace_from(
          STDEXEC::connect,
          exec::sequence(STDEXEC::schedule(this->__sched_), __child_),
          __receiver_t{this});
      }

      constexpr void __start() noexcept {
        STDEXEC::start(*__child_op_);
      }

      constexpr void __cleanup() noexcept final {
        __child_op_.reset();
      }

      constexpr void __repeat() noexcept final {
        STDEXEC_TRY {
          STDEXEC::start(__connect());
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(static_cast<_Receiver &&>(this->__rcvr_), std::current_exception());
        }
      }

      __child_t __child_;
      STDEXEC::__optional<__child_op_t> __child_op_;
    };

    template <class _Sender, class _Receiver>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __repeat_state(_Sender &&, _Receiver) -> __repeat_state<_Sender, _Receiver>;

    STDEXEC_PRAGMA_POP()

    template <
      __mstring _Where = "In repeat_until: "_mstr,
      __mstring _What = "The input sender must send a single value that is convertible to bool"_mstr
    >
    struct _INVALID_ARGUMENT_TO_REPEAT_EFFECT_UNTIL_ { };

    template <class _Sender, class... _Args>
    using __values_t =
      // There's something funny going on with __if_c here. Use std::conditional_t instead. :-(
      std::conditional_t<
        ((sizeof...(_Args) == 1) && (__std::convertible_to<_Args, bool> && ...)),
        std::conditional_t<
          (__bool_constant<_Args, false> && ...),
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

    struct __repeat_until_tag { };

    struct __repeat_until_impl : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        // TODO: port this to use constant evaluation
        return __completions_t<__data_of<_Sender>, _Env...>{};
      };

      static constexpr auto get_state =
        []<class _Sender, class _Receiver>(_Sender &&__sndr, _Receiver &&__rcvr) {
          return __repeat_state{static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr)};
        };

      static constexpr auto start = [](auto &__state) noexcept -> void {
        __state.__start();
      };
    };

    struct repeat_until_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender &&__sndr) const {
        return __make_sexpr<repeat_until_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const {
        return __closure(*this);
      }

      template <class _Sender>
      static constexpr auto transform_sender(STDEXEC::set_value_t, _Sender &&__sndr, __ignore) {
        return STDEXEC::__apply(
          []<class _Child>(__ignore, __ignore, _Child __child) {
            return __make_sexpr<__repeat_until_tag>(std::move(__child));
          },
          static_cast<_Sender &&>(__sndr));
      }
    };

    struct repeat_t {
      struct _never {
        template <class... _Args>
        STDEXEC_ATTRIBUTE(host, device, always_inline)
        constexpr std::false_type operator()(_Args &&...) const noexcept {
          return {};
        }
      };

      template <sender _Sender>
      constexpr auto operator()(_Sender &&__sndr) const {
        return __make_sexpr<repeat_t>({}, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const {
        return __closure(*this);
      }

      template <class _Sender>
      static constexpr auto transform_sender(STDEXEC::set_value_t, _Sender &&__sndr, __ignore) {
        return STDEXEC::__apply(
          [](__ignore, __ignore, auto __child) {
            return repeat_until_t{}(STDEXEC::then(std::move(__child), _never{}));
          },
          static_cast<_Sender &&>(__sndr));
      }
    };
  } // namespace __repeat

  using __repeat::repeat_t;
  inline constexpr repeat_t repeat{};

  using __repeat::repeat_until_t;
  inline constexpr repeat_until_t repeat_until{};

  /// deprecated interfaces
  using repeat_effect_t [[deprecated("use exec::repeat_t instead")]] = repeat_t;
  using repeat_effect_until_t [[deprecated("use exec::repeat_until_t instead")]] = repeat_until_t;
  [[deprecated("use exec::repeat instead")]]
  inline constexpr const repeat_t &repeat_effect = repeat;
  [[deprecated("use exec::repeat_until instead")]]
  inline constexpr const repeat_until_t &repeat_effect_until = repeat_until;
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::__repeat::__repeat_until_tag>
    : exec::__repeat::__repeat_until_impl { }; // namespace STDEXEC

  template <>
  struct __sexpr_impl<exec::repeat_until_t> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      static_assert(sender_expr_for<_Sender, exec::repeat_until_t>);
      using __sndr_t =
        __detail::__transform_sender_result_t<exec::repeat_until_t, set_value_t, _Sender, env<>>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
    }
  };
} // namespace STDEXEC
