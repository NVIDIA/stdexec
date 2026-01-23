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

#include <cstddef>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(expr_has_no_effect)
STDEXEC_PRAGMA_IGNORE_GNU("-Wunused-value")

namespace exec {
  namespace __repeat_n {
    using namespace STDEXEC;

    template <class _Receiver>
    struct __repeat_n_state_base {
      constexpr explicit __repeat_n_state_base(_Receiver &&__rcvr, std::size_t __count) noexcept
        : __rcvr_{static_cast<_Receiver &&>(__rcvr)}
        , __count_{__count} {
      }

      virtual void __cleanup() noexcept = 0;
      virtual void __repeat() noexcept = 0;

      _Receiver __rcvr_;
      std::size_t __count_;
      trampoline_scheduler __sched_;
    };

    template <class _SenderId, class _ReceiverId>
    struct __receiver {
      using _Sender = STDEXEC::__t<_SenderId>;
      using _Receiver = STDEXEC::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        using receiver_concept = STDEXEC::receiver_t;

        constexpr void set_value() noexcept {
          __state_->__repeat();
        }

        template <class _Error>
        constexpr void set_error(_Error &&__err) noexcept {
          STDEXEC_TRY {
            auto __err_copy = static_cast<_Error &&>(__err); // make a copy of the error...
            __state_->__cleanup(); // ... because this could potentially invalidate it.
            STDEXEC::set_error(std::move(__state_->__rcvr_), std::move(__err_copy));
          }
          STDEXEC_CATCH_ALL {
            if constexpr (!__nothrow_decay_copyable<_Error>) {
              STDEXEC::set_error(std::move(__state_->__rcvr_), std::current_exception());
            }
          }
        }

        constexpr void set_stopped() noexcept {
          __state_->__cleanup();
          STDEXEC::set_stopped(std::move(__state_->__rcvr_));
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return STDEXEC::get_env(__state_->__rcvr_);
        }

        __repeat_n_state_base<_Receiver> *__state_;
      };
    };

    template <class _Child>
    struct __child_count_pair {
      _Child __child_;
      std::size_t __count_;
    };

    template <class _Child>
    __child_count_pair(_Child, std::size_t) -> __child_count_pair<_Child>;

    template <class _Sender, class _Receiver>
    struct __repeat_n_state : __repeat_n_state_base<_Receiver> {
      using __child_count_pair_t = __decay_t<__data_of<_Sender>>;
      using __child_t = decltype(__child_count_pair_t::__child_);
      using __receiver_t = STDEXEC::__t<__receiver<__id<_Sender>, __id<_Receiver>>>;
      using __child_on_sched_sender_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler &>, __child_t &>;
      using __child_op_t = STDEXEC::connect_result_t<__child_on_sched_sender_t, __receiver_t>;

      constexpr explicit __repeat_n_state(_Sender &&__sndr, _Receiver &&__rcvr)
        : __repeat_n_state_base<_Receiver>{
            static_cast<_Receiver &&>(__rcvr),
            STDEXEC::__get<1>(static_cast<_Sender &&>(__sndr)).__count_}
        , __child_(STDEXEC::__get<1>(static_cast<_Sender &&>(__sndr)).__child_) {
        if (this->__count_ != 0) {
          __connect();
        }
      }

      constexpr auto __connect() -> __child_op_t & {
        return __child_op_.__emplace_from(
          STDEXEC::connect,
          exec::sequence(STDEXEC::schedule(this->__sched_), __child_),
          __receiver_t{this});
      }

      constexpr void __start() noexcept {
        if (this->__count_ == 0) {
          STDEXEC::set_value(static_cast<_Receiver &&>(this->__rcvr_));
        } else {
          STDEXEC::start(*__child_op_);
        }
      }

      constexpr void __cleanup() noexcept final {
        __child_op_.reset();
      }

      constexpr void __repeat() noexcept final {
        STDEXEC_ASSERT(this->__count_ > 0);
        STDEXEC_TRY {
          if (--this->__count_ == 0) {
            __cleanup();
            STDEXEC::set_value(std::move(this->__rcvr_));
          } else {
            STDEXEC::start(__connect());
          }
        }
        STDEXEC_CATCH_ALL {
          STDEXEC::set_error(std::move(this->__rcvr_), std::current_exception());
        }
      }

      __child_t __child_;
      STDEXEC::__optional<__child_op_t> __child_op_;
    };

    template <class _Sender, class _Receiver>
    STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
      __repeat_n_state(_Sender &&, _Receiver) -> __repeat_n_state<_Sender, _Receiver>;

    struct repeat_n_t;
    struct _REPEAT_N_EXPECTS_A_SENDER_OF_VOID_;

    template <class _Sender, class... _Args>
    using __values_t =
      // There's something funny going on with __if_c here. Use std::conditional_t instead. :-(
      std::conditional_t<
        (sizeof...(_Args) == 0),
        completion_signatures<>,
        __mexception<
          _REPEAT_N_EXPECTS_A_SENDER_OF_VOID_,
          _WHERE_(_IN_ALGORITHM_, repeat_n_t),
          _WITH_PRETTY_SENDER_<_Sender>
        >
      >;

    template <class _Error>
    using __error_t = completion_signatures<set_error_t(__decay_t<_Error>)>;

    template <class _Pair, class... _Env>
    using __completions_t = STDEXEC::transform_completion_signatures<
      __completion_signatures_of_t<decltype(__decay_t<_Pair>::__child_) &, _Env...>,
      STDEXEC::transform_completion_signatures<
        __completion_signatures_of_t<STDEXEC::schedule_result_t<exec::trampoline_scheduler>, _Env...>,
        __eptr_completion,
        __cmplsigs::__default_set_value,
        __error_t
      >,
      __mbind_front_q<__values_t, decltype(__decay_t<_Pair>::__child_)>::template __f,
      __error_t
    >;

    struct __repeat_n_tag { };

    struct __repeat_n_impl : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        // TODO: port this to use constant evaluation
        return __completions_t<__data_of<_Sender>, _Env...>{};
      }

      static constexpr auto get_state = []<class _Sender, class _Receiver>(
                                          _Sender &&__sndr,
                                          _Receiver &&__rcvr) {
        return __repeat_n_state{static_cast<_Sender &&>(__sndr), static_cast<_Receiver &&>(__rcvr)};
      };

      static constexpr auto start = [](auto &__state) noexcept -> void {
        __state.__start();
      };
    };

    struct repeat_n_t {
      template <sender _Sender>
      constexpr auto operator()(_Sender &&__sndr, std::size_t __count) const {
        return __make_sexpr<repeat_n_t>(__count, static_cast<_Sender &&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()(std::size_t __count) const noexcept {
        return __closure(*this, __count);
      }

      template <class _Sender>
      static auto transform_sender(set_value_t, _Sender &&__sndr, __ignore)
        noexcept(__nothrow_decay_copyable<_Sender>) {
        auto &[__tag, __count, __child] = __sndr;
        return __make_sexpr<__repeat_n_tag>(
          __child_count_pair{STDEXEC::__forward_like<_Sender>(__child), __count});
      }
    };
  } // namespace __repeat_n

  using __repeat_n::repeat_n_t;
  inline constexpr repeat_n_t repeat_n{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::__repeat_n::__repeat_n_tag> : exec::__repeat_n::__repeat_n_impl { };

  template <>
  struct __sexpr_impl<exec::repeat_n_t> : __sexpr_defaults {
    template <class _Sender, class... _Env>
    static consteval auto get_completion_signatures() {
      using __sndr_t =
        __detail::__transform_sender_result_t<exec::repeat_n_t, set_value_t, _Sender, env<>>;
      return STDEXEC::get_completion_signatures<__sndr_t, _Env...>();
    };
  };
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
