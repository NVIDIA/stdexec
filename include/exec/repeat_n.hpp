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
    struct __opstate_base : __immovable {
      constexpr explicit __opstate_base(_Receiver &&__rcvr, std::size_t __count) noexcept
        : __rcvr_{static_cast<_Receiver &&>(__rcvr)}
        , __count_{__count} {
      }

      virtual constexpr void __cleanup() noexcept = 0;
      virtual constexpr void __repeat() noexcept = 0;

      _Receiver __rcvr_;
      std::size_t __count_;
      trampoline_scheduler __sched_;
    };

    template <class _Receiver>
    struct __receiver {
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

      [[nodiscard]]
      constexpr auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__state_->__rcvr_);
      }

      __opstate_base<_Receiver> *__state_;
    };

    template <class _Child, class _Receiver>
    struct __opstate final : __opstate_base<_Receiver> {
      using __receiver_t = __receiver<_Receiver>;
      using __bouncy_sndr_t =
        __result_of<exec::sequence, schedule_result_t<trampoline_scheduler>, _Child &>;
      using __child_op_t = STDEXEC::connect_result_t<__bouncy_sndr_t, __receiver_t>;
      static constexpr bool __nothrow_connect =
        STDEXEC::__nothrow_connectable<__bouncy_sndr_t, __receiver_t>;

      constexpr explicit __opstate(std::size_t __count, _Child __child, _Receiver __rcvr)
        noexcept(__nothrow_connect)
        : __opstate_base<_Receiver>{static_cast<_Receiver &&>(__rcvr), __count}
        , __child_(std::move(__child)) {
        if (this->__count_ != 0) {
          __connect();
        }
      }

      constexpr void start() noexcept {
        if (this->__count_ == 0) {
          STDEXEC::set_value(static_cast<_Receiver &&>(this->__rcvr_));
        } else {
          STDEXEC::start(*__child_op_);
        }
      }

      constexpr auto __connect() noexcept(__nothrow_connect) -> __child_op_t & {
        return __child_op_.__emplace_from(
          STDEXEC::connect,
          exec::sequence(STDEXEC::schedule(this->__sched_), __child_),
          __receiver_t{this});
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

      _Child __child_;
      STDEXEC::__optional<__child_op_t> __child_op_;
    };

    struct repeat_n_t;
    struct _THE_INPUT_SENDER_MUST_HAVE_VOID_VALUE_COMPLETION_;

    template <class _Child, class... _Args>
    using __values_t =
      // There's something funny going on with __if_c here. Use std::conditional_t instead. :-(
      std::conditional_t<
        (sizeof...(_Args) == 0),
        completion_signatures<>,
        __mexception<
          _WHAT_(_INVALID_ARGUMENT_),
          _WHERE_(_IN_ALGORITHM_, repeat_n_t),
          _WHY_(_THE_INPUT_SENDER_MUST_HAVE_VOID_VALUE_COMPLETION_),
          _WITH_PRETTY_SENDER_<_Child>
        >
      >;

    template <class _Error>
    using __error_t = completion_signatures<set_error_t(__decay_t<_Error>)>;

    template <class _Child, class... _Env>
    using __completions_t = STDEXEC::transform_completion_signatures<
      __completion_signatures_of_t<_Child &, _Env...>,
      STDEXEC::transform_completion_signatures<
        __completion_signatures_of_t<STDEXEC::schedule_result_t<exec::trampoline_scheduler>, _Env...>,
        __eptr_completion,
        __cmplsigs::__default_set_value,
        __error_t
      >,
      __mbind_front_q<__values_t, _Child>::template __f,
      __error_t
    >;

    struct __impls : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        // TODO: port this to use constant evaluation
        return __completions_t<__child_of<_Sender>, _Env...>{};
      }

      static constexpr auto connect = //
        []<class _Receiver, class _Sender>(_Sender &&__sndr, _Receiver &&__rcvr) noexcept(
          noexcept(__opstate(0, STDEXEC::__get<2>(__declval<_Sender>()), __declval<_Receiver>()))) {
          const std::size_t __count = STDEXEC::__get<1>(__sndr);
          return __opstate(
            __count,
            STDEXEC::__get<2>(static_cast<_Sender &&>(__sndr)),
            static_cast<_Receiver &&>(__rcvr));
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
    };
  } // namespace __repeat_n

  using __repeat_n::repeat_n_t;
  inline constexpr repeat_n_t repeat_n{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::repeat_n_t> : exec::__repeat_n::__impls { };
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
