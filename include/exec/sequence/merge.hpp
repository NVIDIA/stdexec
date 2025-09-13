/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../../stdexec/concepts.hpp"
#include "../../stdexec/execution.hpp"
#include "../sequence_senders.hpp"

#include "../__detail/__basic_sequence.hpp"

namespace exec {
  namespace __merge {
    using namespace stdexec;

    template <class _Receiver>
    struct __operation_base {
      _Receiver __receiver_;
    };

    template <class _ReceiverId>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __receiver;
        __operation_base<_Receiver>* __op_;

        template <class _Item>
          requires __callable<exec::set_next_t, _Receiver&, _Item>
        auto set_next(_Item&& __item) & noexcept(
          __nothrow_callable<set_next_t, _Receiver&, _Item>)
          -> next_sender_of_t<_Receiver, _Item> {
          return exec::set_next(
            __op_->__receiver_, static_cast<_Item&&>(__item));
        }

        void set_value() noexcept {
          stdexec::set_value(static_cast<_Receiver&&>(__op_->__receiver_));
        }

        template <class _Error>
          requires __callable<set_error_t, _Receiver, _Error>
        void set_error(_Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__op_->__receiver_), static_cast<_Error&&>(__error));
        }

        void set_stopped() noexcept
          requires __callable<set_stopped_t, _Receiver>
        {
          stdexec::set_stopped(static_cast<_Receiver&&>(__op_->__receiver_));
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__op_->__receiver_);
        }
      };
    };

    template <class _Sequence0, class _ReceiverId, class _Sequence1>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_Receiver> {
        using __id = __operation;
        subscribe_result_t<_Sequence0, stdexec::__t<__receiver<_ReceiverId>>> __op0_;
        subscribe_result_t<_Sequence1, stdexec::__t<__receiver<_ReceiverId>>> __op1_;

        __t(_Sequence0&& __sq0, _Receiver __rcvr, _Sequence1 __sq1)
          : __operation_base<
              _Receiver
            >{static_cast<_Receiver&&>(__rcvr)}
          , __op0_{exec::subscribe(
              static_cast<_Sequence0&&>(__sq0),
              stdexec::__t<__receiver<_ReceiverId>>{this})}
          , __op1_{exec::subscribe(
              static_cast<_Sequence1&&>(__sq1),
              stdexec::__t<__receiver<_ReceiverId>>{this})} {
        }

        void start() & noexcept {
          stdexec::start(__op0_);
          stdexec::start(__op1_);
        }
      };
    };

    template <class _Receiver>
    struct __subscribe_fn {
      _Receiver& __rcvr_;

      template <class _Sequence1, class _Sequence0>
      auto operator()(__ignore, _Sequence1 __sq1, _Sequence0&& __sq0) noexcept(
        __nothrow_decay_copyable<_Sequence1> && __nothrow_decay_copyable<_Sequence0>
        && __nothrow_move_constructible<_Receiver>)
        -> __t<__operation<_Sequence0, __id<_Receiver>, _Sequence1>> {
        return {
          static_cast<_Sequence0&&>(__sq0),
          static_cast<_Receiver&&>(__rcvr_),
          static_cast<_Sequence1&&>(__sq1)};
      }
    };

    struct merge_t {
      template <sender _Sequence0, sender _Sequence1>
      auto operator()(_Sequence0&& __sq0, _Sequence1&& __sq1) const
        noexcept(__nothrow_decay_copyable<_Sequence0> && __nothrow_decay_copyable<_Sequence1>) {
        return make_sequence_expr<merge_t>(
          static_cast<_Sequence1&&>(__sq1), static_cast<_Sequence0&&>(__sq0));
      }

      template <class _Self, class _Env>
      using __completion_sigs_t = __sequence_completion_signatures_of_t<__child_of<_Self>, _Env>;

      template <sender_expr_for<merge_t> _Self, class _Env>
      static auto
        get_completion_signatures(_Self&&, _Env&&) noexcept -> __completion_sigs_t<_Self, _Env> {
        return {};
      }

      template <class _Self, class... _Env>
      using __item_types_t = stdexec::__mapply<
        stdexec::__munique<stdexec::__q<exec::item_types>>,
          stdexec::__minvoke<
            stdexec::__mconcat<stdexec::__qq<exec::item_types>>,
              item_types_of_t<__child_of<_Self>, _Env...>,
              item_types_of_t<__data_of<_Self>, _Env...>>>;

      template <sender_expr_for<merge_t> _Self, class _Env>
      static auto get_item_types(_Self&&, _Env&&) noexcept -> __item_types_t<_Self, _Env> {
        return {};
      }

      template <class _Self, class _Receiver>
      using __receiver_t = __t<__receiver<__id<_Receiver>>>;

      template <class _Self, class _Receiver>
      using __operation_t = __t<__operation<__child_of<_Self>, __id<_Receiver>, __data_of<_Self>>>;

      template <sender_expr_for<merge_t> _Self, receiver _Receiver>
        requires sequence_receiver_of<_Receiver, __item_types_t<_Self, env_of_t<_Receiver>>>
              && sequence_sender_to<__child_of<_Self>, __receiver_t<_Self, _Receiver>>
              && sequence_sender_to<__data_of<_Self>, __receiver_t<_Self, _Receiver>>
      static auto subscribe(_Self&& __self, _Receiver __rcvr)
        noexcept(__nothrow_callable<__sexpr_apply_t, _Self, __subscribe_fn<_Receiver>>)
          -> __call_result_t<__sexpr_apply_t, _Self, __subscribe_fn<_Receiver>> {
        return __sexpr_apply(static_cast<_Self&&>(__self), __subscribe_fn<_Receiver>{__rcvr});
      }

      template <sender_expr_for<merge_t> _Sexpr>
      static auto get_env(const _Sexpr& __sexpr) noexcept -> env_of_t<__child_of<_Sexpr>> {
        return __sexpr_apply(__sexpr, []<class _Child>(__ignore, __ignore, const _Child& __child) {
          return stdexec::get_env(__child);
        });
      }
    };
  } // namespace __merge

  using __merge::merge_t;
  inline constexpr merge_t merge{};
} // namespace exec
