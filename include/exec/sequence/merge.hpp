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
#include "./transform_each.hpp"
#include "./ignore_all_values.hpp"
#include "stdexec/__detail/__execution_fwd.hpp"
#include "stdexec/__detail/__meta.hpp"
#include "stdexec/__detail/__senders_core.hpp"
#include "stdexec/__detail/__transform_completion_signatures.hpp"

namespace exec {
  namespace __merge {
    using namespace stdexec;

    template <class _Receiver>
    struct __operation_base {
      _Receiver __receiver_;
    };

    template <class _ReceiverId>
    struct __result_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using receiver_concept = stdexec::receiver_t;
        using __id = __result_receiver;

        __operation_base<_Receiver>* __op_;

        void set_value() noexcept {
          stdexec::set_value(static_cast<_Receiver&&>(__op_->__receiver_));
        }

        template <class _Error>
        void set_error(_Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__op_->__receiver_), static_cast<_Error&&>(__error));
        }

        void set_stopped() noexcept {
          stdexec::set_stopped(static_cast<_Receiver&&>(__op_->__receiver_));
        }

        auto get_env() const noexcept -> env_of_t<_Receiver> {
          return stdexec::get_env(__op_->__receiver_);
        }
      };
    };

    template <class _ReceiverId>
    struct __merge_each_fn {
      using _Receiver = stdexec::__t<_ReceiverId>;

      template <stdexec::sender _Item>
      auto operator()(_Item&& __item, __operation_base<_Receiver>* __op) const
        noexcept(__nothrow_callable<set_next_t, _Receiver&, _Item>)
          -> next_sender_of_t<_Receiver, _Item> {
        return exec::set_next(__op->__receiver_, static_cast<_Item&&>(__item));
      }
    };

    struct __combine {
      template <class _ReceiverId>
      using merge_each_fn_t =
        __closure<__merge_each_fn<_ReceiverId>, __operation_base<__t<_ReceiverId>>*>;

      template <class _Sequence, class _ReceiverId>
      using transform_sender_t =
        __call_result_t<exec::transform_each_t, _Sequence, merge_each_fn_t<_ReceiverId>>;
      template <class _Sequence, class _ReceiverId>
      using ignored_sender_t =
        __call_result_t<exec::ignore_all_values_t, transform_sender_t<_Sequence, _ReceiverId>>;

      template <class _ReceiverId, class... _Sequences>
      using result_sender_t =
        __call_result_t<when_all_t, ignored_sender_t<_Sequences, _ReceiverId>...>;
    };

    template <class _ReceiverId, class... _Sequences>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      using merge_each_fn_t = __combine::merge_each_fn_t<_ReceiverId>;

      template <class _ReceiverIdDependent>
      using result_sender_t = __combine::result_sender_t<_ReceiverIdDependent, _Sequences...>;

      struct __t : __operation_base<_Receiver> {
        using __id = __operation;

        connect_result_t<result_sender_t<_ReceiverId>, stdexec::__t<__result_receiver<_ReceiverId>>>
          __op_result_;

        __t(_Receiver __rcvr, _Sequences... __sequences)
          : __operation_base<_Receiver>{static_cast<_Receiver&&>(__rcvr)}
          , __op_result_{stdexec::connect(
              stdexec::when_all(
                exec::ignore_all_values(
                  exec::transform_each(
                    static_cast<_Sequences&&>(__sequences),
                    merge_each_fn_t({}, this)))...),
              stdexec::__t<__result_receiver<_ReceiverId>>{this})} {
        }

        void start() & noexcept {
          stdexec::start(__op_result_);
        }
      };
    };

    template <class _Receiver>
    struct __subscribe_fn {
      _Receiver& __rcvr_;

      template <class... _Sequences>
      auto operator()(__ignore, __ignore, _Sequences... __sequences)
        noexcept(__nothrow_decay_copyable<_Sequences...> && __nothrow_move_constructible<_Receiver>)
          -> __t<__operation<__id<_Receiver>, _Sequences...>> {
        return {static_cast<_Receiver&&>(__rcvr_), static_cast<_Sequences&&>(__sequences)...};
      }
    };

    struct merge_t {
      template <class... _Sequences>
      auto operator()(_Sequences&&... __sequences) const
        noexcept(__nothrow_decay_copyable<_Sequences...>) -> __well_formed_sequence_sender auto {
        return make_sequence_expr<merge_t>(__(), static_cast<_Sequences&&>(__sequences)...);
      }

      template <class _Error>
      using __set_error_t = completion_signatures<set_error_t(__decay_t<_Error>)>;

      struct _INVALID_ARGUMENTS_TO_MERGE_ { };

      template <class _Self, class _Env>
      using __error_t = __mexception<
        _INVALID_ARGUMENTS_TO_MERGE_,
        __children_of<_Self, __q<_WITH_SEQUENCES_>>,
        _WITH_ENVIRONMENT_<_Env>
      >;

      template <class... _Env>
      struct __completions_fn_t {
        template <class... _Sequences>
        using __f = __meval<
          __concat_completion_signatures,
          completion_signatures<set_stopped_t()>,
          __sequence_completion_signatures_of_t<_Sequences, _Env...>...
        >;
      };

      template <class _Self, class... _Env>
      using __completions_t = __children_of<_Self, __completions_fn_t<_Env...>>;

      template <sender_expr_for<merge_t> _Self, class... _Env>
      static auto get_completion_signatures(_Self&&, _Env&&...) noexcept {
        return __minvoke<__mtry_catch<__q<__completions_t>, __q<__error_t>>, _Self, _Env...>();
      }

      template <class... _Env>
      struct __items_fn_t {

        template <class... _Sequences>
        using __f = stdexec::__mapply<
          stdexec::__munique<stdexec::__q<exec::item_types>>,
          stdexec::__minvoke<
            stdexec::__mconcat<stdexec::__qq<exec::item_types>>,
            __item_types_of_t<_Sequences, _Env...>...
          >
        >;
      };

      template <class _Self, class... _Env>
      using __items_t = __children_of<_Self, __items_fn_t<_Env...>>;

      template <sender_expr_for<merge_t> _Self, class... _Env>
      static auto get_item_types(_Self&&, _Env&&...) noexcept {
        return __minvoke<__mtry_catch<__q<__items_t>, __q<__error_t>>, _Self, _Env...>();
      }

      template <sender_expr_for<merge_t> _Self, receiver _Receiver>
      static auto subscribe(_Self&& __self, _Receiver __rcvr)
        noexcept(__nothrow_callable<__sexpr_apply_t, _Self, __subscribe_fn<_Receiver>>)
          -> __sexpr_apply_result_t<_Self, __subscribe_fn<_Receiver>> {
        return __sexpr_apply(static_cast<_Self&&>(__self), __subscribe_fn<_Receiver>{__rcvr});
      }
    };
  } // namespace __merge

  using __merge::merge_t;
  inline constexpr merge_t merge{};
} // namespace exec
