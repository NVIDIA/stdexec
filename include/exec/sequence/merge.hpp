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

#include "../../stdexec/execution.hpp"

#include "../__detail/__basic_sequence.hpp"
#include "../sequence_senders.hpp"
#include "ignore_all_values.hpp"
#include "transform_each.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(not_used_in_template_function_params)

namespace exec {
  namespace __merge {
    using namespace STDEXEC;

    template <class _Receiver>
    struct __operation_base {
      _Receiver __rcvr_;
    };

    template <class _Receiver>
    struct __result_receiver {
      using receiver_concept = STDEXEC::receiver_t;

      void set_value() noexcept {
        STDEXEC::set_value(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        STDEXEC::set_error(
          static_cast<_Receiver&&>(__op_->__rcvr_), static_cast<_Error&&>(__error));
      }

      void set_stopped() noexcept {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__op_->__rcvr_);
      }

      __operation_base<_Receiver>* __op_;
    };

    template <class _Receiver>
    struct __merge_each_fn {
      template <STDEXEC::sender _Item>
      auto operator()(_Item&& __item, __operation_base<_Receiver>* __op) const
        noexcept(__nothrow_callable<set_next_t, _Receiver&, _Item>)
          -> next_sender_of_t<_Receiver, _Item> {
        return exec::set_next(__op->__rcvr_, static_cast<_Item&&>(__item));
      }
    };

    struct __combine {
      template <class _Receiver>
      using merge_each_fn_t = __closure<__merge_each_fn<_Receiver>, __operation_base<_Receiver>*>;

      template <class _Sequence, class _Receiver>
      using transform_sender_t =
        __call_result_t<exec::transform_each_t, _Sequence, merge_each_fn_t<_Receiver>>;
      template <class _Sequence, class _Receiver>
      using ignored_sender_t =
        __call_result_t<exec::ignore_all_values_t, transform_sender_t<_Sequence, _Receiver>>;

      template <class _Receiver, class... _Sequences>
      using result_sender_t =
        __call_result_t<when_all_t, ignored_sender_t<_Sequences, _Receiver>...>;
    };

    template <class _Receiver, class... _Sequences>
    struct __operation : __operation_base<_Receiver> {
      using merge_each_fn_t = __combine::merge_each_fn_t<_Receiver>;

      template <class _ReceiverDependent>
      using result_sender_t = __combine::result_sender_t<_ReceiverDependent, _Sequences...>;

      explicit __operation(_Receiver __rcvr, _Sequences... __sequences)
        : __operation_base<_Receiver>{static_cast<_Receiver&&>(__rcvr)}
        , __op_result_{STDEXEC::connect(
            STDEXEC::when_all(
              exec::ignore_all_values(
                exec::transform_each(
                  static_cast<_Sequences&&>(__sequences),
                  merge_each_fn_t({}, this)))...),
            __result_receiver<_Receiver>{this})} {
      }

      void start() & noexcept {
        STDEXEC::start(__op_result_);
      }

      connect_result_t<result_sender_t<_Receiver>, __result_receiver<_Receiver>> __op_result_;
    };

    struct merge_t {
     private:
      struct __subscribe_fn {
        template <class _Receiver, class... _Sequences>
        auto operator()(_Receiver& __rcvr, __ignore, __ignore, _Sequences... __sequences)
          noexcept(__nothrow_decay_copyable<_Sequences...>)
            -> __operation<_Receiver, _Sequences...> {
          return __operation<_Receiver, _Sequences...>{
            static_cast<_Receiver&&>(__rcvr), static_cast<_Sequences&&>(__sequences)...};
        }
      };

      template <class... _Env>
      static consteval auto __mk_item_transform() {
        return []<class _ItemSender>() {
          return exec::__sequence_completion_signatures_of<_ItemSender, _Env...>();
        };
      }

      static consteval auto __mk_unique_concat_items() {
        return []<class... _ItemLists>(_ItemLists...) {
          return STDEXEC::__minvoke<
            STDEXEC::__mconcat<STDEXEC::__munique<STDEXEC::__qq<exec::item_types>>>,
            _ItemLists...
          >();
        };
      }

      template <class... _Env>
      static consteval auto __mk_get_item_types() {
        return []<class _ItemSender>() {
          return exec::get_item_types<_ItemSender, _Env...>();
        };
      }

     public:
      template <class... _Sequences>
      auto operator()(_Sequences&&... __sequences) const
        noexcept(__nothrow_decay_copyable<_Sequences...>) -> __well_formed_sequence_sender auto {
        return make_sequence_expr<merge_t>({}, static_cast<_Sequences&&>(__sequences)...);
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() noexcept {
        static_assert(STDEXEC::sender_expr_for<_Self, merge_t>);
        auto __items = STDEXEC::__children_of<_Self, STDEXEC::__qq<item_types>>();
        return exec::concat_completion_signatures(
          completion_signatures<set_stopped_t()>(),
          __items.__transform(__mk_item_transform<_Env...>(), exec::concat_completion_signatures));
      }

      template <class _Self, class... _Env>
      static consteval auto get_item_types() {
        static_assert(sender_expr_for<_Self, merge_t>);
        auto __items = STDEXEC::__children_of<_Self, STDEXEC::__qq<item_types>>();
        return __items.__transform(__mk_get_item_types<_Env...>(), __mk_unique_concat_items());
      }

      template <class _Self, receiver _Receiver>
      static auto subscribe(_Self&& __self, _Receiver __rcvr)
        noexcept(__nothrow_applicable<__subscribe_fn, _Self, _Receiver&>)
          -> __apply_result_t<__subscribe_fn, _Self, _Receiver&> {
        static_assert(sender_expr_for<_Self, merge_t>);
        return STDEXEC::__apply(__subscribe_fn{}, static_cast<_Self&&>(__self), __rcvr);
      }
    };
  } // namespace __merge

  using __merge::merge_t;
  inline constexpr merge_t merge{};
} // namespace exec

STDEXEC_PRAGMA_POP()
