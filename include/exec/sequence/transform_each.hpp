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
#include "stdexec/__detail/__diagnostics.hpp"
#include "stdexec/__detail/__meta.hpp"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(not_used_in_template_function_params)

namespace exec {
  template <class _Transform>
  struct _TRANSFORM_EACH_ADAPTOR_INVOCATION_FAILED_;

  template <class _Transform>
  struct _TRANSFORM_EACH_ITEM_TYPES_OF_THE_CHILD_ARE_INVALID_;

  template <class _Transform>
  struct _TRANSFORM_EACH_ITEM_TYPES_CALCULATION_FAILED_;

  template <class _Closure>
  struct _WITH_ADAPTOR_;

  struct _WITH_ITEM_SENDER_ { };

  template <class _Sender>
  using _WITH_PRETTY_ITEM_SENDER_ = _WITH_ITEM_SENDER_(STDEXEC::__demangle_t<_Sender>);

  namespace __transform_each {
    using namespace STDEXEC;

    template <class _Receiver, class _Adaptor>
    struct __operation_base {
      _Receiver __rcvr_;
      _Adaptor __adaptor_;
    };

    template <class _Receiver, class _Adaptor>
    struct __receiver {
      using receiver_concept = STDEXEC::receiver_t;

      template <class _Item>
      auto set_next(_Item&& __item) & noexcept(
        __nothrow_callable<set_next_t, _Receiver&, __call_result_t<_Adaptor&, _Item>>
        && __nothrow_callable<_Adaptor&, _Item>)
        -> next_sender_of_t<_Receiver, __call_result_t<_Adaptor&, _Item>> {
        return exec::set_next(
          __opstate_->__rcvr_, __opstate_->__adaptor_(static_cast<_Item&&>(__item)));
      }

      void set_value() noexcept {
        STDEXEC::set_value(static_cast<_Receiver&&>(__opstate_->__rcvr_));
      }

      template <class _Error>
      void set_error(_Error&& __error) noexcept {
        STDEXEC::set_error(
          static_cast<_Receiver&&>(__opstate_->__rcvr_), static_cast<_Error&&>(__error));
      }

      void set_stopped() noexcept
        requires __callable<set_stopped_t, _Receiver>
      {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__opstate_->__rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__opstate_->__rcvr_);
      }

      __operation_base<_Receiver, _Adaptor>* __opstate_;
    };

    template <class _Sender, class _Receiver, class _Adaptor>
    struct __operation : __operation_base<_Receiver, _Adaptor> {
      __operation(_Sender&& __sndr, _Receiver __rcvr, _Adaptor __adaptor)
        : __operation_base<_Receiver, _Adaptor>{
            static_cast<_Receiver&&>(__rcvr),
            static_cast<_Adaptor&&>(__adaptor)}
        , __opstate_{exec::subscribe(
            static_cast<_Sender&&>(__sndr),
            __receiver<_Receiver, _Adaptor>{this})} {
      }

      void start() & noexcept {
        STDEXEC::start(__opstate_);
      }

      subscribe_result_t<_Sender, __receiver<_Receiver, _Adaptor>> __opstate_;
    };

    template <class _Receiver>
    struct __subscribe_fn {
      _Receiver& __rcvr_;

      template <class _Adaptor, class _Sequence>
      auto operator()(__ignore, _Adaptor __adaptor, _Sequence&& __sequence)
        noexcept(__nothrow_decay_copyable<_Adaptor> && __nothrow_decay_copyable<_Sequence>)
          -> __operation<_Sequence, _Receiver, _Adaptor> {
        return {
          static_cast<_Sequence&&>(__sequence),
          static_cast<_Receiver&&>(__rcvr_),
          static_cast<_Adaptor&&>(__adaptor)};
      }
    };

    struct transform_each_t {
      template <sender _Sequence, __sender_adaptor_closure _Adaptor>
      auto operator()(_Sequence&& __sndr, _Adaptor&& __adaptor) const
        noexcept(__nothrow_decay_copyable<_Sequence> && __nothrow_decay_copyable<_Adaptor>)
          -> __well_formed_sequence_sender auto {
        return make_sequence_expr<transform_each_t>(
          static_cast<_Adaptor&&>(__adaptor), static_cast<_Sequence&&>(__sndr));
      }

      template <class _Adaptor>
      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto
        operator()(_Adaptor&& __adaptor) const noexcept(__nothrow_decay_copyable<_Adaptor>) {
        return __closure(*this, static_cast<_Adaptor&&>(__adaptor));
      }

      template <class _Self, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Self, transform_each_t>);
        return exec::__sequence_completion_signatures_of<__child_of<_Self>, _Env...>();
      }

      static consteval auto __collect_item_types() {
        return []<class... _Senders>(STDEXEC::__mtype<_Senders>...) {
          return STDEXEC::__minvoke<STDEXEC::__munique<STDEXEC::__qq<item_types>>, _Senders...>();
        };
      }

      template <class _Self, class... _Env>
      static consteval auto get_item_types() {
        static_assert(sender_expr_for<_Self, transform_each_t>);
        using __closure_t = STDEXEC::__decay_t<__data_of<_Self>>&;
        auto __child_items = exec::get_item_types<__child_of<_Self>, _Env...>();

        if constexpr (STDEXEC::__merror<decltype(__child_items)>) {
          return exec::__invalid_item_types(__child_items);
        } else {
          return __child_items.__transform(
            []<class _ItemSender>() {
              if constexpr (!__callable<__closure_t, _ItemSender>) {
                return exec::__invalid_item_types<
                  _TRANSFORM_EACH_ADAPTOR_INVOCATION_FAILED_<_Self>,
                  _WITH_PRETTY_SEQUENCE_<__child_of<_Self>>,
                  __fn_t<_WITH_ENVIRONMENT_, _Env>...,
                  _WITH_ADAPTOR_<__data_of<_Self>>,
                  _WITH_PRETTY_ITEM_SENDER_<_ItemSender>
                >();
              } else {
                return STDEXEC::__mtype<__call_result_t<__closure_t, _ItemSender>>();
              }
            },
            __collect_item_types());
        }
      }

      template <class _Self, class _Receiver>
      using __receiver_t = __receiver<_Receiver, __data_of<_Self>>;

      template <class _Self, class _Receiver>
      using __operation_t = __operation<__child_of<_Self>, _Receiver, __data_of<_Self>>;

      template <sender_expr_for<transform_each_t> _Self, receiver _Receiver>
      static auto subscribe(_Self&& __self, _Receiver __rcvr)
        noexcept(__nothrow_applicable<__subscribe_fn<_Receiver>, _Self>)
          -> __apply_result_t<__subscribe_fn<_Receiver>, _Self> {
        return __apply(__subscribe_fn<_Receiver>{__rcvr}, static_cast<_Self&&>(__self));
      }

      template <sender_expr_for<transform_each_t> _Sexpr>
      static auto get_env(const _Sexpr& __sexpr) noexcept -> env_of_t<__child_of<_Sexpr>> {
        return __apply(
          []<class _Child>(__ignore, __ignore, const _Child& __child) {
            return STDEXEC::get_env(__child);
          },
          __sexpr);
      }
    };
  } // namespace __transform_each

  using __transform_each::transform_each_t;
  inline constexpr transform_each_t transform_each{};
} // namespace exec

STDEXEC_PRAGMA_POP()
