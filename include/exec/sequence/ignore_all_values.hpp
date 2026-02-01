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

// include these after execution.hpp
#include "../../stdexec/__detail/__tuple.hpp"
#include "../../stdexec/__detail/__variant.hpp"
#include "../sequence_senders.hpp"

#include "../../stdexec/__detail/__atomic.hpp"

namespace exec {
  template <class _Variant, class _Type, class... _Args>
  concept __variant_emplaceable = requires(_Variant& __var, _Args&&... __args) {
    __var.template emplace<_Type>(static_cast<_Args&&>(__args)...);
  };

  template <class _Variant, class _Type, class... _Args>
  concept __nothrow_variant_emplaceable = requires(_Variant& __var, _Args&&... __args) {
    { __var.template emplace<_Type>(static_cast<_Args&&>(__args)...) } noexcept;
  };

  namespace __ignore_all_values {
    using namespace STDEXEC;

    constexpr auto __complete_fn = []<class _Receiver, class _Tag, class... _Args>(
                                     _Receiver&& __rcvr,
                                     _Tag,
                                     _Args&&... __args) noexcept {
      _Tag()(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
    };

    constexpr auto __visit_fn =
      []<class _Receiver, class _Tuple>(_Receiver&& __rcvr, _Tuple&& __tupl) noexcept {
        STDEXEC::__apply(
          __complete_fn, static_cast<_Tuple&&>(__tupl), static_cast<_Receiver&&>(__rcvr));
      };

    template <class _ResultVariant>
    struct __result_type : __immovable {
      template <class... _Args>
      void __emplace(_Args&&... __args) noexcept {
        int __expected = 0;
        if (__emplaced_.compare_exchange_strong(__expected, 1, __std::memory_order_relaxed)) {
          __result_.template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
          __emplaced_.store(2, __std::memory_order_release);
        }
      }

      template <class _Receiver>
#if STDEXEC_NVHPC() && STDEXEC_NVHPC_VERSION <= 25'09
      // Avoid a codegen issue in NVHPC 25.9 and earlier
      [[gnu::noinline]]
#endif
      void __visit_result(_Receiver __rcvr) noexcept {
        if (__emplaced_.load(__std::memory_order_acquire) == 0) {
          STDEXEC::set_value(static_cast<_Receiver&&>(__rcvr));
        } else if constexpr (STDEXEC::__mapply<STDEXEC::__msize, _ResultVariant>::value != 0) {
          STDEXEC_ASSERT(__result_.index() != __variant_npos);
          STDEXEC::__visit(
            __visit_fn, static_cast<_ResultVariant&&>(__result_), static_cast<_Receiver&&>(__rcvr));
        }
      }

      _ResultVariant __result_{};
      __std::atomic<int> __emplaced_{0};
    };

    template <class _ItemReceiver, class _ResultVariant>
    struct __item_operation_base {
      STDEXEC_ATTRIBUTE(no_unique_address) _ItemReceiver __rcvr_;
      __result_type<_ResultVariant>* __result_;
    };

    template <class _ItemReceiver, class _ResultVariant>
    struct __item_receiver {
      using receiver_concept = STDEXEC::receiver_t;
      __item_operation_base<_ItemReceiver, _ResultVariant>* __op_;

      template <class... _Args>
      void set_value([[maybe_unused]] _Args&&... __args) noexcept {
        // ignore incoming values
        STDEXEC::set_value(static_cast<_ItemReceiver&&>(__op_->__rcvr_));
      }

      template <class _Error>
        requires __variant_emplaceable<
                   _ResultVariant,
                   __decayed_tuple<set_error_t, _Error>,
                   set_error_t,
                   _Error
                 >
              && __callable<STDEXEC::set_stopped_t, _ItemReceiver>
      void set_error(_Error&& __error) noexcept {
        // store error and signal stop
        __op_->__result_->__emplace(set_error_t(), static_cast<_Error&&>(__error));
        STDEXEC::set_stopped(static_cast<_ItemReceiver&&>(__op_->__rcvr_));
      }

      void set_stopped() noexcept
        requires __variant_emplaceable<_ResultVariant, __decayed_tuple<set_stopped_t>, set_stopped_t>
              && __callable<set_stopped_t, _ItemReceiver>
      {
        // stop without error
        __op_->__result_->__emplace(set_stopped_t());
        STDEXEC::set_stopped(static_cast<_ItemReceiver&&>(__op_->__rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<_ItemReceiver> {
        return STDEXEC::get_env(__op_->__rcvr_);
      }
    };

    template <class _Sender, class _ItemReceiver, class _ResultVariant>
    struct __item_operation : __item_operation_base<_ItemReceiver, _ResultVariant> {
      using __base_t = __item_operation_base<_ItemReceiver, _ResultVariant>;
      using __item_receiver_t = __item_receiver<_ItemReceiver, _ResultVariant>;

      __item_operation(
        __result_type<_ResultVariant>* __parent,
        _Sender&& __sndr,
        _ItemReceiver __rcvr)
        noexcept(
          __nothrow_decay_copyable<_ItemReceiver>
          && __nothrow_connectable<_Sender, __item_receiver_t>)
        : __base_t{static_cast<_ItemReceiver&&>(__rcvr), __parent}
        , __op_{STDEXEC::connect(static_cast<_Sender&&>(__sndr), __item_receiver_t{this})} {
      }

      void start() & noexcept {
        STDEXEC::start(__op_);
      }

      connect_result_t<_Sender, __item_receiver_t> __op_;
    };

    template <class _Sender, class _ResultVariant>
    struct __item_sender {
      using sender_concept = STDEXEC::sender_t;
      using completion_signatures = STDEXEC::completion_signatures<set_value_t(), set_stopped_t()>;

      template <class _Self, class _Receiver>
      using __operation_t =
        __item_operation<__copy_cvref_t<_Self, _Sender>, _Receiver, _ResultVariant>;

      template <class _Receiver>
      using __item_receiver_t = __item_receiver<_Receiver, _ResultVariant>;

      template <
        __decays_to<__item_sender> _Self,
        STDEXEC::receiver_of<completion_signatures> _Receiver
      >
        requires sender_to<__copy_cvref_t<_Self, _Sender>, __item_receiver_t<_Receiver>>
      STDEXEC_EXPLICIT_THIS_BEGIN(auto connect)(this _Self&& __self, _Receiver __rcvr)
        -> __operation_t<_Self, _Receiver> {
        return {
          __self.__parent_,
          static_cast<_Self&&>(__self).__sender_,
          static_cast<_Receiver&&>(__rcvr)};
      }
      STDEXEC_EXPLICIT_THIS_END(connect)

      _Sender __sender_;
      __result_type<_ResultVariant>* __parent_;
    };

    template <class _Receiver, class _ResultVariant>
    struct __operation_base : __result_type<_ResultVariant> {
      STDEXEC_IMMOVABLE_NO_UNIQUE_ADDRESS
      _Receiver __rcvr_;
    };

    template <class _Receiver, class _ResultVariant>
    struct __receiver {
      using receiver_concept = STDEXEC::receiver_t;

      constexpr explicit __receiver(__operation_base<_Receiver, _ResultVariant>* __op) noexcept
        : __op_{__op} {
      }

      // nvc++ needs this destructor to be defined to avoid a codegen issue
      STDEXEC_PP_WHEN(STDEXEC_NVHPC(), ~__receiver(){})

      template <sender _Item>
      [[nodiscard]]
      auto set_next(_Item&& __item) & noexcept(__nothrow_decay_copyable<_Item>)
        -> __item_sender<__decay_t<_Item>, _ResultVariant> {
        return {static_cast<_Item&&>(__item), __op_};
      }

      void set_value() noexcept {
        __op_->__visit_result(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      template <class _Error>
      void set_error(_Error&& error) noexcept {
        STDEXEC::set_error(static_cast<_Receiver&&>(__op_->__rcvr_), static_cast<_Error&&>(error));
      }

      void set_stopped() noexcept {
        STDEXEC::set_stopped(static_cast<_Receiver&&>(__op_->__rcvr_));
      }

      auto get_env() const noexcept -> env_of_t<_Receiver> {
        return STDEXEC::get_env(__op_->__rcvr_);
      }

     private:
      __operation_base<_Receiver, _ResultVariant>* __op_;
    };

    template <class _Sigs>
    using __result_variant_ = __transform_completion_signatures_t<
      _Sigs,
      __mconst<__types<>>::__f,
      __mcompose_q<__types, __mbind_front_q<__decayed_tuple, set_error_t>::__f>::__f,
      __types<__tuple<set_stopped_t>>,
      __mconcat<__qq<__variant_for>>::__f
    >;

    template <class _Sender, class _Env>
    using __result_variant_t =
      __result_variant_<__sequence_completion_signatures_of_t<_Sender, _Env>>;

    template <class _Sender, class _Receiver>
    struct __operation
      : __operation_base<_Receiver, __result_variant_t<_Sender, env_of_t<_Receiver>>> {
      using _ResultVariant = __result_variant_t<_Sender, env_of_t<_Receiver>>;
      using __base_type = __operation_base<_Receiver, _ResultVariant>;
      using __receiver_t = __receiver<_Receiver, _ResultVariant>;

      explicit __operation(_Sender&& __sndr, _Receiver __rcvr)
        noexcept(__nothrow_subscribable<_Sender, __receiver_t>)
        : __base_type{{}, static_cast<_Receiver&&>(__rcvr)}
        , __op_{exec::subscribe(static_cast<_Sender&&>(__sndr), __receiver_t{this})} {
      }

      void start() & noexcept {
        STDEXEC::start(__op_);
      }

      subscribe_result_t<_Sender, __receiver_t> __op_;
    };

    struct __connect_fn {
      template <class _Receiver, class _Child>
      using __opstate_t = __operation<_Child, _Receiver>;

      template <class _Receiver, class _Child>
      auto operator()(_Receiver& __rcvr, __ignore, __ignore, _Child&& __child)
        noexcept(__nothrow_constructible_from<__opstate_t<_Receiver, _Child>, _Child, _Receiver>)
          -> __opstate_t<_Receiver, _Child> {
        return __opstate_t<_Receiver, _Child>{
          static_cast<_Child&&>(__child), static_cast<_Receiver&&>(__rcvr)};
      }
    };

    struct ignore_all_values_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        return __make_sexpr<ignore_all_values_t>(__(), static_cast<_Sender&&>(__sndr));
      }

      STDEXEC_ATTRIBUTE(always_inline)
      constexpr auto operator()() const noexcept {
        return __closure(*this);
      }
    };

    struct __ignore_all_values_impl : __sexpr_defaults {
      template <class _Sender, class... _Env>
      static consteval auto get_completion_signatures() {
        static_assert(sender_expr_for<_Sender, ignore_all_values_t>);
        return __sequence_completion_signatures_of<__child_of<_Sender>, _Env...>();
      }

      static constexpr auto connect =
        []<class _Sender, receiver _Receiver>(_Sender&& __sndr, _Receiver __rcvr) noexcept(
          __nothrow_applicable<__connect_fn, _Sender, _Receiver&>)
        -> __apply_result_t<__connect_fn, _Sender, _Receiver&> {
        static_assert(sender_expr_for<_Sender, ignore_all_values_t>);
        return __apply(__connect_fn(), static_cast<_Sender&&>(__sndr), __rcvr);
      };
    };
  } // namespace __ignore_all_values

  using __ignore_all_values::ignore_all_values_t;
  inline constexpr ignore_all_values_t ignore_all_values{};
} // namespace exec

namespace STDEXEC {
  template <>
  struct __sexpr_impl<exec::ignore_all_values_t>
    : exec::__ignore_all_values::__ignore_all_values_impl { };
} // namespace STDEXEC
