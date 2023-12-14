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

#include "../sequence_senders.hpp"

namespace exec {
  template <class _Variant, class _Type, class... _Args>
  concept __variant_emplaceable = requires(_Variant& __v, _Args&&... __args) {
    __v.template emplace<_Type>(static_cast<_Args&&>(__args)...);
  };

  template <class _Variant, class _Type, class... _Args>
  concept __nothrow_variant_emplaceable = requires(_Variant& __v, _Args&&... __args) {
    { __v.template emplace<_Type>(static_cast<_Args&&>(__args)...) } noexcept;
  };

  namespace __ignore_all_values {
    using namespace stdexec;

    template <class _ResultVariant>
    struct __result_type {
      _ResultVariant __result_{};
      std::atomic<int> __emplaced_{0};

      template <class... _Args>
      void __emplace(_Args&&... __args) noexcept {
        int __expected = 0;
        if (__emplaced_.compare_exchange_strong(__expected, 1, std::memory_order_relaxed)) {
          __result_.template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
          __emplaced_.store(2, std::memory_order_release);
        }
      }

      template <class _Receiver>
      void __visit_result(_Receiver&& __rcvr) noexcept {
        int __is_emplaced = __emplaced_.load(std::memory_order_acquire);
        if (__is_emplaced == 0) {
          stdexec::set_value(static_cast<_Receiver&&>(__rcvr));
          return;
        }
        std::visit(
          [&]<class _Tuple>(_Tuple&& __tuple) noexcept {
            if constexpr (__not_decays_to<_Tuple, std::monostate>) {
              std::apply(
                [&]<__completion_tag _Tag, class... _Args>(
                  _Tag __completion, _Args&&... __args) noexcept {
                  __completion(static_cast<_Receiver&&>(__rcvr), static_cast<_Args&&>(__args)...);
                },
                static_cast<_Tuple&&>(__tuple));
            }
          },
          static_cast<_ResultVariant&&>(__result_));
      }
    };

    template <class _ItemReceiver, class _ResultVariant>
    struct __item_operation_base {
      STDEXEC_ATTRIBUTE((no_unique_address)) _ItemReceiver __receiver_;
      __result_type<_ResultVariant>* __result_;
    };

    template <class _ItemReceiver, class _ResultVariant>
    struct __item_receiver {
      struct __t {
        using __id = __item_receiver;
        using receiver_concept = stdexec::receiver_t;
        __item_operation_base<_ItemReceiver, _ResultVariant>* __op_;

        template <same_as<set_value_t> _Tag, same_as<__t> _Self, class... _Args>
        friend void tag_invoke(_Tag, _Self&& __self, [[maybe_unused]] _Args&&... __args) noexcept {
          // ignore incoming values
          stdexec::set_value(static_cast<_ItemReceiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<set_error_t> _Tag, same_as<__t> _Self, class _Error>
          requires __variant_emplaceable< _ResultVariant, __decayed_tuple<_Tag, _Error>, _Tag, _Error>
                && __callable<stdexec::set_stopped_t, _ItemReceiver&&>
        friend void tag_invoke(_Tag, _Self&& __self, _Error&& __error) noexcept {
          // store error and signal stop
          __self.__op_->__result_->__emplace(_Tag{}, static_cast<_Error&&>(__error));
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<set_stopped_t> _Tag, same_as<__t> _Self>
          requires __variant_emplaceable< _ResultVariant, __decayed_tuple<_Tag>, _Tag>
                && __callable<_Tag, _ItemReceiver&&>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          // stop without error
          __self.__op_->__result_->__emplace(_Tag{});
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
        friend env_of_t<_ItemReceiver> tag_invoke(_GetEnv, _Self&& __self) noexcept {
          return stdexec::get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _Sender, class _ItemReceiver, class _ResultVariant>
    struct __item_operation : __item_operation_base<_ItemReceiver, _ResultVariant> {
      using __base_type = __item_operation_base<_ItemReceiver, _ResultVariant>;
      using __item_receiver_t = stdexec::__t<__item_receiver<_ItemReceiver, _ResultVariant>>;

      struct __t : __base_type {
        connect_result_t<_Sender, __item_receiver_t> __op_;

        __t(
          __result_type<_ResultVariant>* __parent,
          _Sender&& __sndr,
          _ItemReceiver __rcvr)                            //
          noexcept(__nothrow_decay_copyable<_ItemReceiver> //
                     && __nothrow_connectable<_Sender, __item_receiver_t>)
          : __base_type{static_cast<_ItemReceiver&&>(__rcvr), __parent}
          , __op_{stdexec::connect(static_cast<_Sender&&>(__sndr), __item_receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _Sender, class _ResultVariant>
    struct __item_sender {
      struct __t {
        using sender_concept = stdexec::sender_t;
        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

        template <class _Self, class _Receiver>
        using __operation_t =
          stdexec::__t<__item_operation<__copy_cvref_t<_Self, _Sender>, _Receiver, _ResultVariant>>;

        template <class _Receiver>
        using __item_receiver_t = stdexec::__t<__item_receiver<_Receiver, _ResultVariant>>;

        _Sender __sender_;
        __result_type<_ResultVariant>* __parent_;

        template < __decays_to<__t> _Self, stdexec::receiver_of<completion_signatures> _Receiver>
          requires sender_to<__copy_cvref_t<_Self, _Sender>, __item_receiver_t<_Receiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Receiver __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return {
            __self.__parent_,
            static_cast<_Self&&>(__self).__sender_,
            static_cast<_Receiver&&>(__rcvr)};
        }
      };
    };

    template <class _Receiver, class _ResultVariant>
    struct __operation_base : __result_type<_ResultVariant> {
      STDEXEC_ATTRIBUTE((no_unique_address)) _Receiver __receiver_;
    };

    template <class _ReceiverId, class _ResultVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        using receiver_concept = stdexec::receiver_t;
        __operation_base<_Receiver, _ResultVariant>* __op_;

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, sender _Item>
        friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __item) //
          noexcept(__nothrow_decay_copyable<_Item>)
            -> stdexec::__t<__item_sender<__decay_t<_Item>, _ResultVariant>> {
          return {static_cast<_Item&&>(__item), __self.__op_};
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          __self.__op_->__visit_result(static_cast<_Receiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op_->__receiver_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__self.__op_->__receiver_), static_cast<_Error&&>(error));
        }

        template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(_GetEnv, _Self&& __self) noexcept {
          return stdexec::get_env(__self.__op_->__receiver_);
        }
      };
    };

    template <class _Tag, class _Sigs>
    using __gather_types =
      __gather_signal<_Tag, _Sigs, __mbind_front_q<__decayed_tuple, _Tag>, __q<__types>>;

    template <class _Sigs>
    using __result_variant_ = __minvoke<
      __mconcat<__nullable_variant_t>,
      __gather_types<set_error_t, _Sigs>,
      __gather_types<set_stopped_t, _Sigs>>;

    template <class _Sender, class _Env>
    using __result_variant_t =
      __result_variant_<__sequence_completion_signatures_of_t<_Sender, _Env>>;

    template <class _Sender, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t : __operation_base<_Receiver, __result_variant_t<_Sender, env_of_t<_Receiver>>> {
        using _ResultVariant = __result_variant_t<_Sender, env_of_t<_Receiver>>;
        using __base_type = __operation_base<_Receiver, _ResultVariant>;
        using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _ResultVariant>>;

        subscribe_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver __rcvr) //
          noexcept(__nothrow_decay_copyable<_Receiver>)
          : __base_type{{}, static_cast<_Receiver&&>(__rcvr)}
          , __op_{exec::subscribe(static_cast<_Sender&&>(__sndr), __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _Receiver>
    struct __connect_fn {
      _Receiver& __rcvr_;

      using _ReceiverId = __id<_Receiver>;
      using _Env = env_of_t<_Receiver>;

      template <class _Child>
      using __operation_t = stdexec::__t<__operation<_Child, _ReceiverId>>;

      template <class _Child>
      using _ResultVariant = __result_variant_t<_Child, _Env>;

      template <class _Child>
      using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _ResultVariant<_Child>>>;

      template <class _Child>
      using __completion_sigs = __sequence_completion_signatures_of_t<_Child, _Env>;

      template <class _Child>
        requires receiver_of<_Receiver, __completion_sigs<_Child>>
              && sequence_sender_to<_Child, __receiver_t<_Child>>
      __operation_t<_Child> operator()(__ignore, __ignore, _Child&& __child) noexcept(
        __nothrow_constructible_from< __operation_t<_Child>, _Child, _Receiver>) {
        return {static_cast<_Child&&>(__child), static_cast<_Receiver&&>(__rcvr_)};
      }
    };

    struct ignore_all_values_t {
      template <sender _Sender>
      auto operator()(_Sender&& __sndr) const {
        auto __domain = __get_early_domain((_Sender&&) __sndr);
        return transform_sender(
          __domain, __make_sexpr<ignore_all_values_t>(__(), (_Sender&&) __sndr));
      }

      constexpr __binder_back<ignore_all_values_t> operator()() const noexcept {
        return {{}, {}, {}};
      }
    };

    struct __ignore_all_values_impl : __sexpr_defaults {
      template <class _Sequence, class _Env>
      using __completion_sigs = __sequence_completion_signatures_of_t<_Sequence, _Env>;

      static constexpr auto get_completion_signatures = //
        []<class _Sender, class _Env>(_Sender&& __sndr, _Env&&)
          -> __completion_sigs<__child_of<_Sender>, _Env> {
          static_assert(sender_expr_for<_Sender, ignore_all_values_t>);
          return {};
        };

      template <class _Child, class _Receiver>
      using _ResultVariant = __result_variant_t<_Child, env_of_t<_Receiver>>;

      template <class _Child, class _Receiver>
      using __receiver_t = __t<__receiver<__id<_Receiver>, _ResultVariant<_Child, _Receiver>>>;

      static constexpr auto connect = //
        []<class _Sender, receiver _Receiver>(_Sender&& __sndr, _Receiver __rcvr) noexcept(
        __nothrow_callable<__sexpr_apply_t, _Sender, __connect_fn<_Receiver>>)
        -> __call_result_t<__sexpr_apply_t, _Sender, __connect_fn<_Receiver>>
        requires receiver_of<_Receiver, __completion_sigs<__child_of<_Sender>, env_of_t<_Receiver>>>
              && sequence_sender_to<
                   __child_of<_Sender>,
                   __receiver_t<__child_of<_Sender>, _Receiver>> {
          static_assert(sender_expr_for<_Sender, ignore_all_values_t>);
        return __sexpr_apply((_Sender&&) __sndr, __connect_fn<_Receiver>{__rcvr});
      };
    };
  }

  using __ignore_all_values::ignore_all_values_t;
  inline constexpr ignore_all_values_t ignore_all_values{};
}

namespace stdexec {
  template <>
  struct __sexpr_impl<exec::ignore_all_values_t> : exec::__ignore_all_values::__ignore_all_values_impl {};
}