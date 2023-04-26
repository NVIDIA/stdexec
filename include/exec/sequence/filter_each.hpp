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

#include "./detail/__shared_value_sender.hpp"

namespace exec {
  namespace __filter_each {
    using namespace stdexec;

    template <class _ReceiverId, class _Predicate>
    struct __operation_base {
      [[no_unique_address]] __t<_ReceiverId> __rcvr_;
      [[no_unique_address]] _Predicate __pred_;
    };

    template <class _ValuesVariant, class _NextReceiverId, class _ReceiverId, class _Predicate>
    struct __item_operation_base : __shared::__value_state<_ValuesVariant> {
      using _NextReceiver = stdexec::__t<_NextReceiverId>;
      using _NextSender = __next_sender_of_t<
        stdexec::__t<_ReceiverId>&,
        __shared::__demat_t<stdexec::__t<__shared::__value_sender<_ValuesVariant>>>>;
      using __receiver_ref_t = stdexec::__t<__shared::__receiver_ref<_NextReceiverId>>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      STDEXEC_NO_UNIQUE_ADDRESS _NextReceiver __next_rcvr_;
      connect_result_t<_NextSender, __receiver_ref_t> __next_;
      __operation_base<_ReceiverId, _Predicate>* __base_op_;

      explicit __item_operation_base(
        __operation_base<_ReceiverId, _Predicate>* __base_op,
        _NextReceiver&& __next_rcvr)
        : __next_rcvr_(static_cast<_NextReceiver&&>(__next_rcvr))
        , __next_(stdexec::connect(
            exec::set_next(
              __base_op->__rcvr_,
              exec::dematerialize(stdexec::__t<__shared::__value_sender<_ValuesVariant>>{this})),
            __receiver_ref_t{&__next_rcvr_}))
        , __base_op_{__base_op} {
      }
    };

    template <class _ValuesVariant, class _NextReceiverId, class _ReceiverId, class _Predicate>
    struct __item_receiver {
      struct __t {
        using __id = __item_receiver;
        using _NextReceiver = stdexec::__t<_NextReceiverId>;
        __item_operation_base<_ValuesVariant, _NextReceiverId, _ReceiverId, _Predicate>* __op_;

        template <
          same_as<set_value_t> _SetValue,
          same_as<__t> _Self,
          __completion_tag _Tag,
          class... _Args>
          requires __callable<set_value_t, _NextReceiver&&>
        friend void tag_invoke(_SetValue, _Self&& __self, _Tag, _Args&&... __args) noexcept {
          try {
            __self.__op_->__values_.template emplace<__decayed_tuple<_Tag, _Args...>>(
              _Tag{}, static_cast<_Args&&>(__args)...);
            if constexpr (same_as<set_value_t, _Tag>) {
              if (std::invoke(__self.__op_->__base_op_->__pred_, __args...)) {
                stdexec::start(__self.__op_->__next_);
              } else {
                stdexec::set_value(static_cast<_NextReceiver&&>(__self.__op_->__next_rcvr_));
              }
            } else {
              stdexec::start(__self.__op_->__next_);
            }
          } catch (...) {
            __self.__op_->__values_.template emplace<std::tuple<set_error_t, std::exception_ptr>>(
              stdexec::set_error, std::current_exception());
            stdexec::start(__self.__op_->__next_);
          }
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend auto tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return stdexec::get_env(__self.__op_->__next_rcvr_);
        }
      };
    };

    template <class _ItemSender, class _NextReceiverId, class _ReceiverId, class _Predicate>
    struct __item_operation {
      using _NextReceiver = stdexec::__t<_NextReceiverId>;
      using _Env = env_of_t<_NextReceiver>;
      using _ValuesVariant = __minvoke<
        __mconcat<__nullable_variant_t>,
        __types<std::tuple<set_error_t, std::exception_ptr>>,
        __value_types_of_t< _ItemSender, _Env>>;

      struct __t : __item_operation_base<_ValuesVariant, _NextReceiverId, _ReceiverId, _Predicate> {
        using __item_receiver_t =
          stdexec::__t<__item_receiver<_ValuesVariant, _NextReceiverId, _ReceiverId, _Predicate>>;

        connect_result_t<_ItemSender, __item_receiver_t> __receive_item_;

        explicit __t(
          _ItemSender __item,
          _NextReceiver __next_rcvr,
          __operation_base<_ReceiverId, _Predicate>* __base_op)
          : __item_operation_base<
            _ValuesVariant,
            _NextReceiverId,
            _ReceiverId,
            _Predicate>{__base_op, static_cast<_NextReceiver&&>(__next_rcvr)}
          , __receive_item_{
              stdexec::connect(static_cast<_ItemSender&&>(__item), __item_receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__receive_item_);
        }
      };
    };

    template <class _ItemSenderId, class _ReceiverId, class _Predicate>
    struct __item_sender {
      struct __t {
        using __id = __item_sender;
        using _ItemSender = stdexec::__t<_ItemSenderId>;
        using _Receiver = stdexec::__t<_ReceiverId>;
        STDEXEC_NO_UNIQUE_ADDRESS _ItemSender __item;
        __operation_base<_ReceiverId, _Predicate>* __base_op_;

        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

        template <class _Self, class _NextRcvr>
        using __item_operation_t = stdexec::__t< __item_operation<
          __copy_cvref_t<_Self, _ItemSender>,
          stdexec::__id<__decay_t<_NextRcvr>>,
          _ReceiverId,
          _Predicate>>;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _NextRcvr>
          requires sequence_receiver_of<
            _Receiver,
            completion_signatures_of_t<_ItemSender, env_of_t<_NextRcvr>>>
        friend auto tag_invoke(connect_t, _Self&& __self, _NextRcvr __next_rcvr)
          -> __item_operation_t<_Self, _NextRcvr> {
          return __item_operation_t<_Self, _NextRcvr>(
            static_cast<_Self&&>(__self).__item,
            static_cast<_NextRcvr&&>(__next_rcvr),
            __self.__base_op_);
        }
      };
    };

    template <class _ReceiverId, class _Predicate>
    struct __receiver {
      struct __t {
        using __id = __receiver;
        using _Receiver = stdexec::__t<_ReceiverId>;
        template <class _Item>
        using __item_sender_t =
          stdexec::__t< __item_sender<stdexec::__id<__shared::__mat_t<_Item>>, _ReceiverId, _Predicate>>;

        __operation_base<_ReceiverId, _Predicate>* __op_;

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, sender _Item>
        friend __item_sender_t<_Item> tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept {
          return __item_sender_t<_Item>{
            exec::materialize(static_cast<_Item&&>(__item)), __self.__op_};
        }

        template <same_as<set_value_t> _SetValue, __decays_to<__t> _Self>
          requires __callable<_SetValue, _Receiver&&>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          _SetValue{}((_Receiver&&) __self.__op_->__rcvr_);
        }

        template <same_as<set_stopped_t> _SetStopped, __decays_to<__t> _Self>
          requires __callable<_SetStopped, _Receiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          _SetStopped{}((_Receiver&&) __self.__op_->__rcvr_);
        }

        template <same_as<set_error_t> _SetError, __decays_to<__t> _Self, class _Error>
          requires __callable<_SetError, _Receiver&&, _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          _SetError{}((_Receiver&&) __self.__op_->__rcvr_, (_Error&&) __error);
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _Predicate>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _Predicate>>;
      using __op_base_t = __operation_base<_ReceiverId, _Predicate>;

      struct __t : __op_base_t {
        sequence_connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver&& __rcvr, _Predicate __pred)
          : __op_base_t{(_Receiver&&) __rcvr, (_Predicate&&) __pred}
          , __op_{exec::sequence_connect((_Sender&&) __sndr, __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _SenderId, class _Predicate>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <class _Self, class _Receiver>
      using __operation_t = stdexec::__t<
        __operation<__copy_cvref_t<_Self, _Sender>, __id<__decay_t<_Receiver>>, _Predicate>>;

      template <class _Receiver>
      using __receiver_t = stdexec::__t<__receiver<__id<__decay_t<_Receiver>>, _Predicate>>;

      struct __t {
        using __id = __sender;
        using is_sequence_sender = void;

        _Sender __sndr_;
        _Predicate __pred_;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Receiver>>
        friend auto tag_invoke(sequence_connect_t, _Self&& __self, _Receiver&& __rcvr)
          -> __operation_t<_Self, _Receiver> {
          return __operation_t<_Self, _Receiver>(
            ((_Self&&) __self).__sndr_, (_Receiver&&) __rcvr, (_Predicate&&) __self.__pred_);
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> completion_signatures_of_t<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    using namespace stdexec;

    struct filter_each_t {
      template <class _Sender, class _Predicate>
        requires tag_invocable<filter_each_t, _Sender, _Predicate>
      auto operator()(_Sender&& __sndr, _Predicate __pred) const
        noexcept(nothrow_tag_invocable<filter_each_t, _Sender, _Predicate>)
          -> tag_invoke_result_t<filter_each_t, _Sender, _Predicate> {
        return tag_invoke(*this, (_Sender&&) __sndr, (_Predicate&&) __pred);
      }

      template <sender _Sender, class _Predicate>
        requires(!tag_invocable<filter_each_t, _Sender, _Predicate>)
      auto operator()(_Sender&& __sndr, _Predicate __pred) const
        -> __t<__sender<__id<__decay_t<_Sender>>, _Predicate>> {
        return {(_Sender&&) __sndr, (_Predicate&&) __pred};
      }

      template <class _Predicate>
      auto operator()(_Predicate __pred) const -> __binder_back<filter_each_t, _Predicate> {
        return {{}, {}, {(_Predicate&&) __pred}};
      }
    };
  }

  using __filter_each::filter_each_t;
  inline constexpr filter_each_t filter_each{};
}