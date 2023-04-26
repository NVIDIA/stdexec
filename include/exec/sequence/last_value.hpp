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
  namespace __last_value {
    using namespace stdexec;

    template <class _ValuesVariant, class _ErrorsVariant>
    struct __shared_last_value_state {
      STDEXEC_NO_UNIQUE_ADDRESS _ValuesVariant __values_{};
      STDEXEC_NO_UNIQUE_ADDRESS _ErrorsVariant __errors_{};
      std::mutex __mutex_{};
      stdexec::in_place_stop_source __stop_source_{};
    };

    template <class _BaseEnv>
    using __env_t = __make_env_t<_BaseEnv, __with<get_stop_token_t, in_place_stop_token>>;

    template <class _NextRcvr, class _ValuesVariant, class _ErrorsVariant>
    struct __item_operation_base {
      __shared_last_value_state<_ValuesVariant, _ErrorsVariant>* __state_;
      STDEXEC_NO_UNIQUE_ADDRESS _NextRcvr __next_rcvr_;
    };

    template <class _NextRcvrId, class _ValuesVariant, class _ErrorsVariant>
    struct __item_receiver {
      struct __t {
        using __id = __item_receiver;
        using _NextRcvr = stdexec::__t<_NextRcvrId>;

        __item_operation_base<_NextRcvr, _ValuesVariant, _ErrorsVariant>* __op;

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_NextRcvr> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return get_env(__self.__op->__next_rcvr_);
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self, class... _Args>
          requires __callable<_SetValue, _NextRcvr&&>
        friend void tag_invoke(_SetValue, _Self&& __self, _Args&&... __args) noexcept {
          std::scoped_lock __lock{__self.__op->__state_->__mutex_};
          try {
            __self.__op->__state_->__values_.template emplace<__decayed_tuple<_Args...>>(
              static_cast<_Args&&>(__args)...);
            set_value(static_cast<_NextRcvr&&>(__self.__op->__next_rcvr_));
          } catch (...) {
            __self.__op->__state_->__errors_.template emplace<std::exception_ptr>(
              std::current_exception());
            __self.__op->__state_->__stop_source_.request_stop();
            set_stopped(static_cast<_NextRcvr&&>(__self.__op->__next_rcvr_));
          }
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __callable<set_stopped_t, _NextRcvr&&>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          std::scoped_lock __lock{__self.__op->__state_->__mutex_};
          try {
            __self.__op->__state_->__errors_.template emplace<__decay_t<_Error>>(
              static_cast<_Error&&>(__error));
          } catch (...) {
            __self.__op->__state_->__errors_.template emplace<std::exception_ptr>(
              std::current_exception());
          }
          __self.__op->__state_->__stop_source_.request_stop();
          set_stopped(static_cast<_NextRcvr&&>(__self.__op->__next_rcvr_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<set_stopped_t, _NextRcvr&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          __self.__op->__state_->__stop_source_.request_stop();
          set_stopped(static_cast<_NextRcvr&&>(__self.__op->__next_rcvr_));
        }
      };
    };

    template <class _Item, class _NextRcvrId, class _ValuesVariant, class _ErrorsVariant>
    struct __item_operation {
      using _NextRcvr = stdexec::__t<_NextRcvrId>;

      struct __t : __item_operation_base<_NextRcvr, _ValuesVariant, _ErrorsVariant> {
        using __id = __item_operation;
        using __item_receiver_t =
          stdexec::__t<__item_receiver<_NextRcvrId, _ValuesVariant, _ErrorsVariant>>;

        connect_result_t<_Item, __item_receiver_t> __op_;

        explicit __t(
          __shared_last_value_state<_ValuesVariant, _ErrorsVariant>* __state,
          _Item __item,
          _NextRcvr __next_rcvr)
          : __item_operation_base<
            _NextRcvr,
            _ValuesVariant,
            _ErrorsVariant>{__state, static_cast<_NextRcvr&&>(__next_rcvr)}
          , __op_{stdexec::connect(static_cast<_Item&&>(__item), __item_receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_);
        }
      };
    };

    template <class _ItemId, class _ValuesVariant, class _ErrorsVariant>
    struct __item_sender {
      struct __t {
        using __id = __item_sender;
        using _Item = stdexec::__t<_ItemId>;

        __shared_last_value_state<_ValuesVariant, _ErrorsVariant>* __state_;
        STDEXEC_NO_UNIQUE_ADDRESS _Item __item_;

        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

        template <class _Self, class _Rcvr>
        using __item_operation_t = stdexec::__t<__item_operation<
          __copy_cvref_t<_Self, _Item>,
          stdexec::__id<__decay_t<_Rcvr>>,
          _ValuesVariant,
          _ErrorsVariant>>;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _NextRcvr>
        friend auto tag_invoke(connect_t, _Self&& __self, _NextRcvr&& __next_rcvr) {
          return __item_operation_t<_Self, _NextRcvr>{
            __self.__state_,
            static_cast<_Item&&>(__self.__item_),
            static_cast<_NextRcvr&&>(__next_rcvr)};
        }
      };
    };

    template <class _Receiver, class _ValuesVariant, class _ErrorsVariant>
    struct __operation_base : __shared_last_value_state<_ValuesVariant, _ErrorsVariant> {
      STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
    };

    template <class _ReceiverId, class _ValuesVariant, class _ErrorsVariant>
    struct __receiver {
      struct __t {
        using __id = __receiver;
        using _Receiver = stdexec::__t<_ReceiverId>;
        __operation_base<_Receiver, _ValuesVariant, _ErrorsVariant>* __op_;

        template <class _Item>
        using __item_sender_t = stdexec::__t<
          __item_sender< stdexec::__id<__decay_t<_Item>>, _ValuesVariant, _ErrorsVariant>>;

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend __env_t<env_of_t<_Receiver>> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return __make_env(
            get_env(__self.__op_->__rcvr_),
            __with_(get_stop_token, __self.__op_->__stop_source_.get_token()));
        }

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, sender _Item>
        friend __item_sender_t<_Item> tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept {
          return {__self.__op_, static_cast<_Item&&>(__item)};
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          if (__self.__op_->__errors_.index()) {
            std::visit(
              [&__self]<class _Error>(_Error&& __error) noexcept {
                if constexpr (__not_decays_to<_Error, std::monostate>) {
                  set_error(
                    static_cast<_Receiver&&>(__self.__op_->__rcvr_),
                    static_cast<_Error&&>(__error));
                }
              },
              static_cast<_ErrorsVariant&&>(__self.__op_->__errors_));
          } else if (__self.__op_->__values_.index()) {
            std::visit(
              [&__self]<class _Value>(_Value&& __value) noexcept {
                if constexpr (__not_decays_to<_Value, std::monostate>) {
                  std::apply(
                    [&__self]<class... _Args>(_Args&&... __args) noexcept {
                      set_value(
                        static_cast<_Receiver&&>(__self.__op_->__rcvr_),
                        static_cast<_Args&&>(__args)...);
                    },
                    static_cast<_Value&&>(__value));
                }
              },
              static_cast<_ValuesVariant&&>(__self.__op_->__values_));
          } else {
            set_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
          }
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          set_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          set_error(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(__error));
        }
      };
    };

    template <class _Sender, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Env = __env_t<env_of_t<_Receiver>>;

      using _ValuesVariant =
        __value_types_of_t<_Sender, _Env, __q<__decayed_tuple>, __nullable_variant_t>;

      using _ErrorsVariant = __minvoke<
        __push_back_unique<__q<std::variant>>,
        __error_types_of_t<_Sender, _Env, __nullable_variant_t>,
        std::exception_ptr>;

      struct __t : __operation_base<_Receiver, _ValuesVariant, _ErrorsVariant> {
        using __id = __operation;

        using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _ValuesVariant, _ErrorsVariant>>;

        sequence_connect_result_t<_Sender, __receiver_t> __seq_op_;

        __t(_Sender __sndr, _Receiver __rcvr)
          : __operation_base<_Receiver, _ValuesVariant, _ErrorsVariant>{{}, static_cast<_Receiver&&>(__rcvr)}
          , __seq_op_{exec::sequence_connect(static_cast<_Sender&&>(__sndr), __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__seq_op_);
        }
      };
    };

    template <class _SenderId>
    struct __sender {
      struct __t {
        using __id = __sender;
        using _Sender = stdexec::__t<_SenderId>;

        template <class _Rcvr>
        using _ValuesVariant = __value_types_of_t<
          _Sender,
          __env_t<env_of_t<_Rcvr>>,
          __q<__decayed_tuple>,
          __nullable_variant_t>;

        template <class _Rcvr>
        using _ErrorsVariant =
          __error_types_of_t<_Sender, __env_t<env_of_t<_Rcvr>>, __nullable_variant_t>;

        template <class _Self, class _Rcvr>
        using __operation_t =
          stdexec::__t<__operation<__copy_cvref_t<_Self, _Sender>, stdexec::__id<__decay_t<_Rcvr>>>>;

        template <class _Rcvr>
        using __receiver_t = stdexec::__t<
          __receiver<stdexec::__id<__decay_t<_Rcvr>>, _ValuesVariant<_Rcvr>, _ErrorsVariant<_Rcvr>>>;

        template <class _Self, class _Env>
        using __completion_sigs = make_completion_signatures<__copy_cvref_t<_Self, _Sender>, _Env>;

        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;

        template <__decays_to<__t> _Self, queryable _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<_Self, _Env>;

        template <__decays_to<__t> _Self, receiver _Rcvr>
          requires receiver_of<_Rcvr, __completion_sigs<_Self, __env_t<env_of_t<_Rcvr>>>>
                && sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Rcvr>>
        friend __operation_t<_Self, _Rcvr> tag_invoke(connect_t, _Self&& __self, _Rcvr&& __rcvr) {
          return __operation_t<_Self, _Rcvr>{
            static_cast<_Self&&>(__self).__sndr_, static_cast<_Rcvr&&>(__rcvr)};
        }
      };
    };

    struct last_value_t {
      template <class _Sender>
      using __sender_t = __t<__sender<__id<__decay_t<_Sender>>>>;

      template <sender _Sender>
      __sender_t<_Sender> operator()(_Sender&& __sndr) const {
        return {static_cast<_Sender&&>(__sndr)};
      }
    };
  } // namespace __last_value

  using __last_value::last_value_t;
  inline constexpr last_value_t last_value{};
} // namespace exec