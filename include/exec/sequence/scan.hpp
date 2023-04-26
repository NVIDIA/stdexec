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
  namespace __scan {
    using namespace stdexec;

    template <class _Ty, class _Fn>
    struct __operation_base {
      STDEXEC_NO_UNIQUE_ADDRESS _Ty __value_;
      STDEXEC_NO_UNIQUE_ADDRESS _Fn __fn_;
      std::mutex __mutex_{};
    };

    template <class _ItemReceiver, class _Ty, class _Fn>
    struct __item_operation_base {
      STDEXEC_NO_UNIQUE_ADDRESS _ItemReceiver __item_receiver_;
      __operation_base<_Ty, _Fn>* __seq_op_;
    };

    template <class _ItemReceiverId, class _Ty, class _Fn>
    struct __item_receiver {
      struct __t {
        using __id = __item_receiver;
        using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
        __item_operation_base<_ItemReceiver, _Ty, _Fn>* __op_;

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_ItemReceiver> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return get_env(__self.__op_->__item_receiver_);
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self, class... _Args>
          requires __callable<_Fn&, _Ty&, _Args...>
                && __callable<set_value_t, _ItemReceiver&&, const _Ty&>
                && __callable<set_error_t, _ItemReceiver&&, std::exception_ptr>
        friend void tag_invoke(_SetValue, _Self&& __self, _Args&&... __args) noexcept {
          try {
            std::scoped_lock __lock(__self.__op_->__seq_op_->__mutex_);
            __self.__op_->__seq_op_->__value_ = __self.__op_->__seq_op_->__fn_(
              static_cast<_Ty&&>(__self.__op_->__seq_op_->__value_),
              static_cast<_Args&&>(__args)...);
            set_value(
              static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_),
              static_cast<const _Ty&>(__self.__op_->__seq_op_->__value_));
          } catch (...) {
            set_error(
              static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_),
              std::current_exception());
          }
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __callable<set_error_t, _ItemReceiver&&, _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          set_error(
            static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_),
            static_cast<_Error&&>(__error));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<set_stopped_t, _ItemReceiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
        }
      };
    };

    template <class _ItemSender, class _ItemReceiverId, class _Ty, class _Fn>
    struct __item_operation {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t : __item_operation_base<_ItemReceiver, _Ty, _Fn> {
        using __id = __item_operation;

        using __item_receiver_t = stdexec::__t<__item_receiver<_ItemReceiverId, _Ty, _Fn>>;
        __operation_base<_Ty, _Fn>* __sequence_op_;
        connect_result_t<_ItemSender, __item_receiver_t> __op_;

        explicit __t(
          __operation_base<_Ty, _Fn>* __sequence_op,
          _ItemSender __sndr,
          _ItemReceiver __rcvr)
          : __item_operation_base<
            _ItemReceiver,
            _Ty,
            _Fn>{static_cast<_ItemReceiver&&>(__rcvr), __sequence_op}
          , __op_{stdexec::connect(static_cast<_ItemSender&&>(__sndr), __item_receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_);
        }
      };
    };

    template <class _ItemSenderId, class _Ty, class _Fn>
    struct __item_sender {
      struct __t {
        using __id = __item_sender;
        using _ItemSender = stdexec::__t<_ItemSenderId>;
        STDEXEC_NO_UNIQUE_ADDRESS _ItemSender __sndr_;
        __operation_base<_Ty, _Fn>* __op_;

        template <class _Self, class _Env>
        using __completion_sigs = __try_make_completion_signatures<
          __copy_cvref_t<_Self, _ItemSender>,
          _Env,
          completion_signatures<set_error_t(std::exception_ptr)>,
          __mconst<completion_signatures<set_value_t(const _Ty&)>>>;

        template <class _ItemRcvr>
        using __item_receiver_t =
          stdexec::__t<__item_receiver<stdexec::__id<__decay_t<_ItemRcvr>>, _Ty, _Fn>>;

        template <class _Self, class _ItemRcvr>
        using __item_operation_t = stdexec::__t<__item_operation<
          __copy_cvref_t<_Self, _ItemSender>,
          stdexec::__id<__decay_t<_ItemRcvr>>,
          _Ty,
          _Fn>>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<_Self, _Env>;

        template <same_as<connect_t> _Connect, __decays_to<__t> _Self, receiver _ItemRcvr>
          requires receiver_of<_ItemRcvr, __completion_sigs<_Self, env_of_t<_ItemRcvr>>>
                && sender_to<__copy_cvref_t<_Self, _ItemSender>, __item_receiver_t<_ItemRcvr>>
        friend auto tag_invoke(_Connect, _Self&& __self, _ItemRcvr&& __rcvr)
          -> __item_operation_t<_Self, _ItemRcvr> {
          return __item_operation_t<_Self, _ItemRcvr>{
            __self.__op_, static_cast<_Self&&>(__self).__sndr_, static_cast<_ItemRcvr&&>(__rcvr)};
        }
      };
    };

    template <class _Receiver, class _Ty, class _Fn>
    struct __operation_base_with_receiver : __operation_base<_Ty, _Fn> {
      STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
    };

    template <class _ReceiverId, class _Ty, class _Fn>
    struct __receiver {
      struct __t {
        using __id = __receiver;
        using _Receiver = stdexec::__t<_ReceiverId>;
        __operation_base_with_receiver<_Receiver, _Ty, _Fn>* __op_;

        template <class _Item>
        using __item_sender_t =
          stdexec::__t<__item_sender<stdexec::__id<__decay_t<_Item>>, _Ty, _Fn>>;

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_Receiver> tag_invoke(_GetEnv, const _Self& __self) noexcept {
          return get_env(__self.__op_->__rcvr_);
        }

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, sender _Item>
          requires __callable<set_next_t, _Receiver&, _Item>
        friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept {
          return set_next(
            __self.__op_->__rcvr_,
            __item_sender_t<_Item>{static_cast<_Item&&>(__item), __self.__op_});
        }

        template <__completion_tag _Tag, same_as<__t> _Self, class... _Args>
          requires __callable<_Tag, _Receiver&&, _Args...>
        friend void tag_invoke(_Tag, _Self&& __self, _Args&&... __args) noexcept {
          _Tag{}(static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Args&&>(__args)...);
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _Ty, class _Fn>
    struct __operation {
      struct __t : __operation_base_with_receiver<stdexec::__t<_ReceiverId>, _Ty, _Fn> {
        using __id = __operation;
        using _Receiver = stdexec::__t<_ReceiverId>;
        using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _Ty, _Fn>>;

        sequence_connect_result_t<_Sender, __receiver_t> __op_;

        explicit __t(_Sender __sndr, _Receiver __rcvr, _Ty __init, _Fn __fn)
        : __operation_base_with_receiver<_Receiver, _Ty, _Fn>{{static_cast<_Ty&&>(__init), static_cast<_Fn&&>(__fn)}, static_cast<_Receiver&&>(__rcvr)}
        , __op_{sequence_connect(static_cast<_Sender&&>(__sndr), __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          start(__self.__op_);
        }
      };
    };

    template <class _SenderId, class _Ty, class _Fn>
    struct __sender {
      struct __t {
        using __id = __sender;
        using _Sender = stdexec::__t<_SenderId>;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;
        STDEXEC_NO_UNIQUE_ADDRESS _Ty __init_;
        STDEXEC_NO_UNIQUE_ADDRESS _Fn __fn_;

        template <class _Self, class _Env>
        using __completion_sigs = __try_make_completion_signatures<
          __copy_cvref_t<_Self, _Sender>,
          _Env,
          completion_signatures<set_error_t(std::exception_ptr)>,
          __mconst<completion_signatures<set_value_t(const _Ty&)>>>;

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<_Self, _Env>;

        template <class _Self, class _Rcvr>
        using __operation_t = stdexec::__t<
          __operation<__copy_cvref_t<_Self, _Sender>, stdexec::__id<__decay_t<_Rcvr>>, _Ty, _Fn>>;

        template <class _Rcvr>
        using __receiver_t = stdexec::__t<__receiver<stdexec::__id<__decay_t<_Rcvr>>, _Ty, _Fn>>;

        template <
          same_as<sequence_connect_t> _SequenceConnect,
          __decays_to<__t> _Self,
          receiver _Rcvr>
          requires sequence_receiver_of<_Rcvr, __completion_sigs<_Self, env_of_t<_Rcvr>>>
                && sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Rcvr>>
        friend __operation_t<_Self, _Rcvr>
          tag_invoke(_SequenceConnect, _Self&& __self, _Rcvr&& __rcvr) {
          return __operation_t<_Self, _Rcvr>{
            static_cast<_Self&&>(__self).__sndr_,
            static_cast<_Rcvr&&>(__rcvr),
            static_cast<_Self&&>(__self).__init_,
            static_cast<_Self&&>(__self).__fn_};
        }
      };
    };

    struct scan_t {
      template <class _Sender, class _Ty, class _Fn>
      using __sender_t = __t<__sender<__id<__decay_t<_Sender>>, __decay_t<_Ty>, __decay_t<_Fn>>>;

      template <sender _Sender, __movable_value _Ty, __movable_value _Fn>
      __sender_t<_Sender, _Ty, _Fn> operator()(_Sender&& __sender, _Ty&& __init, _Fn&& __fn) const {
        return {
          static_cast<_Sender&&>(__sender), static_cast<_Ty&&>(__init), static_cast<_Fn&&>(__fn)};
      }
    };
  }

  using __scan::scan_t;
  inline constexpr scan_t scan{};
}