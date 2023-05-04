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
  namespace __ignore_all {
    using namespace stdexec;

    enum class __error_state {
      __none,
      __emplace,
      __emplace_done,
    };

    struct not_an_error { };

    template <class _Error, class _ErrorsVariant>
    concept __is_valid_error =
      __not_decays_to<_Error, not_an_error>
      && requires(_ErrorsVariant& __variant_, _Error&& __error) {
           __variant_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__error));
         };

    template <class _Rcvr>
    struct __error_visitor {
      _Rcvr __rcvr;

      template <class _Error>
      void operator()(_Error&& __error) noexcept {
        if constexpr (__not_decays_to<_Error, not_an_error>) {
          stdexec::set_error(static_cast<_Rcvr&&>(__rcvr), static_cast<_Error&&>(__error));
        } else {
          STDEXEC_ASSERT(false);
        }
      }
    };

    template <class _ErrorsVariant>
    struct __error_storage {
      STDEXEC_NO_UNIQUE_ADDRESS _ErrorsVariant __variant_{};
      std::atomic<__error_state> __state_{__error_state::__none};

      template <__is_valid_error<_ErrorsVariant> _Error>
      void __emplace(_Error&& __error) {
        __error_state __expected = __error_state::__none;
        if (__state_.compare_exchange_strong(
              __expected, __error_state::__emplace, std::memory_order_relaxed)) {
          __variant_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__error));
          __state_.store(__error_state::__emplace_done, std::memory_order_release);
        }
      }

      template <class _Rcvr>
      bool __visit_if_error(_Rcvr&& __rcvr) noexcept {
        __error_state __state = __state_.load(std::memory_order_acquire);
        switch (__state) {
        case __error_state::__none:
          return false;
        case __error_state::__emplace_done:
          std::visit(
            __error_visitor<_Rcvr>{static_cast<_Rcvr&&>(__rcvr)},
            static_cast<_ErrorsVariant&&>(__variant_));
          return true;
        case __error_state::__emplace:
          [[fallthrough]];
        default:
          STDEXEC_ASSERT(false);
        }
        return false;
      }
    };

    template <>
    struct __error_storage<std::variant<not_an_error>> {
      template <class _Rcvr>
      std::false_type __visit_if_error(_Rcvr&&) const noexcept {
        return {};
      }
    };

    template <class _ErrorsVariant>
    struct __operation_base {
      STDEXEC_NO_UNIQUE_ADDRESS __error_storage<_ErrorsVariant> __error_{};
    };

    template <class _ItemReceiver, class _ErrorsVariant>
    struct __item_operation_base {
      __operation_base<_ErrorsVariant>* __seq_op_;
      STDEXEC_NO_UNIQUE_ADDRESS _ItemReceiver __rcvr_;
    };

    template <class _ItemReceiverId, class _ErrorsVariant>
    struct __item_receiver {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __item_receiver;
        __item_operation_base<_ItemReceiver, _ErrorsVariant>* __op_;

        // We eat all arguments
        template <same_as<set_value_t> _SetValue, same_as<__t> _Self, class... _Args>
          requires __callable<_SetValue, _ItemReceiver&&>
        friend void tag_invoke(_SetValue, _Self&& __self, _Args&&...) noexcept {
          _SetValue{}(static_cast<_ItemReceiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __is_valid_error<_Error, _ErrorsVariant>
                && __callable<set_stopped_t, _ItemReceiver&&>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__seq_op_->__error_.__emplace(static_cast<_Error&&>(__error));
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<_SetStopped, _ItemReceiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          _SetStopped{}(static_cast<_ItemReceiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_ItemReceiver> tag_invoke(_GetEnv, const __t& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Item, class _ItemReceiverId, class _ErrorsVariant>
    struct __item_operation {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using __item_receiver_t = stdexec::__t<__item_receiver<_ItemReceiverId, _ErrorsVariant>>;

      struct __t : __item_operation_base<_ItemReceiver, _ErrorsVariant> {
        connect_result_t<_Item, __item_receiver_t> __op_;

        __t(__operation_base<_ErrorsVariant>* __base_op, _Item&& __item, _ItemReceiver __rcvr)
          : __item_operation_base<
            _ItemReceiver,
            _ErrorsVariant>{__base_op, static_cast<_ItemReceiver&&>(__rcvr)}
          , __op_(stdexec::connect(static_cast<_Item&&>(__item), __item_receiver_t{this})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _ItemId, class _ErrorsVariant>
    struct __item_sender {
      using _Item = stdexec::__t<_ItemId>;

      template <class _Rcvr>
      using __item_receiver_t =
        stdexec::__t<__item_receiver<stdexec::__id<__decay_t<_Rcvr>>, _ErrorsVariant>>;

      template <class _Self, class _Rcvr>
      using __item_operation_t = stdexec::__t<__item_operation<
        __copy_cvref_t<_Self, _Item>,
        stdexec::__id<__decay_t<_Rcvr>>,
        _ErrorsVariant>>;

      struct __t {
        using __id = __item_sender;
        STDEXEC_NO_UNIQUE_ADDRESS _Item __item_;
        __operation_base<_ErrorsVariant>* __base_op_;

        template <__decays_to<__t> _Self, receiver _ItemReceiver>
          requires sender_to<_Item, __item_receiver_t<_ItemReceiver>>
        friend __item_operation_t<_Self, _ItemReceiver>
          tag_invoke(connect_t, _Self&& __self, _ItemReceiver&& __rcvr) {
          return __item_operation_t<_Self, _ItemReceiver>{
            __self.__base_op_,
            static_cast<_Self&&>(__self).__item_,
            static_cast<_ItemReceiver&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&& __self, const _Env& __env)
          -> __try_make_completion_signatures<
            __copy_cvref_t<_Self, _Item>,
            _Env,
            completion_signatures<set_value_t()>,
            __mconst<completion_signatures<set_value_t()>>,
            __mconst<completion_signatures<set_stopped_t()>>>;
      };
    };

    template <class _Sndr, class _ErrorsVariant>
    using __item_sender_t = __t<__item_sender<__id<__decay_t<_Sndr>>, _ErrorsVariant>>;

    template <class _Receiver, class _ErrorsVariant>
    struct __sequence_operation_base : __operation_base<_ErrorsVariant> {
      STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
    };

    template <class _ReceiverId, class _ErrorsVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        STDEXEC_NO_UNIQUE_ADDRESS __sequence_operation_base<_Receiver, _ErrorsVariant>* __op_;

        template <sender _Item>
        friend __item_sender_t<_Item, _ErrorsVariant>
          tag_invoke(set_next_t, __t& __self, _Item&& __item) noexcept {
          return __item_sender_t<_Item, _ErrorsVariant>{static_cast<_Item&&>(__item), __self.__op_};
        }

        template <__same_as<set_error_t> _SetError, class _Error>
        friend void tag_invoke(_SetError, __t&& __self, _Error&& __error) noexcept {
          _SetError{}(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(__error));
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          if (!__self.__op_->__error_.__visit_if_error(
                static_cast<_Receiver&&>(__self.__op_->__rcvr_))) {
            _SetValue{}(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
          }
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          _SetStopped{}(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
        }

        friend env_of_t<_Receiver> tag_invoke(get_env_t, const __t& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Rcvr, class _ErrorsVariant>
    using __receiver_t = __t<__receiver<__id<__decay_t<_Rcvr>>, _ErrorsVariant>>;

    template <class _Sender, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Env = env_of_t<_Receiver>;
      using _ErrorsVariant = __error_sigs_of_t<
        __sequence_signatures_of_t<_Sender, _Env>,
        __mbind_front_q<std::variant, not_an_error>>;

      struct __t : __sequence_operation_base<_Receiver, _ErrorsVariant> {
        sequence_connect_result_t<_Sender, __receiver_t<_Receiver, _ErrorsVariant>> __op_;

        explicit __t(_Sender&& __sndr, _Receiver __rcvr)
          : __sequence_operation_base<
            _Receiver,
            _ErrorsVariant>{{}, static_cast<_Receiver&&>(__rcvr)}
          , __op_(exec::sequence_connect(
              static_cast<_Sender&&>(__sndr),
              __receiver_t<_Receiver, _ErrorsVariant>{this})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class... _Args>
    using __drop_value_args = completion_signatures<set_value_t()>;

    template <class... _Errors>
    using __add_stopped = completion_signatures<set_stopped_t(), set_error_t(_Errors)...>;

    template <class _Sender, class _Env>
    using __completion_sigs = __msuccess_or_t<__try_make_sequence_signatures<
      _Sender,
      _Env,
      completion_signatures<set_value_t()>,
      __q<__drop_value_args>,
      __q<__add_stopped>>>;

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<__decay_t<_SenderId>>;

      template <class _Rcvr>
      using _ErrorsVariant = __error_sigs_of_t<
        __sequence_signatures_of_t<_Sender, env_of_t<_Rcvr>>,
        __mbind_front_q<std::variant, not_an_error>>;

      struct __t {
        using __id = __sender;
        using is_sender = void;
        STDEXEC_NO_UNIQUE_ADDRESS _Sender __sndr_;

        template <class _Self, class _Rcvr>
        using __operation_t =
          stdexec::__t<__operation<__copy_cvref_t<_Self, _Sender>, stdexec::__id<__decay_t<_Rcvr>>>>;

        template <__decays_to<__t> _Self, receiver _Receiver>
          requires receiver_of<
                     _Receiver,
                     __completion_sigs<__copy_cvref_t<_Self, _Sender>, env_of_t<_Receiver>>>
                && sequence_sender_to<
                     __copy_cvref_t<_Self, _Sender>,
                     __receiver_t<_Receiver, _ErrorsVariant<_Receiver>>>
        friend __operation_t<_Self, _Receiver>
          tag_invoke(connect_t, _Self&& __self, _Receiver&& __rcvr) {
          return __operation_t<_Self, _Receiver>{
            static_cast<_Self&&>(__self).__sndr_, static_cast<_Receiver&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<__copy_cvref_t<_Self, _Sender>, _Env>;
      };
    };

    struct ignore_all_t {
      template <class _Sender>
      constexpr auto operator()(_Sender&& __sndr) const {
        return __t<__sender<__id<__decay_t<_Sender>>>>{static_cast<_Sender&&>(__sndr)};
      }

      constexpr auto operator()() const noexcept -> __binder_back<ignore_all_t> {
        return {};
      }
    };
  } // namespace __ignore_all

  using __ignore_all::ignore_all_t;

  inline constexpr ignore_all_t ignore_all;
}