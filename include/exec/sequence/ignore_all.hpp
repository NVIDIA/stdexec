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

    enum class __result_state {
      __none,
      __emplace,
      __emplace_done,
    };

    struct not_an_error { };

    struct have_been_stopped { };

    template <class _Error, class _ResultVariant>
    concept __is_valid_error =                      //
      __not_decays_to<_Error, not_an_error> &&      //
      __not_decays_to<_Error, have_been_stopped> && //
      requires(_ResultVariant& __variant_, _Error&& __error) {
        __variant_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__error));
      };

    enum class __result_type {
      __value,
      __error,
      __stopped
    };

    template <class _Rcvr>
    struct __result_visitor {
      _Rcvr __rcvr;

      template <class _Error>
      __result_type operator()(_Error&& __error) noexcept {
        if constexpr (__decays_to<_Error, have_been_stopped>) {
          return __result_type::__stopped;
        } else if constexpr (__not_decays_to<_Error, not_an_error>) {
          stdexec::set_error(static_cast<_Rcvr&&>(__rcvr), static_cast<_Error&&>(__error));
          return __result_type::__error;
        } else {
          STDEXEC_ASSERT(false);
          return __result_type::__value;
        }
      }
    };

    template <class _ResultVariant>
    struct __result_storage {
      STDEXEC_NO_UNIQUE_ADDRESS _ResultVariant __variant_{};
      std::atomic<__result_state> __state_{__result_state::__none};

      void __set_stopped() {
        __result_state __expected = __result_state::__none;
        if (__state_.compare_exchange_strong(
              __expected, __result_state::__emplace, std::memory_order_relaxed)) {
          __variant_.template emplace<1>();
          __state_.store(__result_state::__emplace_done, std::memory_order_release);
        }
      }

      template <__is_valid_error<_ResultVariant> _Error>
      void __emplace(_Error&& __error) {
        __result_state __expected = __result_state::__none;
        if (__state_.compare_exchange_strong(
              __expected, __result_state::__emplace, std::memory_order_relaxed)) {
          __variant_.template emplace<__decay_t<_Error>>(static_cast<_Error&&>(__error));
          __state_.store(__result_state::__emplace_done, std::memory_order_release);
        }
      }

      template <class _Rcvr>
      __result_type __visit(_Rcvr&& __rcvr) noexcept {
        __result_state __state = __state_.load(std::memory_order_acquire);
        switch (__state) {
        case __result_state::__none:
          return __result_type::__value;
        case __result_state::__emplace_done:
          return std::visit(
            __result_visitor<_Rcvr>{static_cast<_Rcvr&&>(__rcvr)},
            static_cast<_ResultVariant&&>(__variant_));
        case __result_state::__emplace:
          [[fallthrough]];
        default:
          STDEXEC_ASSERT(false);
        }
        return __result_type::__value;
      }
    };

    template <class _ResultVariant>
    struct __operation_base {
      STDEXEC_NO_UNIQUE_ADDRESS __result_storage<_ResultVariant> __error_{};
    };

    template <class _ItemReceiver, class _ResultVariant>
    struct __item_operation_base {
      __operation_base<_ResultVariant>* __seq_op_;
      STDEXEC_NO_UNIQUE_ADDRESS _ItemReceiver __rcvr_;
    };

    template <class _ItemReceiverId, class _ResultVariant>
    struct __item_receiver {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t {
        using is_receiver = void;
        using __id = __item_receiver;
        __item_operation_base<_ItemReceiver, _ResultVariant>* __op_;

        // We eat all arguments
        template <same_as<set_value_t> _SetValue, same_as<__t> _Self, class... _Args>
          requires __callable<_SetValue, _ItemReceiver&&>
        friend void tag_invoke(_SetValue, _Self&& __self, _Args&&...) noexcept {
          _SetValue{}(static_cast<_ItemReceiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
          requires __is_valid_error<_Error, _ResultVariant>
                && __callable<set_stopped_t, _ItemReceiver&&>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__seq_op_->__error_.__emplace(static_cast<_Error&&>(__error));
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
          requires __callable<_SetStopped, _ItemReceiver&&>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          __self.__op_->__seq_op_->__error_.__set_stopped();
          _SetStopped{}(static_cast<_ItemReceiver&&>(__self.__op_->__rcvr_));
        }

        template <same_as<get_env_t> _GetEnv, same_as<__t> _Self>
        friend env_of_t<_ItemReceiver> tag_invoke(_GetEnv, const __t& __self) noexcept {
          return stdexec::get_env(__self.__op_->__rcvr_);
        }
      };
    };

    template <class _Item, class _ItemReceiverId, class _ResultVariant>
    struct __item_operation {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using __item_receiver_t = stdexec::__t<__item_receiver<_ItemReceiverId, _ResultVariant>>;

      struct __t : __item_operation_base<_ItemReceiver, _ResultVariant> {
        connect_result_t<_Item, __item_receiver_t> __op_;

        __t(__operation_base<_ResultVariant>* __base_op, _Item&& __item, _ItemReceiver __rcvr)
          : __item_operation_base<
            _ItemReceiver,
            _ResultVariant>{__base_op, static_cast<_ItemReceiver&&>(__rcvr)}
          , __op_(stdexec::connect(static_cast<_Item&&>(__item), __item_receiver_t{this})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _ItemId, class _ResultVariant>
    struct __item_sender {
      using _Item = stdexec::__t<_ItemId>;

      template <class _Rcvr>
      using __item_receiver_t =
        stdexec::__t<__item_receiver<stdexec::__id<__decay_t<_Rcvr>>, _ResultVariant>>;

      template <class _Self, class _Rcvr>
      using __item_operation_t = stdexec::__t<__item_operation<
        __copy_cvref_t<_Self, _Item>,
        stdexec::__id<__decay_t<_Rcvr>>,
        _ResultVariant>>;

      struct __t {
        using __id = __item_sender;
        STDEXEC_NO_UNIQUE_ADDRESS _Item __item_;
        __operation_base<_ResultVariant>* __base_op_;

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

    template <class _Sndr, class _ResultVariant>
    using __item_sender_t = __t<__item_sender<__id<__decay_t<_Sndr>>, _ResultVariant>>;

    template <class _Receiver, class _ResultVariant>
    struct __sequence_operation_base : __operation_base<_ResultVariant> {
      STDEXEC_NO_UNIQUE_ADDRESS _Receiver __rcvr_;
    };

    template <class _ReceiverId, class _ResultVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;
        STDEXEC_NO_UNIQUE_ADDRESS __sequence_operation_base<_Receiver, _ResultVariant>* __op_;

        template <sender _Item>
        friend __item_sender_t<_Item, _ResultVariant>
          tag_invoke(set_next_t, __t& __self, _Item&& __item) noexcept {
          return __item_sender_t<_Item, _ResultVariant>{static_cast<_Item&&>(__item), __self.__op_};
        }

        template <__same_as<set_error_t> _SetError, class _Error>
        friend void tag_invoke(_SetError, __t&& __self, _Error&& __error) noexcept {
          _SetError{}(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(__error));
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          auto __type = __self.__op_->__error_.__visit(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_));
          if (__type == __result_type::__value) {
            _SetValue{}(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
          } else if (__type == __result_type::__stopped) {
            stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
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

    template <class _Rcvr, class _ResultVariant>
    using __receiver_t = __t<__receiver<__id<__decay_t<_Rcvr>>, _ResultVariant>>;

    template <class _Sender, class _ReceiverId>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Env = env_of_t<_Receiver>;
      using _ResultVariant = __error_sigs_of_t<
        __sequence_signatures_of_t<_Sender, _Env>,
        __mbind_front_q<std::variant, not_an_error, have_been_stopped>>;

      struct __t : __sequence_operation_base<_Receiver, _ResultVariant> {
        sequence_connect_result_t<_Sender, __receiver_t<_Receiver, _ResultVariant>> __op_;

        explicit __t(_Sender&& __sndr, _Receiver __rcvr)
          : __sequence_operation_base<
            _Receiver,
            _ResultVariant>{{}, static_cast<_Receiver&&>(__rcvr)}
          , __op_(exec::sequence_connect(
              static_cast<_Sender&&>(__sndr),
              __receiver_t<_Receiver, _ResultVariant>{this})) {
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
      using _ResultVariant = __error_sigs_of_t<
        __sequence_signatures_of_t<_Sender, env_of_t<_Rcvr>>,
        __mbind_front_q<std::variant, not_an_error, have_been_stopped>>;

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
                     __receiver_t<_Receiver, _ResultVariant<_Receiver>>>
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