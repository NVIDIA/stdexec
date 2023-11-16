/*
 * Copyright (c) 2023 Maikel Nadolski
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

#include "../env.hpp"
#include "../sequence_senders.hpp"

#include <optional>

namespace exec {
  namespace __merge_each {
    using namespace stdexec;

    struct __default_stop_callback {
      stdexec::in_place_stop_source& __stop_source_;

      void operator()() const noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _Receiver>
    struct __error_visitor {
      _Receiver& __receiver_;

      template <class _Error>
      void operator()(_Error&& __error) const noexcept {
        if constexpr (__not_decays_to<_Error, std::monostate>) {
          stdexec::set_error(static_cast<_Receiver&&>(__receiver_), static_cast<_Error&&>(__error));
        }
      }
    };

    template <class _Receiver, class _ErrorsVariant>
    struct __operation_base {
      using __stop_token_t = stop_token_of_t<env_of_t<_Receiver>>;
      using __default_stop_callback_t =
        typename __stop_token_t::template callback_type<__default_stop_callback>;

      void __notify_completion() noexcept {
        if (__n_pending_ops_.fetch_sub(1, std::memory_order_relaxed) == 1) {
          __on_receiver_stopped_.reset();
          int __error_emplaced = __error_emplaced_.load(std::memory_order_acquire);
          if (__error_emplaced == 2) {
            std::visit(
              __error_visitor<_Receiver>{__receiver_}, static_cast<_ErrorsVariant&&>(__errors_));
          } else {
            exec::__set_value_unless_stopped(static_cast<_Receiver&&>(__receiver_));
          }
        }
      }

      template <class Error>
      void __emplace_error(Error&& error) noexcept {
        int __expected = 0;
        if (__error_emplaced_.compare_exchange_strong(__expected, 1, std::memory_order_relaxed)) {
          __errors_.template emplace<__decay_t<Error>>(static_cast<Error&&>(error));
          __error_emplaced_.store(2, std::memory_order_release);
        }
      }

      std::atomic<int> __n_pending_ops_;
      _Receiver __receiver_;
      _ErrorsVariant __errors_{};
      std::atomic<int> __error_emplaced_{0};
      in_place_stop_source __stop_source_{};
      std::optional<__default_stop_callback_t> __on_receiver_stopped_{};
    };

    template <class _Receiver, class _ErrorsVariant>
    struct __next_receiver {
      class __t;
    };

    template <class _Variant, class _Tp, class... _Args>
    concept __emplaceable = requires(_Variant&& __variant, _Args&&... __args) {
      { __variant.template emplace<_Tp>(static_cast<_Args&&>(__args)...) };
    };

    template <class _ReceiverId, class _ErrorsVariant>
    class __next_receiver<_ReceiverId, _ErrorsVariant>::__t {
      using _Receiver = stdexec::__t<_ReceiverId>;
     public:
      using __id = __next_receiver;
      using is_receiver = void;

      explicit __t(__operation_base<_Receiver, _ErrorsVariant>* __parent) noexcept
        : __op_{__parent} {
      }

     private:
      __operation_base<_Receiver, _ErrorsVariant>* __op_;

      template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Item>
        requires __callable<_SetNext, _Receiver&, _Item>
      friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept(
        __nothrow_callable<_SetNext, _Receiver&, _Item>) -> next_sender_of_t<_Receiver&, _Item> {
        return exec::set_next(__self.__op_->__receiver_, static_cast<_Item&&>(__item));
      }

      template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
      friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
        __self.__op_->__notify_completion();
      }

      template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
      friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
        __self.__op_->__notify_completion();
      }

      template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        requires __callable<set_error_t, _Receiver&&, _Error> //
              && __emplaceable<_ErrorsVariant, __decay_t<_Error>, _Error>
      friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
        __self.__op_->__emplace_error(static_cast<_Error&&>(__error));
        __self.__op_->__stop_source_.request_stop();
        __self.__op_->__notify_completion();
      }

      template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
      friend auto tag_invoke(_GetEnv, _Self&& __self) noexcept
        -> make_env_t<env_of_t<_Receiver>, with_t<get_stop_token_t, in_place_stop_token>> {
        return exec::make_env(
          stdexec::get_env(__self.__op_->__receiver_),
          exec::with(get_stop_token, __self.__op_->__stop_source_.get_token()));
      }
    };

    template <class _Receiver, class... _Senders>
    struct __traits {
      using __env = env_of_t<_Receiver>;

      using __errors_variant = //
        __minvoke<__mconcat<__nullable_variant_t>, error_types_of_t<_Senders, __env, __types>...>;

      using __next_receiver_t = __t<__next_receiver<__id<_Receiver>, __errors_variant>>;
    };

    template <class _ReceiverId, class... _Senders>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Base =
        __operation_base<_Receiver, typename __traits<_Receiver, _Senders...>::__errors_variant>;
      class __t;
    };

    template <class _ReceiverId, class... _Senders>
    class __operation<_ReceiverId, _Senders...>::__t
      : __operation<_ReceiverId, _Senders...>::_Base {
      using __next_receiver_t = typename __traits<_Receiver, _Senders...>::__next_receiver_t;
      std::tuple<subscribe_result_t<_Senders, __next_receiver_t>...> __ops_;

     public:
      __t(_Receiver __rcvr, _Senders&&... __sndrs)
        : _Base{sizeof...(_Senders), static_cast<_Receiver&&>(__rcvr)}
        , __ops_{__conv{[&] {
          return exec::subscribe(static_cast<_Senders&&>(__sndrs), __next_receiver_t{this});
        }}...} {
      }

      friend void tag_invoke(stdexec::start_t, __t& __self) noexcept {
        __apply([](auto&... __op) { (stdexec::start(__op), ...); }, __self.__ops_);
      }
    };

    template <class _Sequence, class _Env>
    using __item_completion_signatures_of_t =
      __concat_item_signatures_t<item_types_of_t<_Sequence, _Env>, _Env>;

    template <class _Sequence, class _Env>
    using __single_item_value_t = __gather_signal<
      set_value_t,
      __item_completion_signatures_of_t<_Sequence, _Env>,
      __msingle_or<void>,
      __q<__msingle>>;

    template <class _Env, class... _Senders>
    concept __sequence_factory =                            //
      sizeof...(_Senders) == 1 &&                           //
      __single_typed_sender<__mfront<_Senders...>, _Env> && //
      sequence_sender_in<__single_item_value_t<__mfront<_Senders...>, _Env>, _Env>;

    template <class _Receiver, class _ErrorsVariant>
    struct __dynamic_item_stop {
      __operation_base<_Receiver, _ErrorsVariant>* __parent_;

      void operator()() const noexcept {
        __parent_->__stop_source_.request_stop();
        __parent_->__notify_completion();
      }
    };

    template <class _ItemReceiver, class _Receiver, class _ErrorsVariant>
    struct __dynamic_item_operation_base {
      _ItemReceiver __item_receiver_;
      __operation_base<_Receiver, _ErrorsVariant>* __parent_;
      using __stop_token_t = stop_token_of_t<env_of_t<_ItemReceiver>>;
      using __stop_callback_t = typename __stop_token_t::template callback_type<
        __dynamic_item_stop<_Receiver, _ErrorsVariant>>;
      std::optional<__stop_callback_t> __on_item_receiver_stopped_{};
    };

    template <class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __dynamic_next_receiver {
      class __t;
    };

    template <class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    class __dynamic_next_receiver<_ItemReceiverId, _ReceiverId, _ErrorsVariant>::__t {
     private:
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      __dynamic_item_operation_base<_ItemReceiver, _Receiver, _ErrorsVariant>* __op_;

      template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
      friend auto tag_invoke(_GetEnv, _Self&& __self) noexcept
        -> make_env_t<env_of_t<_ItemReceiver>, with_t<get_stop_token_t, in_place_stop_token>> {
        return exec::make_env(
          stdexec::get_env(__self.__op_->__item_receiver_),
          exec::with(get_stop_token, __self.__op_->__parent_->__stop_source_.get_token()));
      }

      template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Sender>
        requires __callable<_SetNext, _Receiver&, _Sender>
      friend auto tag_invoke(_SetNext, _Self& __self, _Sender&& sender) noexcept(
        __nothrow_callable<_SetNext, _Receiver&, _Sender>) { // -> next_sender_of_t<_Receiver, _Sender> {
        return exec::set_next(__self.__op_->__parent_->__receiver_, static_cast<_Sender&&>(sender));
      }

      template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        requires __callable<_SetValue, _ItemReceiver&&>
      friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
        __self.__op_->__on_item_receiver_stopped_.reset();
        __operation_base<_Receiver, _ErrorsVariant>* __parent = __self.__op_->__parent_;
        stdexec::set_value(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
        __parent->__notify_completion();
      }

      template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        requires __callable<_SetStopped, _ItemReceiver&&>
      friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
        __self.__op_->__on_item_receiver_stopped_.reset();
        __operation_base<_Receiver, _ErrorsVariant>* __parent = __self.__op_->__parent_;
        stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
        __parent->__notify_completion();
      }

      template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        requires __callable<set_stopped_t, _ItemReceiver&&>
      friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
        __self.__op_->__on_item_receiver_stopped_.reset();
        __operation_base<_Receiver, _ErrorsVariant>* __parent = __self.__op_->__parent_;
        __parent->__emplace_error(static_cast<_Error&&>(__error));
        __parent->__stop_source_.request_stop();
        stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
        __parent->__notify_completion();
      }
     public:
      using __id = __dynamic_next_receiver;
      using is_receiver = void;

      explicit __t(
        __dynamic_item_operation_base<_ItemReceiver, _Receiver, _ErrorsVariant>* __op) noexcept
        : __op_{__op} {
      }
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __subsequence_operation;

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __subsequence_operation
      : __dynamic_item_operation_base<
          stdexec::__t<_ItemReceiverId>,
          stdexec::__t<_ReceiverId>,
          _ErrorsVariant> {

      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _Subsequence = __single_sender_value_t<_Item>;

      std::optional<subscribe_result_t<
        _Subsequence,
        stdexec::__t<__dynamic_next_receiver<_ItemReceiverId, _ReceiverId, _ErrorsVariant>>>>
        __op_;

      __subsequence_operation(
        _ItemReceiver __item_receiver,
        __operation_base<_Receiver, _ErrorsVariant>* __parent)
        : __dynamic_item_operation_base<_ItemReceiver, _Receiver, _ErrorsVariant>(
          static_cast<_ItemReceiver&&>(__item_receiver),
          __parent) {
      }
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __receive_subsequence {
      class __t;
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    class __receive_subsequence<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>::__t {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __subsequence_operation_t =
        __subsequence_operation<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>;

      using __dynamic_next_receiver_t =
        stdexec::__t<__dynamic_next_receiver<_ItemReceiverId, _ReceiverId, _ErrorsVariant>>;

      __subsequence_operation_t* __op_;

      template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
      friend auto tag_invoke(_GetEnv, _Self&& __self) noexcept
        -> make_env_t<env_of_t<_ItemReceiver>, with_t<get_stop_token_t, in_place_stop_token>> {
        return exec::make_env(
          stdexec::get_env(__self.__op_->__item_receiver_),
          exec::with(get_stop_token, __self.__op_->__parent_->__stop_source_.get_token()));
      }

      template <same_as<set_value_t> _SetValue, same_as<__t> _Self, class _Subsequence>
        requires sequence_sender_to<_Subsequence, __dynamic_next_receiver_t>
      friend void tag_invoke(_SetValue, _Self&& __self, _Subsequence&& subsequence) noexcept {
        try {
          auto& __next_op = __self.__op_->__op_.emplace(stdexec::__conv{[&] {
            return exec::subscribe(
              static_cast<_Subsequence&&>(subsequence), __dynamic_next_receiver_t{__self.__op_});
          }});
          stdexec::start(__next_op);
        } catch (...) {
          __operation_base<_Receiver, _ErrorsVariant>* __parent = __self.__op_->__parent_;
          __self.__op_->__on_item_receiver_stopped_.reset();
          __parent->__emplace_error(std::current_exception());
          __parent->__stop_source_.request_stop();
          stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
          __parent->__notify_completion();
        }
      }

      template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
      friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
        __self.__op_->__on_item_receiver_stopped_.reset();
        __operation_base<_Receiver, _ErrorsVariant>* __parent = __self.__op_->__parent_;
        stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
        __parent->__notify_completion();
      }

      template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
      friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
        __self.__op_->__on_item_receiver_stopped_.reset();
        __operation_base<_Receiver, _ErrorsVariant>* __parent = __self.__op_->__parent_;
        __parent->__emplace_error(static_cast<_Error&&>(__error));
        __parent->__stop_source_.request_stop();
        stdexec::set_stopped(static_cast<_ItemReceiver&&>(__self.__op_->__item_receiver_));
        __parent->__notify_completion();
      }
     public:
      using __id = __receive_subsequence;
      using is_receiver = void;

      explicit __t(__subsequence_operation_t* __op) noexcept
        : __op_{__op} {
      }
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    struct __dynamic_item_operation {
      class __t;
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ErrorsVariant>
    class __dynamic_item_operation<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>::__t
      : __subsequence_operation<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant> {

      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      using _Base = __subsequence_operation<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>;

      using __receive_subsequence_t =
        stdexec::__t<__receive_subsequence<_Item, _ItemReceiverId, _ReceiverId, _ErrorsVariant>>;

      connect_result_t<_Item, __receive_subsequence_t> __receive_op_;

     public:
      explicit __t(
        _Item&& __item,
        _ItemReceiver __item_receiver,
        __operation_base<_Receiver, _ErrorsVariant>* __parent)
        : _Base(static_cast<_ItemReceiver&&>(__item_receiver), __parent)
        , __receive_op_{
            stdexec::connect(static_cast<_Item&&>(__item), __receive_subsequence_t{this})} {
      }

      template <same_as<__t> _Self>
      friend void tag_invoke(stdexec::start_t, _Self& __self) noexcept {
        __self.__on_item_receiver_stopped_.emplace(
          stdexec::get_stop_token(stdexec::get_env(__self.__item_receiver_)),
          __dynamic_item_stop<_Receiver, _ErrorsVariant>{__self.__parent_});
        stdexec::start(__self.__receive_op_);
      }
    };

    template <class _SubsequenceId, class _ReceiverId, class _ErrorsVariant>
    struct __dynamic_item_sender {
      class __t;
    };

    template <class _ItemId, class _ReceiverId, class _ErrorsVariant>
    class __dynamic_item_sender<_ItemId, _ReceiverId, _ErrorsVariant>::__t {
      using __id = __dynamic_item_sender;
      using _Item = stdexec::__t<_ItemId>;
      using _Receiver = stdexec::__t<_ReceiverId>;

      template <class _ItemReceiver>
      using __operation_t = stdexec::__t<
        __dynamic_item_operation<_Item, stdexec::__id<_ItemReceiver>, _ReceiverId, _ErrorsVariant>>;

      template <class _ItemReceiver>
      using __receive_subsequence_t = stdexec::__t<
        __receive_subsequence<_Item, stdexec::__id<_ItemReceiver>, _ReceiverId, _ErrorsVariant>>;

      _Item __item_;
      __operation_base<_Receiver, _ErrorsVariant>* __parent_;

      template <__decays_to<__t> _Self, class _ItemReceiver>
        requires sender_to<__copy_cvref_t<_Self, _Item>, __receive_subsequence_t<_ItemReceiver>>
      friend auto tag_invoke(stdexec::connect_t, _Self&& self, _ItemReceiver __item_receiver)
        -> __operation_t<_ItemReceiver> {
        return __operation_t<_ItemReceiver>{
          static_cast<_Self&&>(self).__item_,
          static_cast<_ItemReceiver&&>(__item_receiver),
          self.__parent_};
      }

     public:
      using is_sender = void;
      using completion_signatures = stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

      template <class _CvItem>
      explicit __t(_CvItem&& __item, __operation_base<_Receiver, _ErrorsVariant>* __parent)
        : __item_(static_cast<_CvItem&&>(__item))
        , __parent_(__parent) {
      }
    };

    template <class _ReceiverId, class _ErrorsVariant>
    struct __dynamic_receiver {
      class __t;
    };

    template <class _ReceiverId, class _ErrorsVariant>
    class __dynamic_receiver<_ReceiverId, _ErrorsVariant>::__t {
      using _Receiver = stdexec::__t<_ReceiverId>;

      template <class _Item>
      using __next_sender_t = stdexec::__t<
        __dynamic_item_sender<stdexec::__id<__decay_t<_Item>>, _ReceiverId, _ErrorsVariant>>;

      __operation_base<_Receiver, _ErrorsVariant>* __parent_;

      template <same_as<get_env_t> _GetEnv, __decays_to<__t> _Self>
      friend auto tag_invoke(_GetEnv, _Self&& __self) noexcept
        -> make_env_t<env_of_t<_Receiver>, with_t<get_stop_token_t, in_place_stop_token>> {
        return exec::make_env(
          stdexec::get_env(__self.__parent_->__receiver_),
          exec::with(get_stop_token, __self.__parent_->__stop_source_.get_token()));
      }

      template <same_as<set_next_t> _SetNext, same_as<__t> _Self, class _Item>
      friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept(
        __nothrow_decay_copyable<_Item>) //
        -> __next_sender_t<_Item> {
        return __next_sender_t<_Item>{static_cast<_Item&&>(__item), __self.__parent_};
      }

      template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
      friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
        __self.__parent_->__on_receiver_stopped_.reset();
        int __error_emplaced = __self.__parent_->__error_emplaced_.load(std::memory_order_acquire);
        if (__error_emplaced == 2) {
          std::visit(
            __error_visitor<_Receiver>{__self.__parent_->__receiver_},
            static_cast<_ErrorsVariant&&>(__self.__parent_->__errors_));
        } else {
          stdexec::set_value(static_cast<_Receiver&&>(__self.__parent_->__receiver_));
        }
      }

      template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
      friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
        __self.__parent_->__on_receiver_stopped_.reset();
        int __error_emplaced = __self.__parent_->__error_emplaced_.load(std::memory_order_acquire);
        if (__error_emplaced == 2) {
          std::visit(
            __error_visitor<_Receiver>{__self.__parent_->__receiver_},
            static_cast<_ErrorsVariant&&>(__self.__parent_->__errors_));
        } else {
          exec::__set_value_unless_stopped(static_cast<_Receiver&&>(__self.__parent_->__receiver_));
        }
      }

      template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
      friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
        __self.__parent_->__on_receiver_stopped_.reset();
        stdexec::set_error(
          static_cast<_Receiver&&>(__self.__parent_->__receiver_), static_cast<_Error&&>(__error));
      }

     public:
      using __id = __dynamic_receiver;
      using is_receiver = void;

      explicit __t(__operation_base<_Receiver, _ErrorsVariant>* __parent) noexcept
        : __parent_{__parent} {
      }
    };

    template <class _Sender, class _ReceiverId, class _ErrorsVariant>
    struct __dynamic_operation {
      class __t;
    };

    template <class _Sender, class _ReceiverId, class _ErrorsVariant>
    class __dynamic_operation<_Sender, _ReceiverId, _ErrorsVariant>::__t
      : __operation_base<stdexec::__t<_ReceiverId>, _ErrorsVariant> {
      using _Receiver = stdexec::__t<_ReceiverId>;

      template <same_as<__t> _Self>
      friend void tag_invoke(stdexec::start_t, _Self& __self) noexcept {
        __self.__on_receiver_stopped_.emplace(
          stdexec::get_stop_token(stdexec::get_env(__self.__receiver_)),
          __default_stop_callback{__self.__stop_source_});
        stdexec::start(__self.__op_);
      }

      subscribe_result_t<_Sender, stdexec::__t<__dynamic_receiver<_ReceiverId, _ErrorsVariant>>>
        __op_;

     public:
      __t(_Sender&& sndr, _Receiver rcvr)
        : __operation_base<_Receiver, _ErrorsVariant>{1, static_cast<_Receiver&&>(rcvr)}
        , __op_{exec::subscribe(
            static_cast<_Sender&&>(sndr),
            stdexec::__t<__dynamic_receiver<_ReceiverId, _ErrorsVariant>>{this})} {
      }
    };

    template <class... _Senders>
    struct __sequence {
      class __t;
    };

    template <class... _Senders>
    class __sequence<_Senders...>::__t {
      template <class _Self, class _Env>
      using __value_type_t =
        __single_item_value_t<__copy_cvref_t<_Self, __mfront<_Senders...>>, _Env>;

      template <class _Self, class _Receiver>
      using __errors_variant_t =
        typename __traits<_Receiver, __copy_cvref_t<_Self, _Senders>...>::__errors_variant;

      template <class _Self, class _Receiver>
      using __dynamic_operation_t = stdexec::__t<__dynamic_operation<
        __copy_cvref_t<_Self, __mfront<_Senders...>>,
        stdexec::__id<_Receiver>,
        __errors_variant_t<_Self, _Receiver>>>;

      template <class _Self, class _Receiver>
      using __static_operation_t =
        stdexec::__t<__operation< stdexec::__id<_Receiver>, __copy_cvref_t<_Self, _Senders>...>>;

      std::tuple<_Senders...> __senders_;

      template <__decays_to<__t> _Self, class _Receiver>
        requires(!__sequence_factory<env_of_t<_Receiver>, __copy_cvref_t<_Self, _Senders>...>)
      friend auto tag_invoke(subscribe_t, _Self&& self, _Receiver receiver)
        -> __static_operation_t<_Self, _Receiver> {
        return __apply(
          [&]<class... _Sndrs>(_Sndrs&&... sndrs) {
            return __static_operation_t<_Self, _Receiver>{
              static_cast<_Receiver&&>(receiver), static_cast<_Sndrs&&>(sndrs)...};
          },
          static_cast<_Self&&>(self).__senders_);
      }

      template <__decays_to<__t> _Self, class _Receiver>
        requires __sequence_factory<env_of_t<_Receiver>, __copy_cvref_t<_Self, _Senders>...>
      friend auto tag_invoke(subscribe_t, _Self&& self, _Receiver receiver) //
        -> __dynamic_operation_t<_Self, _Receiver> {
        return __dynamic_operation_t<_Self, _Receiver>{
          std::get<0>(static_cast<_Self&&>(self).__senders_), static_cast<_Receiver&&>(receiver)};
      }

      template <__decays_to<__t> _Self, class _Env>
        requires(!__sequence_factory<_Env, __copy_cvref_t<_Self, _Senders>...>)
      friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
        -> __concat_completion_signatures_t<
          __to_sequence_completion_signatures<__copy_cvref_t<_Self, _Senders>, _Env>...>;

      template <__decays_to<__t> _Self, class _Env>
        requires __sequence_factory<_Env, __copy_cvref_t<_Self, _Senders>...>
      friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env&&)
        -> __to_sequence_completion_signatures<__value_type_t<_Self, _Env>, _Env>;

      template <class _Self, class _Env>
      static auto get_item_types() noexcept {
        if constexpr (!__sequence_factory<_Env, __copy_cvref_t<_Self, _Senders>...>) {
          using _Result = __minvoke<
            __mconcat<__q<item_types>>,
            item_types_of_t<__copy_cvref_t<_Self, _Senders>, _Env>...>;
          return (_Result(*)()) nullptr;
        } else {
          using _Result = item_types_of_t<__value_type_t<_Self, _Env>, _Env>;
          return (_Result(*)()) nullptr;
        }
      }

      template <__decays_to<__t> _Self, class _Env>
      friend auto tag_invoke(get_item_types_t, _Self&&, _Env&&)
        -> decltype(get_item_types<_Self, _Env>()()) {
        return {};
      }

     public:
      using __id = __sequence;
      using is_sender = exec::sequence_tag;

      __t(_Senders&&... sndrs)
        : __senders_{static_cast<_Senders&&>(sndrs)...} {
      }
    };

    struct merge_each_t {
      template <class... _Senders>
      auto operator()(_Senders&&... senders) const
        noexcept((__nothrow_decay_copyable<_Senders> && ...))
          -> __t<__sequence<__decay_t<_Senders>...>> {
        return {static_cast<_Senders&&>(senders)...};
      }

      auto operator()() const noexcept -> __binder_back<merge_each_t> {
        return {{}, {}, {}};
      }
    };
  } // namespace __merge_each

  using __merge_each::merge_each_t;

  inline constexpr merge_each_t merge_each{};
}