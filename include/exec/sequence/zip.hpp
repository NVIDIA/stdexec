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
#include "../../stdexec/__detail/__intrusive_queue.hpp"

namespace exec {
  namespace __zip {
    using namespace stdexec;

    template <class _BaseEnv>
    using __env_t = __make_env_t<_BaseEnv, __with<get_stop_token_t, in_place_stop_token>>;

    template <class _ReceiverId, class _Sender>
    using __next_sender_of_t =
      decltype(exec::set_next(__declval<__t<_ReceiverId>&>(), __declval<_Sender>()));

    template <class _ItemResult>
    struct __item_operation_result {
      using __result_type = _ItemResult;
      __item_operation_result* __next_item_op_{nullptr};
      void (*__complete_)(__item_operation_result*) noexcept = nullptr;
      std::optional<_ItemResult> __item_result_{};
      int __item_ready_{0};
    };

    template <class _ResultTuple>
    using __concat_result_types = __mapply<__mconcat<__q<std::tuple>>, _ResultTuple>;

    template <class... _Ts>
    using __to_op_queue =
      std::tuple<__intrusive_queue<&__item_operation_result<_Ts>::__next_item_op_>...>;

    struct __on_stop_requested {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _ReceiverId, class _ResultTuple, class _ErrorVariant>
    struct __operation_base : __immovable {
      using _Receiver = stdexec::__t<_ReceiverId>;

      using __on_stop = //
        typename stop_token_of_t<env_of_t<_Receiver&>>::template callback_type< __on_stop_requested>;

      __operation_base(__operation_base&&) = delete;

      template <__decays_to<_Receiver> _Rcvr>
      explicit __operation_base(_Rcvr&& __rcvr)
        : __receiver_{(_Rcvr&&) __rcvr} {
      }

      [[no_unique_address]] _Receiver __receiver_;
      __mapply<__transform<__mconst<std::mutex>, __q<std::tuple>>, _ResultTuple> __mutexes_{};
      std::atomic<int> __n_ready_next_items_{};
      __mapply<__q<__to_op_queue>, _ResultTuple> __item_queues_{};
      _ErrorVariant __error_{};
      std::mutex __stop_mutex_{};
      in_place_stop_source __stop_source_{};
      std::ptrdiff_t __n_pending_ops_{};
      std::optional<__on_stop> __stop_callback_{};

      template <std::size_t _Index>
      bool __increase_n_ready_items(
        __item_operation_result<std::tuple_element_t<_Index, _ResultTuple>>* __op) noexcept {
        std::scoped_lock __lock(std::get<_Index>(__mutexes_));
        __op->__item_ready_ = 1;
        STDEXEC_ASSERT(!std::get<_Index>(__item_queues_).empty());
        if (std::get<_Index>(__item_queues_).front() == __op) {
          return __n_ready_next_items_.fetch_add(1, std::memory_order_relaxed) + 1
              == std::tuple_size_v<_ResultTuple>;
        } else {
          return false;
        }
      }

      bool __increase_op_count() noexcept {
        std::scoped_lock __lock(__stop_mutex_);
        if (__stop_source_.stop_requested()) {
          return false;
        }
        __n_pending_ops_ += 1;
        return true;
      }

      template <std::size_t _Index>
      bool __push_back_item_op(
        __item_operation_result<std::tuple_element_t<_Index, _ResultTuple>>* __op) noexcept {
        if (__increase_op_count()) {
          std::scoped_lock __lock(std::get<_Index>(__mutexes_));
          std::get<_Index>(__item_queues_).push_back(__op);
          return true;
        }
        return false;
      }

      // Check whether we must complete the whole sequence operation
      void __notify_op_completion() noexcept {
        std::scoped_lock __lock(__stop_mutex_);
        __n_pending_ops_ -= 1;
        if (__n_pending_ops_ == 0 && __stop_source_.stop_requested()) {
          __stop_callback_.reset();
          auto token = get_stop_token(get_env(__receiver_));
          if (token.stop_requested()) {
            stdexec::set_stopped((_Receiver&&) __receiver_);
          } else if (__error_.index()) {
            std::visit(
              [&]<class _Error>(const _Error& __error) {
                stdexec::set_error((_Receiver&&) __receiver_, __error);
              },
              __error_);
          } else {
            stdexec::set_value((_Receiver&&) __receiver_);
          }
        }
      }

      template <class _Error>
        requires std::is_assignable_v<_ErrorVariant&, _Error&&>
      void __notify_error(_Error&& __error) {
        {
          std::scoped_lock __lock(__stop_mutex_);
          if (__error_.index() == 0) {
            __error_ = (_Error&&) __error;
          }
        }
        __stop_source_.request_stop();
        __notify_op_completion();
      }

      void __notify_stop() {
        __stop_source_.request_stop();
        __notify_op_completion();
      }
    };

    template < class _ReceiverId, class _ResultTuple, class _ErrorsVariant>
    struct __zipped_operation_base {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __operation_base_t = __operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>;

      template <class _Tp>
      using __to_item_result = __item_operation_result<_Tp>*;

      __operation_base_t* __parent_op_;
      std::optional<__mapply<__transform<__q<__to_item_result>, __q<std::tuple>>, _ResultTuple>>
        __items_{};

      void __complete_all_item_ops() noexcept {
        STDEXEC_ASSERT(__items_);
        std::apply(
          [&]<class... _Ts>(__item_operation_result<_Ts>*... __item_ops) {
            (__item_ops->__complete_(__item_ops), ...);
          },
          *__items_);
      }
    };

    template < class _ReceiverId, class _ResultTuple, class _ErrorsVariant>
    struct __zipped_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __zipped_receiver;
        __zipped_operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>* __op_{nullptr};

        template <class... _Args>
        friend void tag_invoke(set_value_t, __t&& __self, _Args&&...) noexcept {
          __self.__op_->__parent_op_->__notify_op_completion();
          __self.__op_->__complete_all_item_ops();
        }

        template <class _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          __self.__op_->__parent_op_->__notify_error((_Error&&) __error);
          __self.__op_->__complete_all_item_ops();
        }

        friend void tag_invoke(set_stopped_t, __t&& __self) noexcept {
          __self.__op_->__parent_op_->__notify_stop();
          __self.__op_->__complete_all_item_ops();
        }

        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const __t& __self) noexcept {
          return {
            stdexec::get_env(__self.__op_->__parent_op_->__receiver_),
            {__self.__op_->__parent_op_->__stop_source_.get_token()}};
        }
      };
    };

    template <class... _Ts>
    using __just_t = decltype(just(__declval<_Ts>()...));

    template <class _ResultTuple>
    using __just_sender_t = __mapply<__q<__just_t>, __concat_result_types<_ResultTuple>>;

    template <
      std::size_t _Index,
      class _ReceiverId,
      class _ResultTuple,
      class _ErrorsVariant,
      class _ItemReceiverId>
    struct __item_operation_base
      : __item_operation_result<std::tuple_element_t<_Index, _ResultTuple>>
      , __zipped_operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant> {
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using __base_t = __item_operation_result<std::tuple_element_t<_Index, _ResultTuple>>;
      using __zipped_base_t = __zipped_operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>;
      using __operation_base_t = __operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>;

      void __notify_result_completion() noexcept {
        // Check whether this is the the last item operation to complete such that we can start the zipped operation
        if (this->__parent_op_->template __increase_n_ready_items<_Index>(this)) {
          // 1. Collect all results and assemble one big tuple
          __concat_result_types<_ResultTuple> __result = std::apply(
            [&](auto&... __queues) {
              return std::tuple_cat(
                (__decay_t<decltype(*__queues.front()->__item_result_)>&&) *__queues.front()
                  ->__item_result_...);
            },
            this->__parent_op_->__item_queues_);

          // 2. pop front items from shared queues into a private storage of this op.
          std::apply(
            [&]<same_as<std::mutex>... _Mutex>(_Mutex&... __mutexes) {
              std::scoped_lock __lock(__mutexes...);
              this->__items_.emplace(std::apply(
                [](auto&... __queues) { return std::tuple{__queues.pop_front()...}; },
                this->__parent_op_->__item_queues_));
              const int __count = std::apply(
                [](auto&... __queues) {
                  return ((__queues.empty() ? 0 : __queues.front()->__item_ready_) + ...);
                },
                this->__parent_op_->__item_queues_);
              STDEXEC_ASSERT(__count < static_cast<int>(std::tuple_size_v<_ResultTuple>));
              this->__parent_op_->__n_ready_next_items_.store(__count, std::memory_order_relaxed);
            },
            this->__parent_op_->__mutexes_);

          // 3.a. Check whether we need to stop
          if (!this->__parent_op_->__increase_op_count()) {
            this->__complete_all_item_ops();
            return;
          }

          // 3.b. If continue, then start the zipped operation.
          auto& __op = __zipped_op_.emplace(__conv{[&] {
            return stdexec::connect(
              exec::set_next(
                this->__parent_op_->__receiver_,
                std::apply(
                  []<class... _Args>(_Args&&... __args) { return just((_Args&&) __args...); },
                  (__concat_result_types<_ResultTuple>&&) __result)),
              __t<__zipped_receiver<_ReceiverId, _ResultTuple, _ErrorsVariant>>{this});
          }});
          stdexec::start(__op);
        }
      }

      static void __complete(__base_t* __base) noexcept {
        __item_operation_base* __self = static_cast<__item_operation_base*>(__base);
        __operation_base_t* __parent_op = __self->__parent_op_;
        if (
          stdexec::get_stop_token(stdexec::get_env(__self->__item_rcvr_)).stop_requested()
          || __parent_op->__stop_source_.stop_requested()) {
          stdexec::set_stopped((_ItemReceiver&&) __self->__item_rcvr_);
        } else {
          stdexec::set_value((_ItemReceiver&&) __self->__item_rcvr_);
        }
        __parent_op->__notify_op_completion();
      }

      __item_operation_base(_ItemReceiver&& __item_rcvr, __operation_base_t* __parent_op)
        : __base_t{{}, &__complete, {}}
        , __zipped_base_t{__parent_op}
        , __item_rcvr_((_ItemReceiver&&) __item_rcvr) {
      }

      [[no_unique_address]] _ItemReceiver __item_rcvr_;
      std::optional<connect_result_t<
        __next_sender_of_t<_ReceiverId, __just_sender_t<_ResultTuple>>,
        __t<__zipped_receiver<_ReceiverId, _ResultTuple, _ErrorsVariant>>>>
        __zipped_op_{};
    };

    template <
      std::size_t _Index,
      class _ReceiverId,
      class _ResultTuple,
      class _ErrorsVariant,
      class _ItemReceiverId>
      requires receiver_of<
        stdexec::__t<_ItemReceiverId>,
        completion_signatures<set_value_t(), set_stopped_t()>>
    struct __item_receiver {
      using __item_operation_t =
        __item_operation_base<_Index, _ReceiverId, _ResultTuple, _ErrorsVariant, _ItemReceiverId>;

      using _Env = env_of_t<stdexec::__t<_ReceiverId>>;

      struct __t {
        using __id = __item_receiver;

        __item_operation_t* __op_;

        template <class... _Args>
        friend void tag_invoke(set_value_t, __t&& __self, _Args&&... __args) noexcept {
          try {
            __self.__op_->__item_result_.emplace((_Args&&) __args...);
            __self.__op_->__notify_result_completion();
          } catch (...) {
            __self.__op_->__parent_op_->__notify_error(std::current_exception());
          }
        }

        template <class _Error>
          requires std::is_assignable_v<_ErrorsVariant&, _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          __self.__op_->__parent_op_->__notify_error((_Error&&) __error);
        }

        friend void tag_invoke(set_stopped_t, __t&& __self) noexcept {
          __self.__op_->__parent_op_->__notify_stop();
        }

        friend __env_t<_Env> tag_invoke(get_env_t, const __t& __self) noexcept {
          using __with_token = __with<get_stop_token_t, in_place_stop_token>;
          __with_token token{__self.__op_->__parent_op_->__stop_source_.get_token()};
          return stdexec::__make_env(stdexec::get_env(__self.__op_->__item_rcvr_), token);
        }
      };
    };

    template <
      std::size_t _Index,
      class _ReceiverId,
      class _ResultTuple,
      class _ErrorsVariant,
      class _ItemId,
      class _ItemReceiverId>
    struct __item_operation {
      using __base_type =
        __item_operation_base<_Index, _ReceiverId, _ResultTuple, _ErrorsVariant, _ItemReceiverId>;

      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      using _Item = stdexec::__t<_ItemId>;

      using __item_receiver_t = stdexec::__t<
        __item_receiver<_Index, _ReceiverId, _ResultTuple, _ErrorsVariant, _ItemReceiverId>>;

      using __operation_base_t = __operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>;

      struct __t : __base_type {
        connect_result_t<_Item, __item_receiver_t> __item_op_;

        template <class _ISndr, class NextRcvr>
        __t(__operation_base_t* __parent_op, _ISndr&& __item, NextRcvr&& __item_rcvr)
          : __base_type((NextRcvr&&) __item_rcvr, __parent_op)
          , __item_op_(connect((_ISndr&&) __item, __item_receiver_t{.__op_ = this})) {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          if (__self.__parent_op_->template __push_back_item_op<_Index>(&__self)) {
            start(__self.__item_op_);
          } else {
            stdexec::set_stopped((_ItemReceiver&&) __self.__item_rcvr_);
          }
        }
      };
    };

    template <
      std::size_t _Index,
      class _ReceiverId,
      class _ResultTuple,
      class _ErrorsVariant,
      class _ItemId>
      requires __valid<__next_sender_of_t, _ReceiverId, __just_sender_t<_ResultTuple>>
    struct __next_sender {
      using __operation_base_t = __operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>;
      using _Item = stdexec::__t<_ItemId>;
      template <class _Self, class _Rcvr>
      using __item_operation_t = stdexec::__t<__item_operation<
        _Index,
        _ReceiverId,
        _ResultTuple,
        _ErrorsVariant,
        __copy_cvref_t<_Self, _ItemId>,
        __id<__decay_t<_Rcvr>>>>;

      template <class _Rcvr>
      using __item_receiver_t = stdexec::__t<
        __item_receiver<_Index, _ReceiverId, _ResultTuple, _ErrorsVariant, __id<__decay_t<_Rcvr>>>>;

      struct __t {
        using completion_signatures =
          stdexec::completion_signatures<set_value_t(), set_stopped_t()>;

        using __id = __next_sender;

        [[no_unique_address]] _Item __item_;
        __operation_base_t* __parent_op_;

        template <__decays_to<__t> _Self, receiver _ItemReceiver>
          requires sender_to<_Item, __item_receiver_t<_ItemReceiver>>
        friend __item_operation_t<_Self, _ItemReceiver>
          tag_invoke(connect_t, _Self&& __self, _ItemReceiver&& __item_rcvr) {
          return {__self.__parent_op_, ((_Self&&) __self).__item_, (_ItemReceiver&&) __item_rcvr};
        }
      };
    };

    template <std::size_t _Is, class _ReceiverId, class _ResultTuple, class _ErrorsVariant>
    struct __receiver {
      template <class _Sndr>
      using __next_sender_t = stdexec::__t<
        __next_sender<_Is, _ReceiverId, _ResultTuple, _ErrorsVariant, __id<__decay_t<_Sndr>>>>;

      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        using __id = __receiver;

        __operation_base<_ReceiverId, _ResultTuple, _ErrorsVariant>* __op_;

        template <sender _Item>
        friend __next_sender_t<_Item> tag_invoke(set_next_t, __t& __self, _Item&& __item) noexcept {
          return __next_sender_t<_Item>{(_Item&&) __item, __self.__op_};
        }

        template <__one_of<set_value_t, set_stopped_t> _Tag>
        friend void tag_invoke(_Tag __complete, __t&& __self) noexcept {
          __self.__op_->__notify_stop();
        }

        template <class _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          __self.__op_->__notify_error((_Error&&) __error);
        }

        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const __t& __self) noexcept {
          using __with_token = __with<get_stop_token_t, in_place_stop_token>;
          auto __token = __with_token{__self.__op_->__stop_source_.get_token()};
          return stdexec::__make_env(get_env(__self.__op_->__receiver_), __token);
        }
      };
    };

    template <class _Tp>
    using __decay_rvalue_ref = __decay_t<_Tp>&&;

    template <class _Sender, class _Env>
    concept __max1_sender =
      sender_in<_Sender, _Env>
      && __valid<__value_types_of_t, _Sender, _Env, __mconst<int>, __msingle_or<void>>;

    template <class _Env, class _Sender>
    using __single_values_of_t = //
      __value_types_of_t<
        _Sender,
        _Env,
        __transform<__q<__decay_rvalue_ref>, __q<__types>>,
        __q<__msingle>>;

    template <class _Env, class _Sender>
    using __values_tuple_t = //
      __value_types_of_t<_Sender, __env_t<_Env>, __q<__decayed_tuple>, __q<__msingle>>;

    template <class _Env, class... _Senders>
    using __set_values_sig_t = //
      completion_signatures<
        __minvoke< __mconcat<__qf<set_value_t>>, __single_values_of_t<_Env, _Senders>...>>;

    template <class _Env, __max1_sender<_Env>... _Senders>
    using __completions_t = //
      __concat_completion_signatures_t<
        completion_signatures<set_stopped_t(), set_error_t(std::exception_ptr&&)>,
        __minvoke<
          __with_default<__mbind_front_q<__set_values_sig_t, _Env>, completion_signatures<>>,
          _Senders...>,
        __make_completion_signatures<
          _Senders,
          _Env,
          completion_signatures<>,
          __mconst<completion_signatures<>>,
          __mcompose<__q<completion_signatures>, __qf<set_error_t>, __q<__decay_rvalue_ref>>>...>;

    template <receiver _Receiver, __max1_sender<env_of_t<_Receiver>>... _Senders>
    struct __traits {

      using __result_tuple = __minvoke<
        __transform<__mbind_front_q<__values_tuple_t, env_of_t<_Receiver>>, __q<std::tuple>>,
        _Senders...>;

      using __errors_variant = //
        __minvoke<
          __mconcat<__transform<__q<__decay_t>, __nullable_variant_t>>,
          error_types_of_t<_Senders, __env_t<env_of_t<_Receiver>>, __types>... >;

      using __operation_base =
        __zip::__operation_base<__id<_Receiver>, __result_tuple, __errors_variant>;

      template <std::size_t _Is>
      using __receiver =
        stdexec::__t<__zip::__receiver<_Is, __id<_Receiver>, __result_tuple, __errors_variant>>;

      template <class _Sender, class _Index>
      using __op_state = sequence_connect_result_t<_Sender, __receiver<__v<_Index>>>;

      template <class _Tuple = __q<std::tuple>>
      using __op_states_tuple = //
        __minvoke<
          __mzip_with2<__q<__op_state>, _Tuple>,
          __types<_Senders...>,
          __mindex_sequence_for<_Senders...>>;
    };

    template <class _From, class _ToId>
    using __cvref_id = __copy_cvref_t<_From, __t<_ToId>>;

    template <class _Cvref, class _ReceiverId, class... _SenderIds>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __traits_t =
        __traits<stdexec::__t<_ReceiverId>, __minvoke<_Cvref, stdexec::__t<_SenderIds>>...>;
      using __operation_base_t = typename __traits_t::__operation_base;
      using __op_states_tuple_t = typename __traits_t::template __op_states_tuple<>;

      template <std::size_t _Index>
      using __receiver_t = typename __traits_t::template __receiver<_Index>;

      class __t : public __operation_base_t {
       public:
        template <class _Rcvr, class _SendersTuple, std::size_t... _Is>
        __t(_Rcvr&& __rcvr, _SendersTuple&& __senders, std::index_sequence<_Is...>)
          : __operation_base_t((_Rcvr&&) __rcvr)
          , __child_ops_(__conv{[&] {
            return sequence_connect(
              std::get<_Is>((_SendersTuple&&) __senders), __receiver_t<_Is>{this});
          }}...) {
        }

        template <class _Rcvr, class _SendersTuple>
        __t(_Rcvr&& __rcvr, _SendersTuple&& __senders)
          : __t(
            (_Rcvr&&) __rcvr,
            (_SendersTuple&&) __senders,
            std::index_sequence_for<_SenderIds...>{}) {
        }
       private:
        __op_states_tuple_t __child_ops_;

        friend void tag_invoke(start_t, __t& __self) noexcept {
          auto token = get_stop_token(get_env(__self.__receiver_));
          __self.__stop_callback_.emplace(token, __on_stop_requested{__self.__stop_source_});
          auto __start_op = [&__self](auto& __op) {
            if (__self.__increase_op_count()) {
              start(__op);
            }
          };
          std::apply([&](auto&... __ops) { (__start_op(__ops), ...); }, __self.__child_ops_);
        }
      };
    };

    template <class _Indices, class... _SenderIds>
    struct __sender;

    template <std::size_t... _Is, class... _SenderIds>
    struct __sender<std::index_sequence<_Is...>, _SenderIds...> {
      template <class _Self, class _Env>
      using __completions_t_ = __zip::__completions_t<_Env, __cvref_id<_Self, _SenderIds>...>;

      template <class _Self, class _Receiver, std::size_t _Index>
      using __receiver_t =
        typename __traits<_Receiver, __cvref_id<_Self, _SenderIds>...>::template __receiver< _Index>;

      template <class _Self, class _Receiver>
      using __operation_t = stdexec::__t<
        __operation<__copy_cvref_fn<_Self>, stdexec::__id<__decay_t<_Receiver>>, _SenderIds...>>;

      class __t {
       public:
        using __id = __sender;

        template <class... _Sndrs>
        explicit(sizeof...(_Sndrs) == 1) __t(_Sndrs&&... __senders)
          : __senders_((_Sndrs&&) __senders...) {
        }

       private:
        template <class _Self, receiver _Receiver>
          requires(__max1_sender<__cvref_id<_Self, _SenderIds>, env_of_t<_Receiver>> && ...)
               && (sequence_sender_to<
                     __cvref_id<_Self, _SenderIds>,
                     __receiver_t<_Self, __decay_t<_Receiver>, _Is>>
                   && ...)
        friend __operation_t<_Self, _Receiver> tag_invoke(
          sequence_connect_t,
          _Self&& __self,
          _Receiver&& __rcvr) {
          return {(_Receiver&&) __rcvr, ((_Self&&) __self).__senders_};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, _Env)
          -> __completions_t_<_Self, _Env>;

        friend empty_env tag_invoke(get_env_t, const __t& __self) noexcept {
          return {};
        }

        std::tuple<stdexec::__t<_SenderIds>...> __senders_;
      };
    };

    template <class... _Senders>
    using __sender_t =
      __t<__sender<std::index_sequence_for<_Senders...>, __id<__decay_t<_Senders>>...>>;

    struct zip_t {
      template <sender... _Senders>
        requires tag_invocable<zip_t, _Senders...>
              && sender<tag_invoke_result_t<zip_t, _Senders...>>
      auto operator()(_Senders&&... __sndrs) const
        noexcept(nothrow_tag_invocable<zip_t, _Senders...>)
          -> tag_invoke_result_t<zip_t, _Senders...> {
        return tag_invoke(*this, (_Senders&&) __sndrs...);
      }

      template <sender... _Senders>
        requires(!tag_invocable<zip_t, _Senders...>) && sender<__sender_t<_Senders...>>
      constexpr __sender_t<_Senders...> operator()(_Senders&&... __senders) const {
        return __sender_t<_Senders...>{(_Senders&&) __senders...};
      }
    };
  } // namespace __zip

  using __zip::zip_t;
  inline constexpr zip_t zip;
} // namespace exec