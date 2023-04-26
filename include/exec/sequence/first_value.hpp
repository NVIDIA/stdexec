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
  namespace __first_value {
    using namespace stdexec;

    template <class _BaseEnv>
    using __env_t = __make_env_t<_BaseEnv, __with<get_stop_token_t, in_place_stop_token>>;

    struct __on_stop_requested {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _ReceiverId, class _ValuesVariant>
    struct __operation_base {
      enum __state {
        __empty,
        __emplace_in_progress,
        __emplaced
      };

      using _Receiver = __t<_ReceiverId>;
      using __on_stop =
        typename stop_token_of_t<env_of_t<_Receiver&>>::template callback_type<__on_stop_requested>;

      [[no_unique_address]] _Receiver __rcvr_;
      std::atomic<__state> __state_{__state::__empty};
      _ValuesVariant __values_{};
      in_place_stop_source __stop_source_{};
      std::optional<__on_stop> __on_stop_{};

      template <class... _Args>
      void __notify_value(_Args&&... __args) noexcept {
        __state __expected = __state::__empty;
        if (__state_.compare_exchange_strong(
              __expected, __state::__emplace_in_progress, std::memory_order_relaxed)) {
          __values_.template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
          __stop_source_.request_stop();
          __state_.store(__state::__emplaced, std::memory_order_release);
        }
      }

      struct __visitor {
        __operation_base* __self;

        void operator()(std::monostate) const noexcept {
          STDEXEC_ASSERT(false);
        }

        template <class... _Args>
        void operator()(std::tuple<_Args...>&& __tuple) const noexcept {
          std::apply(
            [this](_Args&&... __args) noexcept -> void {
              stdexec::set_value(
                static_cast<_Receiver&&>(__self->__rcvr_), static_cast<_Args&&>(__args)...);
            },
            static_cast<std::tuple<_Args...>&&>(__tuple));
        }
      };

      void __notify_completion() noexcept {
        __state __result_state = __state_.load(std::memory_order_acquire);
        switch (__result_state) {
        case __state::__empty:
          stdexec::set_stopped(static_cast<_Receiver&&>(__rcvr_));
          break;
        case __state::__emplace_in_progress:
          stdexec::set_error(
            static_cast<_Receiver&&>(__rcvr_),
            std::make_exception_ptr(std::logic_error(
              "sequence-receiver contract have been "
              "violated: A completion function have been called before all item senders have been "
              "completed")));
          break;
        case __state::__emplaced:
          STDEXEC_ASSERT(__values_.index());
          std::visit(__visitor{this}, static_cast<_ValuesVariant&&>(__values_));
        }
      }
    };

    template <class _ItemReceiverId, class _ReceiverId, class _ValuesVariant>
    struct __item_op_base {
      using _ItemReceiver = __t<_ItemReceiverId>;

      __item_op_base(
        _ItemReceiver&& __rcvr,
        __operation_base<_ReceiverId, _ValuesVariant>* __parent)
        : __item_rcvr_{static_cast<_ItemReceiver&&>(__rcvr)}
        , __parent_{__parent} {
      }

      [[no_unique_address]] _ItemReceiver __item_rcvr_;
      __operation_base<_ReceiverId, _ValuesVariant>* __parent_;
      in_place_stop_source __stop_source_{};

      using __on_item_stop = typename stop_token_of_t<
        env_of_t<_ItemReceiver&>>::template callback_type<__on_stop_requested>;
      std::optional<__on_item_stop> __on_item_stop_{};
      std::optional<in_place_stop_token::callback_type<__on_stop_requested>> __on_parent_stop_{};
    };

    template <class _ItemReceiverId, class _ReceiverId, class _ValuesVariant>
    struct __item_receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;

      struct __t {
        using __id = __item_receiver;
        __item_op_base<_ItemReceiverId, _ReceiverId, _ValuesVariant>* __op_;

        template <same_as<set_stopped_t> _Tag, same_as<__t> _Self>
        friend void tag_invoke(_Tag, _Self&& __self) noexcept {
          __self.__op_->__on_item_stop_.reset();
          __self.__op_->__on_parent_stop_.reset();
          stdexec::set_value(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<set_value_t> _Tag, same_as<__t> _Self, class... _Args>
        friend void tag_invoke(_Tag, _Self&& __self, _Args&&... __args) noexcept {
          __self.__op_->__parent_->__notify_value(static_cast<_Args&&>(__args)...);
          __self.__op_->__on_item_stop_.reset();
          __self.__op_->__on_parent_stop_.reset();
          stdexec::set_value(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<set_error_t> _Tag, same_as<__t> _Self, class _Error>
        friend void tag_invoke(_Tag, _Self&& __self, _Error&& __error) noexcept {
          __self.__op_->__on_item_stop_.reset();
          __self.__op_->__on_parent_stop_.reset();
          stdexec::set_value(static_cast<_ItemReceiver&&>(__self.__op_->__item_rcvr_));
        }

        template <same_as<__t> _Self>
        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return stdexec::__make_env(
            stdexec::get_env(static_cast<const _ItemReceiver&>(__self.__op_->__item_rcvr_)),
            __with_(get_stop_token, __self.__op_->__stop_source_.get_token()));
        }
      };
    };

    template <class _Item, class _ItemReceiverId, class _ReceiverId, class _ValuesVariant>
    struct __item_operation {
      using __base_op_t = __item_op_base<_ItemReceiverId, _ReceiverId, _ValuesVariant>;
      using _ItemReceiver = stdexec::__t<_ItemReceiverId>;
      using __item_receiver_t =
        stdexec::__t<__item_receiver<_ItemReceiverId, _ReceiverId, _ValuesVariant>>;

      struct __t : __base_op_t {
        using __id = __item_operation;
        connect_result_t<_Item, __item_receiver_t> __op_;

        __t(
          _Item&& __item,
          _ItemReceiver&& __item_rcvr,
          __operation_base<_ReceiverId, _ValuesVariant>* __parent)
          : __base_op_t(static_cast<_ItemReceiver&&>(__item_rcvr), __parent)
          , __op_{stdexec::connect(static_cast<_Item&&>(__item), __item_receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__on_item_stop_.emplace(
            stdexec::get_stop_token(stdexec::get_env(__self.__item_rcvr_)),
            __on_stop_requested{__self.__stop_source_});
          __self.__on_parent_stop_.emplace(
            __self.__parent_->__stop_source_.get_token(),
            __on_stop_requested{__self.__stop_source_});
          stdexec::start(__self.__op_);
        }
      };
    };

    template <class _ItemId, class _ReceiverId, class _ValuesVariant>
    struct __item_sender {

      using _Item = stdexec::__t<_ItemId>;

      template <class _Self, class _Rcvr>
      using __item_operation_t = stdexec::__t<__item_operation<
        __copy_cvref_t<_Self, _Item>,
        __id<__decay_t<_Rcvr>>,
        _ReceiverId,
        _ValuesVariant>>;

      template <class _Rcvr>
      using __item_receiver_t =
        stdexec::__t<__item_receiver< __id<__decay_t<_Rcvr>>, _ReceiverId, _ValuesVariant>>;

      struct __t {
        using __id = __item_sender;
        using completion_signatures = stdexec::completion_signatures<set_value_t()>;

        [[no_unique_address]] _Item __item_;
        __operation_base<_ReceiverId, _ValuesVariant>* __op_;

        template <__decays_to<__t> _Self, receiver_of<completion_signatures> _ItemReceiver>
          requires sender_to<__copy_cvref_t<_Self, _Item>, __item_receiver_t<_ItemReceiver>>
        friend auto tag_invoke(connect_t, _Self&& __self, _ItemReceiver&& __rcvr)
          -> __item_operation_t<_Self, _ItemReceiver> {
          return {
            static_cast<_Self&&>(__self).__item_,
            static_cast<_ItemReceiver&&>(__rcvr),
            __self.__op_};
        }
      };
    };

    template <class _ReceiverId, class _ValuesVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      template <class _Item>
      using __item_sender_t =
        stdexec::__t<__item_sender<__id<__decay_t<_Item>>, _ReceiverId, _ValuesVariant>>;

      struct __t {
        __operation_base<_ReceiverId, _ValuesVariant>* __op_;

        template <same_as<set_next_t> _SetNext, same_as<__t> _Self, sender _Item>
        friend auto tag_invoke(_SetNext, _Self& __self, _Item&& __item) noexcept
          -> __item_sender_t<_Item> {
          return {static_cast<_Item&&>(__item), __self.__op_};
        }

        template <same_as<set_stopped_t> _SetStopped, same_as<__t> _Self>
        friend void tag_invoke(_SetStopped, _Self&& __self) noexcept {
          if (
            !__self.__op_->__stop_source_.stop_requested()
            || stdexec::get_stop_token(stdexec::get_env(__self.__op_->__rcvr_)).stop_requested()) {
            stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
          } else {
            __self.__op_->__notify_completion();
          }
        }

        template <same_as<set_value_t> _SetValue, same_as<__t> _Self>
        friend void tag_invoke(_SetValue, _Self&& __self) noexcept {
          __self.__op_->__notify_completion();
        }

        template <same_as<set_error_t> _SetError, same_as<__t> _Self, class _Error>
        friend void tag_invoke(_SetError, _Self&& __self, _Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(__error));
        }

        template <same_as<__t> _Self>
        friend __env_t<env_of_t<_Receiver>> tag_invoke(get_env_t, const _Self& __self) noexcept {
          return stdexec::__make_env(
            stdexec::get_env(__self.__op_->__rcvr_),
            stdexec::__with_(get_stop_token, __self.__op_->__stop_source_.get_token()));
        }
      };
    };

    template <class _Sender, class _ReceiverId, class _ValuesVariant>
    struct __operation {
      using _Receiver = stdexec::__t<_ReceiverId>;
      using __receiver_t = stdexec::__t<__receiver<_ReceiverId, _ValuesVariant>>;

      struct __t : __operation_base<_ReceiverId, _ValuesVariant> {
        sequence_connect_result_t<_Sender, __receiver_t> __op_;

        __t(_Sender&& __sndr, _Receiver&& __rcvr)
          : __operation_base<_ReceiverId, _ValuesVariant>{static_cast<_Receiver&&>(__rcvr)}
          , __op_{exec::sequence_connect(static_cast<_Sender&&>(__sndr), __receiver_t{this})} {
        }

        friend void tag_invoke(start_t, __t& __self) noexcept {
          __self.__on_stop_.emplace(
            get_stop_token(get_env(__self.__rcvr_)), __on_stop_requested{__self.__stop_source_});
          start(__self.__op_);
        }
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      template <class _Sender, class _Rcvr>
      using __values_variant_t =
        __value_types_of_t<_Sender, env_of_t<_Rcvr>, __q<__decayed_tuple>, __nullable_variant_t>;

      template <class _Self, class _Rcvr>
      using __receiver_t = stdexec::__t<__receiver<
        __id<__decay_t<_Rcvr>>,
        __values_variant_t<__copy_cvref_t<_Self, _Sender>, _Rcvr>>>;

      template <class _Self, class _Rcvr>
      using __operation_t = stdexec::__t<__operation<
        __copy_cvref_t<_Self, _Sender>,
        __id<__decay_t<_Rcvr>>,
        __values_variant_t<__copy_cvref_t<_Self, _Sender>, _Rcvr>>>;

      template <class _Self, class _Env>
      using __completion_sigs = make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>,
            __compl_sigs::__default_set_value>;

      struct __t {
        [[no_unique_address]] _Sender __sndr;

        template <__decays_to<__t> _Self, class _Rcvr>
          requires receiver_of<_Rcvr, __completion_sigs<_Self, env_of_t<_Rcvr>>>
                // && sequence_sender_to<__copy_cvref_t<_Self, _Sender>, __receiver_t<_Self, _Rcvr>>
        friend auto tag_invoke(connect_t, _Self&& __self, _Rcvr&& __rcvr)
          -> __operation_t<_Self, _Rcvr> {
          return {static_cast<_Self&&>(__self).__sndr, static_cast<_Rcvr&&>(__rcvr)};
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> __completion_sigs<_Self, _Env>;
      };
    };

    template <class _Sndr>
    using __sender_t = __t<__sender<__id<__decay_t<_Sndr>>>>;

    struct first_value_t {
      template <class _Sender>
        requires tag_invocable<first_value_t, _Sender>
      auto operator()(_Sender&& __sender) const
        noexcept(nothrow_tag_invocable<first_value_t, _Sender>)
          -> tag_invoke_result_t<first_value_t, _Sender> {
        return tag_invoke(*this, static_cast<_Sender&&>(__sender));
      }

      template <class _Sender>
        requires(!tag_invocable<first_value_t, _Sender>) && sender<_Sender>
      auto operator()(_Sender&& __sender) const -> __sender_t<_Sender> {
        return __sender_t<_Sender>{static_cast<_Sender&&>(__sender)};
      }
    };

  } // namespace __first_value

  using __first_value::first_value_t;
  inline constexpr first_value_t first_value{};
} // namespace exec