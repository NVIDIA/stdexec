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

    struct __on_stop_requested {
      in_place_stop_source& __stop_source_;

      void operator()() noexcept {
        __stop_source_.request_stop();
      }
    };

    template <class _ReceiverId, class _ValuesVariant>
    struct __operation_base {
      using _Receiver = __t<_ReceiverId>;
      using __on_stop =
        stop_token_of_t<env_of_t<_Receiver&>>::template callback_type<__on_stop_requested>;

      [[no_unique_address]] _Receiver __rcvr_;
      std::mutex __mutex_;
      _ValuesVariant __values_{};
      in_place_stop_source __stop_source_{};
      std::optional<__on_stop> __on_stop_{};

      template <class... _Args>
      void __notify_value(_Args&&... __args) noexcept {
        std::scoped_lock __lock{__mutex_};
        if (!__values_.index()) {
          __values_.template emplace<__decayed_tuple<_Args...>>(static_cast<_Args&&>(__args)...);
          __stop_source_.request_stop();
        }
      }

      void __notify_completion() noexcept {
        std::scoped_lock __lock{__mutex_};
        if (__values_.index()) {
          std::visit(
            [this]<class... _Args>(std::tuple<_Args...>&& __args) noexcept {
              std::apply(
                [this](_Args&&... __args) noexcept {
                  stdexec::set_value(
                    static_cast<_Receiver&&>(__rcvr_), static_cast<_Args&&>(__args)...);
                },
                static_cast<std::tuple<_Args...>&&>(__args));
            },
            static_cast<_ValuesVariant&&>(__values_));
        } else {
          stdexec::set_error(
            static_cast<_Receiver&&>(__rcvr_),
            std::make_exception_ptr(std::runtime_error("No value was produced by the sequence "
                                                       "sender")));
        }
      }
    };

    template <class _ReceiverId, class _ValuesVariant>
    struct __receiver {
      using _Receiver = stdexec::__t<_ReceiverId>;

      struct __t {
        __operation_base<_ReceiverId, _ValuesVariant>* __op_;

        template <class _Item>
        // TODO require that each value completion of _Item should be emplaceable into _ValuesVariant
        friend void tag_invoke(set_next_t, __t& __self, _Item&& __item) noexcept {
          return __item_sender_t<_Item>{static_cast<_Item&&>(__item), __self.__op_};
        }

        friend void tag_invoke(set_stopped_t, __t&& __self) noexcept {
          stdexec::set_stopped(static_cast<_Receiver&&>(__self.__op_->__rcvr_));
        }

        friend void tag_invoke(set_value_t, __t&& __self) noexcept {
          __self.__op_->__notify_completion();
        }

        template <class _Error>
        friend void tag_invoke(set_error_t, __t&& __self, _Error&& __error) noexcept {
          stdexec::set_error(
            static_cast<_Receiver&&>(__self.__op_->__rcvr_), static_cast<_Error&&>(__error));
        }
      };
    };

    template <class _SenderId>
    struct __sender {
      using _Sender = stdexec::__t<_SenderId>;

      struct __t {
        [[no_unique_address]] _Sender __sndr;

        template <__decays_to<__t> _Self, class _Rcvr>
        friend void tag_invoke(connect_t, _Self&& __self, _Rcvr&& __rcvr) {
          return exec::sequence_connect(
            static_cast<__copy_cvref_t<_Self, _Sender>>(__sndr),
            __receiver_t<_Rcvr>(static_cast<_Rcvr&&>(__rcvr)));
        }

        template <__decays_to<__t> _Self, class _Env>
        friend auto tag_invoke(get_completion_signatures_t, _Self&&, const _Env&)
          -> make_completion_signatures<
            __copy_cvref_t<_Self, _Sender>,
            _Env,
            completion_signatures<set_error_t(std::exception_ptr), set_stopped_t()>>;
      };
    };

    template <class _Sndr>
    using __sender_t = __t<__sender<__id<__decay_t<_Sndr>>>>;

    struct front_t {
      template <class _Sender>
        requires tag_invocable<front_t, _Sender>
      auto operator()(_Sender&& __sender) const noexcept(nothrow_tag_invocable<front_t, _Sender>)
        -> tag_invoke_result_t<front_t, _Sender> {
        return tag_invoke(*this, static_cast<_Sender&&>(__sender));
      }

      template <class _Sender>
        requires(!tag_invocable<front_t, _Sender>) && sender<_Sender>
      auto operator()(_Sender&& __sender) const -> __sender_t<_Sender> {
        return __sender_t<_Sender>{static_cast<_Sender&&>(__sender)};
      }
    };

  } // namespace __first_value

  using __first_value::front_t;
  inline constexpr front_t front{};
} // namespace exec