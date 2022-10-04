/*
 * Copyright (c) 2022 NVIDIA Corporation
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

#include <execution.hpp>
#include <type_traits>

#include "common.cuh"
#include "schedulers/detail/throw_on_cuda_error.cuh"

namespace example::cuda::stream {
  namespace split {
    template <class _SenderId, class _SharedState>
      class __receiver : public receiver_base_t {
        using Sender = _P2300::__t<_SenderId>;

        _SharedState &__sh_state_;

      public:
        template <_P2300::__one_of<std::execution::set_value_t, std::execution::set_error_t, std::execution::set_stopped_t> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
        friend void tag_invoke(_Tag __tag, __receiver&& __self, _As&&... __as) noexcept {
          _SharedState &__state = __self.__sh_state_;
          cudaStream_t stream = __state.__op_state2_.stream_;
          THROW_ON_CUDA_ERROR(cudaStreamSynchronize(stream));

          _NVCXX_EXPAND_PACK(_As, __as,
            using __tuple_t = decayed_tuple<_Tag, _As...>;
            __state.__data_->template emplace<__tuple_t>(_Tag{}, __as...);
          )
          __state.__notify();
        }

        friend auto tag_invoke(std::execution::get_env_t, const __receiver& __self)
          -> std::execution::make_env_t<std::execution::with_t<std::execution::get_stop_token_t, std::execution::in_place_stop_token>> {
          return std::execution::make_env(std::execution::with(std::execution::get_stop_token, __self.__sh_state_.__stop_source_.get_token()));
        }

        explicit __receiver(_SharedState &__sh_state) noexcept
          : __sh_state_(__sh_state) {
        }
    };

    struct __operation_base {
      using __notify_fn = void(__operation_base*) noexcept;

      __operation_base * __next_{};
      __notify_fn* __notify_{};
    };

    template <class _SenderId>
      struct __sh_state {
        using _Sender = _P2300::__t<_SenderId>;

        template <class... _Ts>
          using __bind_tuples =
            _P2300::__mbind_front_q<
              variant_t,
              tuple_t<std::execution::set_stopped_t>, // Initial state of the variant is set_stopped
              // tuple_t<std::execution::set_error_t, cudaError_t>,
              // tuple_t<std::execution::set_error_t, std::exception_ptr>,
              _Ts...>;

        using __bound_values_t =
          std::execution::__value_types_of_t<
            _Sender,
            std::execution::make_env_t<std::execution::with_t<std::execution::get_stop_token_t, std::execution::in_place_stop_token>>,
            _P2300::__mbind_front_q<decayed_tuple, std::execution::set_value_t>,
            _P2300::__q<__bind_tuples>>;

        using __variant_t =
          std::execution::__error_types_of_t<
            _Sender,
            std::execution::make_env_t<std::execution::with_t<std::execution::get_stop_token_t, std::execution::in_place_stop_token>>,
            _P2300::__transform<
              _P2300::__mbind_front_q<decayed_tuple, std::execution::set_error_t>,
              __bound_values_t>>;

        using __receiver_ = __receiver<_SenderId, __sh_state>;
        using inner_op_state_t = std::execution::connect_result_t<_Sender, __receiver_>;

        std::execution::in_place_stop_source __stop_source_{};
        inner_op_state_t __op_state2_;
        __variant_t *__data_;
        std::atomic<void*> __head_;

        explicit __sh_state(_Sender& __sndr)
          : __op_state2_(std::execution::connect((_Sender&&) __sndr, __receiver_{*this}))
          , __head_{nullptr} {
          THROW_ON_CUDA_ERROR(cudaMallocManaged(&__data_, sizeof(__variant_t)));
          new (__data_) __variant_t();
        }

        ~__sh_state() {
          THROW_ON_CUDA_ERROR(cudaFree(__data_));
        }

        void __notify() noexcept {
          void* const __completion_state = static_cast<void*>(this);
          void *__old = __head_.exchange(__completion_state, std::memory_order_acq_rel);
          __operation_base *__op_state = static_cast<__operation_base*>(__old);

          while(__op_state != nullptr) {
            __operation_base *__next = __op_state->__next_;
            __op_state->__notify_(__op_state);
            __op_state = __next;
          }
        }
      };

    // TODO Stream operation
    template <class _SenderId, class _ReceiverId>
      class __operation : public __operation_base {
        using _Sender = _P2300::__t<_SenderId>;
        using _Receiver = _P2300::__t<_ReceiverId>;

        struct __on_stop_requested {
          std::execution::in_place_stop_source& __stop_source_;
          void operator()() noexcept {
            __stop_source_.request_stop();
          }
        };
        using __on_stop = std::optional<typename std::execution::stop_token_of_t<
            std::execution::env_of_t<_Receiver> &>::template callback_type<__on_stop_requested>>;

        _Receiver __recvr_;
        __on_stop __on_stop_{};
        std::shared_ptr<__sh_state<_SenderId>> __shared_state_;

      public:
        __operation(_Receiver&& __rcvr,
                    std::shared_ptr<__sh_state<_SenderId>> __shared_state)
            noexcept(std::is_nothrow_move_constructible_v<_Receiver>)
          : __operation_base{nullptr, __notify}
          , __recvr_((_Receiver&&)__rcvr)
          , __shared_state_(move(__shared_state)) {
        }
        _P2300_IMMOVABLE(__operation);

        static void __notify(__operation_base* __self) noexcept {
          __operation *__op = static_cast<__operation*>(__self);
          __op->__on_stop_.reset();

          visit([&](auto& __tupl) noexcept -> void {
            apply([&](auto __tag, auto&&... __args) noexcept -> void {
              __tag((_Receiver&&) __op->__recvr_, __args...);
            }, __tupl);
          }, *__op->__shared_state_->__data_);
        }

        friend void tag_invoke(std::execution::start_t, __operation& __self) noexcept {
          __sh_state<_SenderId>* __shared_state = __self.__shared_state_.get();
          std::atomic<void*>& __head = __shared_state->__head_;
          void* const __completion_state = static_cast<void*>(__shared_state);
          void* __old = __head.load(std::memory_order_acquire);

          if (__old != __completion_state) {
            __self.__on_stop_.emplace(
                std::execution::get_stop_token(std::execution::get_env(__self.__recvr_)),
                __on_stop_requested{__shared_state->__stop_source_});
          }

          do {
            if (__old == __completion_state) {
              __self.__notify(&__self);
              return;
            }
            __self.__next_ = static_cast<__operation_base*>(__old);
          } while (!__head.compare_exchange_weak(
              __old, static_cast<void *>(&__self),
              std::memory_order_release,
              std::memory_order_acquire));

          if (__old == nullptr) {
            // the inner sender isn't running
            if (__shared_state->__stop_source_.stop_requested()) {
              // 1. resets __head to completion state
              // 2. notifies waiting threads
              // 3. propagates "stopped" signal to `out_r'`
              __shared_state->__notify();
            } else {
              std::execution::start(__shared_state->__op_state2_);
            }
          }
        }
      };
  } // namespace split

  template <class _SenderId>
    class split_sender_t : sender_base_t {
      using _Sender = _P2300::__t<_SenderId>;
      using __sh_state_ = split::__sh_state<_SenderId>;
      template <class _Receiver>
        using __operation = split::__operation<_SenderId, _P2300::__x<std::remove_cvref_t<_Receiver>>>;

      _Sender __sndr_;
      std::shared_ptr<__sh_state_> __shared_state_;

    public:
      template <_P2300::__decays_to<split_sender_t> _Self, std::execution::receiver _Receiver>
          requires std::execution::receiver_of<_Receiver, std::execution::completion_signatures_of_t<_Self, std::execution::__empty_env>>
        friend auto tag_invoke(std::execution::connect_t, _Self&& __self, _Receiver&& __recvr)
          noexcept(std::is_nothrow_constructible_v<std::decay_t<_Receiver>, _Receiver>)
          -> __operation<_Receiver> {
          return __operation<_Receiver>{(_Receiver &&) __recvr,
                                        __self.__shared_state_};
        }

      template <std::execution::tag_category<std::execution::forwarding_sender_query> _Tag, class... _As _NVCXX_CAPTURE_PACK(_As)>
          requires (!_P2300::__is_instance_of<_Tag, std::execution::get_completion_scheduler_t>) &&
            _P2300::__callable<_Tag, const _Sender&, _As...>
        friend auto tag_invoke(_Tag __tag, const split_sender_t& __self, _As&&... __as)
          noexcept(_P2300::__nothrow_callable<_Tag, const _Sender&, _As...>)
          -> _P2300::__call_result_if_t<std::execution::tag_category<_Tag, std::execution::forwarding_sender_query>, _Tag, const _Sender&, _As...> {
          _NVCXX_EXPAND_PACK_RETURN(_As, __as,
            return ((_Tag&&) __tag)(__self.__sndr_, (_As&&) __as...);
          )
        }

      template <class... _Tys>
      using __set_value_t = std::execution::completion_signatures<std::execution::set_value_t(const std::decay_t<_Tys>&...)>;

      template <class _Ty>
      using __set_error_t = std::execution::completion_signatures<std::execution::set_error_t(const std::decay_t<_Ty>&)>;

      template <_P2300::__decays_to<split_sender_t> _Self, class _Env>
        friend auto tag_invoke(std::execution::get_completion_signatures_t, _Self&&, _Env) ->
          std::execution::make_completion_signatures<
            _Sender,
            std::execution::make_env_t<std::execution::with_t<std::execution::get_stop_token_t, std::execution::in_place_stop_token>>,
            std::execution::completion_signatures<std::execution::set_error_t(const std::exception_ptr&)>,
            __set_value_t,
            __set_error_t>;

      explicit split_sender_t(_Sender __sndr)
          : __sndr_((_Sender&&) __sndr)
          , __shared_state_{std::make_shared<__sh_state_>(__sndr_)}
      {}
    };
}

