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

#include <stdexec/execution.hpp>

#include <cuda/std/type_traits>
#include <cuda/atomic>
#include <cuda/std/tuple>

#include "nvexec/detail/throw_on_cuda_error.cuh"
#include "nvexec/detail/queue.cuh"
#include "nvexec/detail/variant.cuh"
#include "nvexec/detail/tuple.cuh"

namespace nvexec {

  enum class device_type {
    host,
    device
  };

#if defined(__clang__) && defined(__CUDA__)
  __host__ inline device_type get_device_type() { return device_type::host; }
  __device__ inline device_type get_device_type() { return device_type::device; }
#else
  __host__ __device__ inline device_type get_device_type() {
    NV_IF_TARGET(NV_IS_HOST,
                 (return device_type::host;),
                 (return device_type::device;));
  }
#endif

  inline __host__ __device__ bool is_on_gpu() {
    return get_device_type() == device_type::device;
  }
}

namespace nvexec {

  struct stream_context;
  struct stream_scheduler;
  struct stream_sender_base {};
  struct stream_receiver_base {
    constexpr static std::size_t memory_allocation_size = 0;
  };

  template <class... Ts>
    using decayed_tuple = tuple_t<std::decay_t<Ts>...>;

  template <class S>
    concept stream_sender =
      std::execution::sender<S> &&
      std::is_base_of_v<stream_sender_base, std::decay_t<S>>;

  template <class R>
    concept stream_receiver =
      std::execution::receiver<R> &&
      std::is_base_of_v<stream_receiver_base, std::decay_t<R>>;

  namespace detail {
    struct stream_op_state_base{};

    template <class EnvId, class VariantId>
      class stream_enqueue_receiver : public stream_receiver_base {
        using Env = stdexec::__t<EnvId>;
        using Variant = stdexec::__t<VariantId>;

        Env env_;
        Variant* variant_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

      public:
        template <stdexec::__one_of<std::execution::set_value_t,
                                std::execution::set_error_t,
                                std::execution::set_stopped_t> Tag,
                  class... As _NVCXX_CAPTURE_PACK(As)>
          friend void tag_invoke(Tag tag, stream_enqueue_receiver&& self, As&&... as) noexcept {
            _NVCXX_EXPAND_PACK(As, as,
              self.variant_->template emplace<decayed_tuple<Tag, As...>>(Tag{}, (As&&)as...);
            );
            self.producer_(self.task_);
          }


        friend Env tag_invoke(std::execution::get_env_t, const stream_enqueue_receiver& self) {
          return self.env_;
        }

        stream_enqueue_receiver(
            Env env,
            Variant* variant,
            queue::task_base_t* task,
            queue::producer_t producer)
          : env_(env)
          , variant_(variant)
          , task_(task)
          , producer_(producer) {}
      };

    template <class Receiver, class Tag, class... As>
      __launch_bounds__(1) __global__ void continuation_kernel(Receiver receiver, Tag tag, As... as) {
        tag(std::move(receiver), (As&&)as...);
      }

    template <stream_receiver Receiver, class Variant>
      struct continuation_task_t : queue::task_base_t {
        Receiver receiver_;
        Variant* variant_;
        cudaError_t status_{cudaSuccess};

        continuation_task_t (Receiver receiver, Variant* variant) noexcept 
          : receiver_{receiver}
          , variant_{variant} {
          this->execute_ = [](task_base_t* t) noexcept {
            continuation_task_t &self = *reinterpret_cast<continuation_task_t*>(t);

            visit([&self](auto& tpl) noexcept {
                apply([&self](auto tag, auto... as) noexcept {
                  tag(std::move(self.receiver_), decltype(as)(as)...);
                }, tpl);
            }, *self.variant_);
          };
          this->next_ = nullptr;

          constexpr std::size_t ptr_size = sizeof(this->atom_next_);
          status_ = STDEXEC_DBG_ERR(cudaMalloc(&this->atom_next_, ptr_size));

          if (status_ == cudaSuccess) {
            status_ = STDEXEC_DBG_ERR(cudaMemset(this->atom_next_, 0, ptr_size));
          }
        }

        ~continuation_task_t() {
          STDEXEC_DBG_ERR(cudaFree(this->atom_next_));
        }
      };
  }

  template <class OuterReceiverId>
    struct operation_state_base_t : detail::stream_op_state_base {
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;

      bool owner_{false};
      cudaStream_t stream_{0};
      void *temp_storage_{nullptr};
      outer_receiver_t receiver_;
      cudaError_t status_{cudaSuccess};

      operation_state_base_t(outer_receiver_t receiver)
        : receiver_(receiver) {}

      template <class Tag, class... As _NVCXX_CAPTURE_PACK(As)>
      void propagate_completion_signal(Tag tag, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          if constexpr (stream_receiver<outer_receiver_t>) {
            tag((outer_receiver_t&&)receiver_, (As&&)as...);
          } else {
            detail::continuation_kernel
              <std::decay_t<outer_receiver_t>, Tag, As...>
              <<<1, 1, 0, stream_>>>(
                receiver_, tag, (As&&)as...);
          }
        );
      }

      cudaStream_t allocate() {
        if (stream_ == 0) {
          owner_ = true;
          status_ = STDEXEC_DBG_ERR(cudaStreamCreate(&stream_));
        }

        return stream_;
      }

      ~operation_state_base_t() {
        if (owner_) {
          STDEXEC_DBG_ERR(cudaStreamDestroy(stream_));
          stream_ = 0;
          owner_ = false;
        }

        if (temp_storage_) {
          STDEXEC_DBG_ERR(cudaFree(temp_storage_));
        }
      }
    };

  template <class ReceiverId>
    struct propagate_receiver_t : stream_receiver_base {
      operation_state_base_t<ReceiverId>& operation_state_;

      template <stdexec::__one_of<std::execution::set_value_t,
                              std::execution::set_error_t,
                              std::execution::set_stopped_t> Tag,
                class... As  _NVCXX_CAPTURE_PACK(As)>
      friend void tag_invoke(Tag tag, propagate_receiver_t&& self, As&&... as) noexcept {
        _NVCXX_EXPAND_PACK(As, as,
          self.operation_state_.template propagate_completion_signal<Tag, As...>(tag, (As&&)as...);
        );
      }

      friend std::execution::env_of_t<stdexec::__t<ReceiverId>>
      tag_invoke(std::execution::get_env_t, const propagate_receiver_t& self) {
        return std::execution::get_env(self.operation_state_.receiver_);
      }
    };

  namespace detail {
    template <class SenderId, class InnerReceiverId, class OuterReceiverId>
      struct operation_state_t : operation_state_base_t<OuterReceiverId> {
        using sender_t = stdexec::__t<SenderId>;
        using inner_receiver_t = stdexec::__t<InnerReceiverId>;
        using outer_receiver_t = stdexec::__t<OuterReceiverId>;
        using env_t = std::execution::env_of_t<outer_receiver_t>;

        template <class... _Ts>
          using variant =
            stdexec::__minvoke<
              stdexec::__if_c<
                sizeof...(_Ts) != 0,
                stdexec::__transform<stdexec::__q1<std::decay_t>, stdexec::__munique<stdexec::__q<variant_t>>>,
                stdexec::__mconst<stdexec::__not_a_variant>>,
              _Ts...>;

        template <class... _Ts>
          using bind_tuples =
            stdexec::__mbind_front_q<
              variant,
              tuple_t<std::execution::set_error_t, cudaError_t>,
              _Ts...>;

        using bound_values_t =
          stdexec::__value_types_of_t<
            sender_t,
            env_t,
            stdexec::__mbind_front_q<decayed_tuple, std::execution::set_value_t>,
            stdexec::__q<bind_tuples>>;

        using variant_t =
          stdexec::__error_types_of_t<
            sender_t,
            env_t,
            stdexec::__transform<
              stdexec::__mbind_front_q<decayed_tuple, std::execution::set_error_t>,
              bound_values_t>>;

        using task_t = detail::continuation_task_t<inner_receiver_t, variant_t>;
        using intermediate_receiver = stdexec::__t<std::conditional_t<
          stream_sender<sender_t>,
          stdexec::__x<inner_receiver_t>,
          stdexec::__x<detail::stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>>>>;
        using inner_op_state_t = std::execution::connect_result_t<sender_t, intermediate_receiver>;

        friend void tag_invoke(std::execution::start_t, operation_state_t& op) noexcept {
          op.stream_ = op.get_stream();

          if (op.status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            op.propagate_completion_signal(std::execution::set_error, std::move(op.status_));
            return;
          }

          if constexpr (stream_receiver<inner_receiver_t>) {
            if (inner_receiver_t::memory_allocation_size) {
              if (cudaError_t status = 
                    STDEXEC_DBG_ERR(cudaMallocManaged(&op.temp_storage_, inner_receiver_t::memory_allocation_size)); 
                    status != cudaSuccess) {
                // Couldn't allocate memory for intermediate receiver, complete with error
                op.propagate_completion_signal(std::execution::set_error, std::move(status));
                return;
              }
            }
          }

          std::execution::start(op.inner_op_);
        }

        cudaStream_t get_stream() {
          cudaStream_t stream{};

          if constexpr (std::is_base_of_v<detail::stream_op_state_base, inner_op_state_t>) {
            stream = inner_op_.get_stream();
          } else {
            stream = this->allocate();
          }

          return stream;
        }

        template <stdexec::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t>
        operation_state_t(sender_t&& sender, queue::task_hub_t*, OutR&& out_receiver, ReceiverProvider receiver_provider)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver)
          , inner_op_{std::execution::connect((sender_t&&)sender, receiver_provider(*this))}
        {}

        template <stdexec::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires (!stream_sender<sender_t>)
        operation_state_t(sender_t&& sender, queue::task_hub_t* hub, OutR&& out_receiver, ReceiverProvider receiver_provider)
          : operation_state_base_t<OuterReceiverId>((outer_receiver_t&&)out_receiver)
          , hub_(hub)
          , storage_(queue::make_host<variant_t>(this->status_))
          , task_(queue::make_host<task_t>(this->status_, receiver_provider(*this), storage_.get()))
          , inner_op_{
              std::execution::connect((sender_t&&)sender,
              detail::stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>{
                std::execution::get_env(out_receiver), storage_.get(), task_.get(), hub_->producer()})} {
          if (this->status_ == cudaSuccess) {
            this->status_ = task_->status_;
          }
        }

        queue::task_hub_t* hub_;
        queue::host_ptr<variant_t> storage_;
        queue::host_ptr<task_t> task_;
        inner_op_state_t inner_op_;
      };
  }

  template <class S>
    concept stream_completing_sender =
      std::execution::sender<S> &&
      std::is_same_v<
          std::tag_invoke_result_t<
            std::execution::get_completion_scheduler_t<std::execution::set_value_t>, S>,
          stream_scheduler>;

  template <class Sender, class InnerReceiver, class OuterReceiver>
    using stream_op_state_t = detail::operation_state_t<stdexec::__x<Sender>,
                                                        stdexec::__x<InnerReceiver>,
                                                        stdexec::__x<OuterReceiver>>;

  template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>, OuterReceiver>
    stream_op_state(Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
      detail::queue::task_hub_t* hub = std::execution::get_completion_scheduler<std::execution::set_value_t>(sndr).hub_;

      return stream_op_state_t<
        Sender,
        std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>,
        OuterReceiver>(
          (Sender&&)sndr,
          hub,
          (OuterReceiver&&)out_receiver, receiver_provider);
    }

  template <class Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>, OuterReceiver>
    stream_op_state(detail::queue::task_hub_t* hub, Sender&& sndr, OuterReceiver&& out_receiver, ReceiverProvider receiver_provider) {
      return stream_op_state_t<
        Sender,
        std::invoke_result_t<ReceiverProvider, operation_state_base_t<stdexec::__x<OuterReceiver>>&>,
        OuterReceiver>(
          (Sender&&)sndr,
          hub,
          (OuterReceiver&&)out_receiver, receiver_provider);
    }
}

