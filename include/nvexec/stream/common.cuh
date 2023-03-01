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

#include <atomic>
#include <memory_resource>
#include "../../stdexec/execution.hpp"

#include <cuda/std/type_traits>
#include <cuda/std/tuple>
#include <optional>
#include <type_traits>

#include "../detail/config.cuh"
#include "../detail/cuda_atomic.cuh"
#include "../detail/cuda_error_handling.cuh"
#include "../detail/queue.cuh"
#include "../detail/variant.cuh"

namespace nvexec {

  struct unbound {};

  template <typename T>
    constexpr bool is_bound = !std::same_as<T, unbound>;

  enum class stream_priority {
    high,
    normal,
    low
  };

  enum class device_type {
    host,
    device
  };

#if defined(__clang__) && defined(__CUDA__)
  __host__ inline device_type get_device_type() {
    return device_type::host;
  }

  __device__ inline device_type get_device_type() {
    return device_type::device;
  }
#else
  __host__ __device__ inline device_type get_device_type() {
    NV_IF_TARGET(NV_IS_HOST, (return device_type::host;), (return device_type::device;));
  }
#endif

  inline STDEXEC_DETAIL_CUDACC_HOST_DEVICE bool is_on_gpu() {
    return get_device_type() == device_type::device;
  }
}

namespace nvexec {
  struct stream_context;

  namespace STDEXEC_STREAM_DETAIL_NS {
    struct context_state_t {
      std::pmr::memory_resource* pinned_resource_{nullptr};
      std::pmr::memory_resource* managed_resource_{nullptr};
      queue::task_hub_t* hub_{nullptr};
      stream_priority priority_;

      context_state_t(
        std::pmr::memory_resource* pinned_resource,
        std::pmr::memory_resource* managed_resource,
        queue::task_hub_t* hub,
        stream_priority priority = stream_priority::normal)
        : pinned_resource_(pinned_resource)
        , managed_resource_(managed_resource)
        , hub_(hub)
        , priority_(priority) {
      }
    };

    inline std::pair<int, cudaError_t> get_stream_priority(stream_priority priority) {
      int least{};
      int greatest{};

      if (cudaError_t status = STDEXEC_CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&least, &greatest));
          status != cudaSuccess) {
        return std::make_pair(0, status);
      }

      if (priority == stream_priority::low) {
        return std::make_pair(least, cudaSuccess);
      } else if (priority == stream_priority::high) {
        return std::make_pair(greatest, cudaSuccess);
      }

      return std::make_pair(0, cudaSuccess);
    }

    inline std::pair<cudaStream_t, cudaError_t>
      create_stream_with_priority(stream_priority priority) {
      cudaStream_t stream{};
      cudaError_t status{cudaSuccess};

      if (priority == stream_priority::normal) {
        status = STDEXEC_CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
      } else {
        int cuda_priority{};
        std::tie(cuda_priority, status) = get_stream_priority(priority);

        if (status != cudaSuccess) {
          return std::make_pair(cudaStream_t{}, status);
        }

        status = STDEXEC_CHECK_CUDA_ERROR(
          cudaStreamCreateWithPriority(&stream, cudaStreamDefault, cuda_priority));
      }

      return std::make_pair(stream, status);
    }

    struct stream_scheduler;

    struct stream_sender_base {
      using is_sender = void;
    };

    struct stream_receiver_base {
      constexpr static std::size_t memory_allocation_size = 0;
    };

    struct stream_env_base {
      cudaStream_t stream_;
    };

    struct get_stream_t {
      template <class Env>
        requires stdexec::tag_invocable<get_stream_t, Env>
      cudaStream_t operator()(const Env& env) const noexcept {
        return stdexec::tag_invoke(get_stream_t{}, env);
      }
    };

    template <class... Ts>
    using decayed_tuple = ::cuda::std::tuple<std::decay_t<Ts>...>;

    namespace stream_storage_impl {
      template <class... _Ts>
      using variant = //
        stdexec::__minvoke<
          stdexec::__if_c<
            sizeof...(_Ts) != 0,
            stdexec::
              __transform<stdexec::__q<std::decay_t>, stdexec::__munique<stdexec::__q<variant_t>>>,
            stdexec::__mconst<stdexec::__not_a_variant>>,
          _Ts...>;

      template <class... _Ts>
      using bind_tuples = //
        stdexec::__mbind_front_q<
          variant,
          ::cuda::std::tuple<stdexec::set_stopped_t>,
          ::cuda::std::tuple<stdexec::set_error_t, cudaError_t>,
          _Ts...>;

      template <class Sender, class Env>
      using bound_values_t = //
        stdexec::__value_types_of_t<
          Sender,
          Env,
          stdexec::__mbind_front_q<decayed_tuple, stdexec::set_value_t>,
          stdexec::__q<bind_tuples>>;
    }

    template <class Sender, class Env>
    using variant_storage_t = //
      stdexec::__error_types_of_t<
        Sender,
        Env,
        stdexec::__transform<
          stdexec::__mbind_front_q<decayed_tuple, stdexec::set_error_t>,
          stream_storage_impl::bound_values_t<Sender, Env>>>;

    inline constexpr get_stream_t get_stream{};

    template <class BaseEnvId>
    struct stream_env : stream_env_base {
      using BaseEnv = stdexec::__t<BaseEnvId>;
      BaseEnv base_env_;

      template <class Tag, stdexec::same_as<stream_env> Self, class... As>
        requires stdexec::__callable<Tag, const BaseEnv&, As...>
      friend auto tag_invoke(Tag tag, const Self& self, As&&... as) noexcept -> stdexec::
        __call_result_if_t<stdexec::same_as<Self, stream_env>, Tag, const BaseEnv&, As...> {
        return ((Tag&&) tag)(self.base_env_, (As&&) as...);
      }

      template <stdexec::same_as<stream_env> Self>
        requires(!stdexec::__callable<get_stream_t, const BaseEnv&>)
      friend cudaStream_t tag_invoke(get_stream_t, const Self& self) noexcept {
        return self.stream_;
      }
    };

    template <class BaseEnvId>
    struct terminal_stream_env : stream_env_base {
      using BaseEnv = stdexec::__t<BaseEnvId>;
      BaseEnv base_env_;

      template < //
        stdexec::__none_of<get_stream_t> Tag,
        stdexec::same_as<terminal_stream_env> Self,
        class... As>
        requires stdexec::__callable<Tag, const BaseEnv&, As...>
      friend auto tag_invoke(Tag tag, const Self& self, As&&... as) noexcept
        -> stdexec::__call_result_t<Tag, const BaseEnv&, As...> {
        return ((Tag&&) tag)(self.base_env_, (As&&) as...);
      }

      template <stdexec::same_as<terminal_stream_env> Self>
      friend cudaStream_t tag_invoke(get_stream_t, const Self& self) noexcept {
        return self.stream_;
      }
    };

    template <class BaseEnv>
    using make_stream_env_t = stream_env<stdexec::__x<BaseEnv>>;

    template <class BaseEnv>
    using make_terminal_stream_env_t = terminal_stream_env<stdexec::__x<BaseEnv>>;

    template <class BaseEnv>
    make_stream_env_t<BaseEnv> make_stream_env(BaseEnv base, cudaStream_t stream) noexcept {
      return make_stream_env_t<BaseEnv>{{stream}, base};
    }

    template <class BaseEnv>
    make_terminal_stream_env_t<BaseEnv>
      make_terminal_stream_env(BaseEnv base, cudaStream_t stream) noexcept {
      return make_terminal_stream_env_t<BaseEnv>{{stream}, base};
    }

    template <class S>
    concept stream_sender = stdexec::sender<S>
                         && std::is_base_of_v<stream_sender_base, std::decay_t<S>>;

    template <class R>
    concept stream_receiver =
      stdexec::receiver<R> && std::is_base_of_v<stream_receiver_base, std::decay_t<R>>;

    struct stream_op_state_base { };

    template <class EnvId, class VariantId>
    struct stream_enqueue_receiver {
      using Env = stdexec::__t<EnvId>;
      using Variant = stdexec::__t<VariantId>;

      class __t {
        Env env_;
        Variant* variant_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

       public:
        using __id = stream_enqueue_receiver;

        template <
          stdexec::__one_of<stdexec::set_value_t, stdexec::set_error_t, stdexec::set_stopped_t> Tag,
          class... As>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void
          tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.variant_->template emplace<decayed_tuple<Tag, As...>>(Tag{}, std::move(as)...);
          self.producer_(self.task_);
        }

        template <stdexec::__decays_to<std::exception_ptr> E>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void
          tag_invoke(stdexec::set_error_t, __t&& self, E&& e) noexcept {
          // What is `exception_ptr` but death pending
          self.variant_->template emplace<decayed_tuple<stdexec::set_error_t, cudaError_t>>(
            stdexec::set_error, cudaErrorUnknown);
          self.producer_(self.task_);
        }

        friend Env tag_invoke(stdexec::get_env_t, const __t& self) {
          return self.env_;
        }

        __t(Env env, Variant* variant, queue::task_base_t* task, queue::producer_t producer)
          : env_(env)
          , variant_(variant)
          , task_(task)
          , producer_(producer) {
        }
      };
    };

    template <class Receiver, class Tag, class... As>
    __launch_bounds__(1) __global__
      void continuation_kernel(Receiver receiver, Tag tag, As&&... as) {
      tag(::cuda::std::move(receiver), (As&&) as...);
    }

    template <class Receiver, class Variant>
    struct continuation_task_t : queue::task_base_t {
      Receiver receiver_;
      Variant* variant_;
      cudaStream_t stream_{};
      std::pmr::memory_resource* pinned_resource_{};
      cudaError_t status_{cudaSuccess};

      continuation_task_t(   //
        Receiver receiver,   //
        Variant* variant,    //
        cudaStream_t stream, //
        std::pmr::memory_resource* pinned_resource) noexcept
        : receiver_{receiver}
        , variant_{variant}
        , stream_{stream}
        , pinned_resource_(pinned_resource) {
        this->execute_ = [](task_base_t* t) noexcept {
          continuation_task_t& self = *static_cast<continuation_task_t*>(t);

          visit(
            [&self](auto&& tpl) noexcept {
              ::cuda::std::apply(
                [&self](auto tag, auto&&... as) noexcept {
                  tag(std::move(self.receiver_), std::move(as)...);
                },
                std::move(tpl));
            },
            std::move(*self.variant_));
        };

        this->free_ = [](task_base_t* t) noexcept {
          continuation_task_t& self = *static_cast<continuation_task_t*>(t);
          STDEXEC_CHECK_CUDA_ERROR(cudaFreeAsync(self.atom_next_, self.stream_));
          STDEXEC_CHECK_CUDA_ERROR(cudaStreamDestroy(self.stream_));
          self.pinned_resource_->deallocate(
            t, sizeof(continuation_task_t), std::alignment_of_v<continuation_task_t>);
        };

        this->next_ = nullptr;

        constexpr std::size_t ptr_size = sizeof(this->atom_next_);
        status_ = STDEXEC_CHECK_CUDA_ERROR(cudaMallocAsync(&this->atom_next_, ptr_size, stream_));

        if (status_ == cudaSuccess) {
          status_ = STDEXEC_CHECK_CUDA_ERROR(cudaMemsetAsync(this->atom_next_, 0, ptr_size, stream_));
        }
      }
    };

    template <class Env>
      requires stdexec::tag_invocable<get_stream_t, const std::decay_t<Env>&>
    constexpr bool borrows_stream_h() {
      return true;
    }

    template <class Env>
      requires(!stdexec::tag_invocable<get_stream_t, const std::decay_t<Env>>)
    constexpr bool borrows_stream_h() {
      return false;
    }

    template <class OuterReceiverId>
    struct operation_state_base_ {
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;
      using outer_env_t = stdexec::env_of_t<outer_receiver_t>;
      static constexpr bool borrows_stream = borrows_stream_h<outer_env_t>();

      struct __t : stream_op_state_base {
        using __id = operation_state_base_;
        using env_t = make_stream_env_t<outer_env_t>;

        context_state_t context_state_;
        void* temp_storage_{nullptr};
        outer_receiver_t receiver_;
        cudaError_t status_{cudaSuccess};
        std::optional<cudaStream_t> own_stream_{};
        bool defer_stream_destruction_{false};

        __t(outer_receiver_t receiver, context_state_t context_state, bool defer_stream_destruction)
          : context_state_(context_state)
          , receiver_(receiver)
          , defer_stream_destruction_(defer_stream_destruction) {
          if constexpr (!borrows_stream) {
            std::tie(own_stream_, status_) = create_stream_with_priority(context_state_.priority_);
          }
        }

        cudaStream_t get_stream() const {
          cudaStream_t stream{};

          if constexpr (borrows_stream) {
            const outer_env_t& env = stdexec::get_env(receiver_);
            stream = ::nvexec::STDEXEC_STREAM_DETAIL_NS::get_stream(env);
          } else {
            stream = *own_stream_;
          }

          return stream;
        }

        env_t make_env() const {
          return make_stream_env(stdexec::get_env(receiver_), get_stream());
        }

        template <class Tag, class... As>
        void propagate_completion_signal(Tag tag, As&&... as) noexcept {
          if constexpr (stream_receiver<outer_receiver_t>) {
            tag((outer_receiver_t&&) receiver_, (As&&) as...);
          } else {
            continuation_kernel<std::decay_t<outer_receiver_t>, Tag, As...>
              <<<1, 1, 0, get_stream()>>>(receiver_, tag, (As&&) as...);
          }
        }

        ~__t() {
          if (own_stream_) {
            if (!defer_stream_destruction_) {
              STDEXEC_CHECK_CUDA_ERROR(cudaStreamDestroy(*own_stream_));
            }
            own_stream_.reset();
          }
        }
      };
    };

    template <class OuterReceiverId>
    using operation_state_base_t = stdexec::__t<operation_state_base_<OuterReceiverId>>;

    template <class OuterReceiverId>
    struct propagate_receiver_t {
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;

      struct __t : stream_receiver_base {
        using __id = propagate_receiver_t;

        operation_state_base_t<OuterReceiverId>& operation_state_;

        template <
          stdexec::__one_of<stdexec::set_value_t, stdexec::set_error_t, stdexec::set_stopped_t> Tag,
          class... As >
        friend void tag_invoke(Tag tag, __t&& self, As&&... as) noexcept {
          self.operation_state_.propagate_completion_signal(tag, (As&&) as...);
        }

        friend make_stream_env_t<stdexec::env_of_t<outer_receiver_t>>
          tag_invoke(stdexec::get_env_t, const __t& self) {
          return self.operation_state_.make_env();
        }
      };
    };

    template <class CvrefSenderId, class InnerReceiverId, class OuterReceiverId>
    struct operation_state_ {
      struct __t : operation_state_base_t<OuterReceiverId> {
        using __id = operation_state_;
        using sender_t = stdexec::__cvref_t<CvrefSenderId>;
        using inner_receiver_t = stdexec::__t<InnerReceiverId>;
        using outer_receiver_t = stdexec::__t<OuterReceiverId>;
        using typename operation_state_base_t<OuterReceiverId>::env_t;
        using variant_t = variant_storage_t<sender_t, env_t>;

        using task_t = continuation_task_t<inner_receiver_t, variant_t>;
        using stream_enqueue_receiver_t =
          stdexec::__t<stream_enqueue_receiver<stdexec::__x<env_t>, stdexec::__x<variant_t>>>;
        using intermediate_receiver =
          stdexec::__if_c<stream_sender<sender_t>, inner_receiver_t, stream_enqueue_receiver_t>;
        using inner_op_state_t = stdexec::connect_result_t<sender_t, intermediate_receiver>;

        friend void tag_invoke(stdexec::start_t, __t& op) noexcept {
          op.started_.test_and_set(::cuda::std::memory_order::relaxed);

          if (op.status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            op.propagate_completion_signal(stdexec::set_error, std::move(op.status_));
            return;
          }

          if constexpr (stream_receiver<inner_receiver_t>) {
            if (inner_receiver_t::memory_allocation_size) {
              try {
                op.temp_storage_ = op.context_state_.managed_resource_->allocate(
                  inner_receiver_t::memory_allocation_size);
              } catch (...) {
                op.propagate_completion_signal(stdexec::set_error, cudaErrorMemoryAllocation);
                return;
              }
            }
          }

          stdexec::start(op.inner_op_);
        }

        template <stdexec::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t>
        __t(
          sender_t&& sender,
          OutR&& out_receiver,
          ReceiverProvider receiver_provider,
          context_state_t context_state)
          : operation_state_base_t<OuterReceiverId>(
            (outer_receiver_t&&) out_receiver,
            context_state,
            false)
          , inner_op_{stdexec::connect((sender_t&&) sender, receiver_provider(*this))} {
        }

        template <stdexec::__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
        __t(
          sender_t&& sender,
          OutR&& out_receiver,
          ReceiverProvider receiver_provider,
          context_state_t context_state)
          : operation_state_base_t<OuterReceiverId>(
            (outer_receiver_t&&) out_receiver,
            context_state,
            true)
          , storage_(queue::make_host<variant_t>(this->status_, context_state.pinned_resource_))
          , task_(queue::make_host<task_t>(
                    this->status_,
                    context_state.pinned_resource_,
                    receiver_provider(*this),
                    storage_.get(),
                    this->get_stream(),
                    context_state.pinned_resource_)
                    .release())
          , inner_op_{stdexec::connect(
              (sender_t&&) sender,
              stream_enqueue_receiver_t{
                this->make_env(),
                storage_.get(),
                task_,
                context_state.hub_->producer()})} {
          if (this->status_ == cudaSuccess) {
            this->status_ = task_->status_;
          }
        }

        ~__t() {
          if (!started_.test(::cuda::memory_order_relaxed)) {
            if (task_) {
              task_->free_(task_);
            }
          }

          if (this->temp_storage_) {
            this->context_state_.managed_resource_->deallocate(
              this->temp_storage_, inner_receiver_t::memory_allocation_size);
          }
        }

        STDEXEC_IMMOVABLE(__t);

        queue::host_ptr<variant_t> storage_;
        task_t* task_{};
        ::cuda::std::atomic_flag started_{};

        inner_op_state_t inner_op_;
      };
    };

    template <class CvrefSenderId, class InnerReceiverId, class OuterReceiverId>
    using operation_state_t =
      stdexec::__t<operation_state_<CvrefSenderId, InnerReceiverId, OuterReceiverId>>;

    template <class CvrefSender, class OuterReceiver>
      requires stream_receiver<OuterReceiver>
    using exit_operation_state_t = //
      operation_state_t<
        stdexec::__cvref_id<CvrefSender, std::remove_cvref_t<CvrefSender>>,
        stdexec::__id<stdexec::__t<propagate_receiver_t<stdexec::__id<OuterReceiver>>>>,
        stdexec::__id<OuterReceiver>>;

    template <class Sender, class OuterReceiver>
    exit_operation_state_t<Sender, OuterReceiver>
      exit_op_state(Sender&& sndr, OuterReceiver&& rcvr, context_state_t context_state) noexcept {
      using ReceiverId = stdexec::__id<OuterReceiver>;
      return exit_operation_state_t<Sender, OuterReceiver>(
        (Sender&&) sndr,
        (OuterReceiver&&) rcvr,
        [](operation_state_base_t<ReceiverId>& op)
          -> stdexec::__t<propagate_receiver_t<ReceiverId>> {
          return stdexec::__t<propagate_receiver_t<ReceiverId>>{{}, op};
        },
        context_state);
    }

    template <class S>
    concept stream_completing_sender = stdexec::sender<S> && requires(const S& sndr) {
      {
        stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr))
          .context_state_
      } -> stdexec::__decays_to<context_state_t>;
    };

    template <class R>
    concept receiver_with_stream_env = stdexec::receiver<R> && requires(const R& rcvr) {
      {
        stdexec::get_scheduler(stdexec::get_env(rcvr)).context_state_
      } -> stdexec::__decays_to<context_state_t>;
    };

    template <class InnerReceiverProvider, class OuterReceiver>
    using inner_receiver_t = //
      stdexec::__call_result_t<
        InnerReceiverProvider,
        operation_state_base_t<stdexec::__id<OuterReceiver>>&>;

    template <class CvrefSender, class InnerReceiver, class OuterReceiver>
    using stream_op_state_t = //
      operation_state_t<
        stdexec::__cvref_id<CvrefSender, std::remove_cvref_t<CvrefSender>>,
        stdexec::__id<InnerReceiver>,
        stdexec::__id<OuterReceiver>>;

    template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver>
      stream_op_state(
        Sender&& sndr,
        OuterReceiver&& out_receiver,
        ReceiverProvider receiver_provider) {
      auto sch = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));
      context_state_t context_state = sch.context_state_;

      return stream_op_state_t<
        Sender,
        inner_receiver_t<ReceiverProvider, OuterReceiver>,
        OuterReceiver>(
        (Sender&&) sndr, (OuterReceiver&&) out_receiver, receiver_provider, context_state);
    }

    template <class Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t< Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver>
      stream_op_state(
        Sender&& sndr,
        OuterReceiver&& out_receiver,
        ReceiverProvider receiver_provider,
        context_state_t context_state) {
      return stream_op_state_t<
        Sender,
        inner_receiver_t<ReceiverProvider, OuterReceiver>,
        OuterReceiver>(
        (Sender&&) sndr, (OuterReceiver&&) out_receiver, receiver_provider, context_state);
    }
  }
}
