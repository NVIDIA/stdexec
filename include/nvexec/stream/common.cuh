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
#include "../detail/throw_on_cuda_error.cuh"
#include "../detail/queue.cuh"
#include "../detail/variant.cuh"
#include "../detail/optional.cuh"

namespace nvexec {
  using stdexec::operator""__csz;

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

#if STDEXEC_HAS_BUILTIN(__is_reference)
    template <class... Ts>
    concept trivially_copyable = ((STDEXEC_IS_TRIVIALLY_COPYABLE(Ts) || __is_reference(Ts)) && ...);
#else
    template <class... Ts>
    concept trivially_copyable =
      ((STDEXEC_IS_TRIVIALLY_COPYABLE(Ts) || std::is_reference_v<Ts>) &&...);
#endif

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

      if (cudaError_t status = STDEXEC_DBG_ERR(cudaDeviceGetStreamPriorityRange(&least, &greatest));
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
        status = STDEXEC_DBG_ERR(cudaStreamCreate(&stream));
      } else {
        int cuda_priority{};
        std::tie(cuda_priority, status) = get_stream_priority(priority);

        if (status != cudaSuccess) {
          return std::make_pair(cudaStream_t{}, status);
        }

        status = STDEXEC_DBG_ERR(
          cudaStreamCreateWithPriority(&stream, cudaStreamDefault, cuda_priority));
      }

      return std::make_pair(stream, status);
    }

    struct stream_scheduler;

    struct stream_sender_base {
      using is_sender = void;
    };

    struct stream_receiver_base {
      using temporary_storage_type = void;
    };

    struct stream_env_base {
      cudaStream_t stream_;
    };

    struct get_stream_t {
      template <class Env>
        requires tag_invocable<get_stream_t, Env>
      cudaStream_t operator()(const Env& env) const noexcept {
        return tag_invoke(get_stream_t{}, env);
      }
    };

    template <class... Ts>
    using decayed_tuple = ::cuda::std::tuple<__decay_t<Ts>...>;

    namespace stream_storage_impl {
      template <class... _Ts>
      using variant = //
        __minvoke<
          __if_c<
            sizeof...(_Ts) != 0,
            __transform< __q<__decay_t>, __munique<__q<variant_t>>>,
            __mconst<__not_a_variant>>,
          _Ts...>;

      template <class _State, class... _Tuples>
      using __make_bind_ = __mbind_back<_State, _Tuples...>;

      template <class _State>
      using __make_bind = __mbind_front_q<__make_bind_, _State>;

      template <class _Tag>
      using __tuple_t = __mbind_front_q<decayed_tuple, _Tag>;

      template <class _Sender, class _Env, class _State, class _Tag>
      using __bind_completions_t =
        __gather_completions_for<_Tag, _Sender, _Env, __tuple_t<_Tag>, __make_bind<_State>>;
    }

    struct set_noop {
      template <class... Ts>
      STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
        void
        operator()(Ts&&...) const noexcept {
        // TODO TRAP
        std::printf("ERROR: use of empty variant.");
      }
    };

    template <class _Sender, class _Env>
    using variant_storage_t = //
      __minvoke< __minvoke<
        __mfold_right<
          __mbind_front_q<stream_storage_impl::variant, ::cuda::std::tuple<set_noop>>,
          __mbind_front_q<stream_storage_impl::__bind_completions_t, _Sender, _Env>>,
        set_value_t,
        set_error_t,
        set_stopped_t>>;

    inline constexpr get_stream_t get_stream{};

    template <class BaseEnv>
    auto make_stream_env(BaseEnv&& base_env, cudaStream_t stream) noexcept {
      return __env::__join_env(
        __env::__env_fn{[stream](get_stream_t) noexcept {
          return stream;
        }},
        (BaseEnv&&) base_env);
    }

    template <class BaseEnv>
      requires __callable<get_stream_t, const BaseEnv&>
    BaseEnv make_stream_env(BaseEnv&& base_env, cudaStream_t) noexcept {
      return (BaseEnv&&) base_env;
    }
    template <class BaseEnv>
    using stream_env =
      decltype(STDEXEC_STREAM_DETAIL_NS::make_stream_env(__declval<BaseEnv>(), cudaStream_t()));

    template <class BaseEnv>
    auto make_terminal_stream_env(BaseEnv&& base_env, cudaStream_t stream) noexcept {
      return __env::__join_env(
        __env::__env_fn{[stream](get_stream_t) noexcept {
          return stream;
        }},
        (BaseEnv&&) base_env);
    }
    template <class BaseEnv>
    using terminal_stream_env = decltype(STDEXEC_STREAM_DETAIL_NS::make_terminal_stream_env(
      __declval<BaseEnv>(),
      cudaStream_t()));

    template <class BaseEnv>
    using make_stream_env_t = stream_env<BaseEnv>;

    template <class BaseEnv>
    using make_terminal_stream_env_t = terminal_stream_env<BaseEnv>;

    template <class S>
    concept stream_sender = //
      sender<S> &&          //
      std::is_base_of_v<stream_sender_base, __decay_t<S>>;

    template <class R>
    concept stream_receiver = //
      receiver<R> &&          //
      std::is_base_of_v<stream_receiver_base, __decay_t<R>>;

    struct stream_op_state_base { };

    template <class EnvId, class Variant>
    struct stream_enqueue_receiver {
      using Env = stdexec::__t<EnvId>;

      class __t {
        Env env_;
        Variant* variant_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

       public:
        using __id = stream_enqueue_receiver;

        template <__one_of<set_value_t, set_stopped_t> Tag, class... As>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void
          tag_invoke(Tag, __t&& self, As&&... as) noexcept {
          self.variant_->template emplace<decayed_tuple<Tag, As...>>(Tag(), std::move(as)...);
          self.producer_(self.task_);
        }

        template <same_as<set_error_t> _Tag, class Error>
        STDEXEC_DETAIL_CUDACC_HOST_DEVICE //
          friend void
          tag_invoke(_Tag, __t&& self, Error&& e) noexcept {
          if constexpr (__decays_to<Error, std::exception_ptr>) {
            // What is `exception_ptr` but death pending
            self.variant_->template emplace<decayed_tuple<set_error_t, cudaError_t>>(
              stdexec::set_error, cudaErrorUnknown);
          } else {
            self.variant_->template emplace<decayed_tuple<set_error_t, Error>>(
              set_error_t{}, std::move(e));
          }
          self.producer_(self.task_);
        }

        friend const Env& tag_invoke(get_env_t, const __t& self) noexcept {
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

    template <class... As, class Receiver, class Tag>
    __launch_bounds__(1) __global__ void continuation_kernel(Receiver receiver, Tag, As... as) {
      Tag()(::cuda::std::move(receiver), static_cast<As&&>(as)...);
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
            [&self](auto& tpl) noexcept {
              ::cuda::std::apply(
                [&self]<class Tag, class... As>(Tag, As&... as) noexcept {
                  Tag()(std::move(self.receiver_), std::move(as)...);
                },
                tpl);
            },
            *self.variant_);
        };

        this->free_ = [](task_base_t* t) noexcept {
          continuation_task_t& self = *static_cast<continuation_task_t*>(t);
          STDEXEC_DBG_ERR(cudaFreeAsync(self.atom_next_, self.stream_));
          STDEXEC_DBG_ERR(cudaStreamDestroy(self.stream_));
          self.pinned_resource_->deallocate(
            t, sizeof(continuation_task_t), std::alignment_of_v<continuation_task_t>);
        };

        this->next_ = nullptr;

        constexpr std::size_t ptr_size = sizeof(this->atom_next_);
        status_ = STDEXEC_DBG_ERR(cudaMallocAsync(&this->atom_next_, ptr_size, stream_));

        if (status_ == cudaSuccess) {
          status_ = STDEXEC_DBG_ERR(cudaMemsetAsync(this->atom_next_, 0, ptr_size, stream_));
        }
      }
    };

    template <class Env>
      requires tag_invocable<get_stream_t, const __decay_t<Env>&>
    constexpr bool borrows_stream_h() {
      return true;
    }

    template <class Env>
      requires(!tag_invocable<get_stream_t, const __decay_t<Env>>)
    constexpr bool borrows_stream_h() {
      return false;
    }

    template <class OuterReceiverId, class TempStorage>
    struct operation_state_base_ {
      struct __t;
    };

    template <class OuterReceiverId>
    struct operation_state_base_<OuterReceiverId, void> {
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;
      using outer_env_t = env_of_t<outer_receiver_t>;
      static constexpr bool borrows_stream = borrows_stream_h<outer_env_t>();

      struct __t;
    };

    template <class OuterReceiverId, class TempStorage>
    struct operation_state_base_<OuterReceiverId, TempStorage>::__t
      : operation_state_base_<OuterReceiverId, void>::__t {
      using operation_state_base_<OuterReceiverId, void>::__t::__t;

      using temp_storage_t = TempStorage;
      host_ptr<optional<temp_storage_t>> temp_storage_;
    };

    template <class OuterReceiverId>
    struct operation_state_base_<OuterReceiverId, void>::__t : stream_op_state_base {
      using __id = operation_state_base_;
      using env_t = make_stream_env_t<outer_env_t>;
      using temp_storage_t = void;

      context_state_t context_state_;
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
          const outer_env_t& env = get_env(receiver_);
          stream = ::nvexec::STDEXEC_STREAM_DETAIL_NS::get_stream(env);
        } else {
          stream = *own_stream_;
        }

        return stream;
      }

      env_t make_env() const noexcept {
        return make_stream_env(get_env(receiver_), get_stream());
      }

      template <class Tag, class... As>
      void propagate_completion_signal(Tag, As&&... as) noexcept {
        if constexpr (stream_receiver<outer_receiver_t>) {
          Tag()((outer_receiver_t&&) receiver_, (As&&) as...);
        } else if constexpr (same_as<Tag, set_error_t>) {
          continuation_kernel<As...> // by value
            <<<1, 1, 0, get_stream()>>>(std::move(receiver_), Tag(), (As&&) as...);
        } else {
          continuation_kernel<As&&...> // by reference
            <<<1, 1, 0, get_stream()>>>(std::move(receiver_), Tag(), (As&&) as...);
        }
      }

      ~__t() {
        if (own_stream_) {
          if (!defer_stream_destruction_) {
            STDEXEC_DBG_ERR(cudaStreamDestroy(*own_stream_));
          }
          own_stream_.reset();
        }
      }
    };

    template <class OuterReceiverId, class TempStorage = void>
    using operation_state_base_t =
      stdexec::__t<operation_state_base_<OuterReceiverId, TempStorage>>;

    template <class OuterReceiverId>
    struct propagate_receiver_t {
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;

      struct __t : stream_receiver_base {
        using __id = propagate_receiver_t;

        operation_state_base_t<OuterReceiverId>& operation_state_;

        template < __completion_tag Tag, class... As >
        friend void tag_invoke(Tag, __t&& self, As&&... as) noexcept {
          self.operation_state_.propagate_completion_signal(Tag(), (As&&) as...);
        }

        friend make_stream_env_t<env_of_t<outer_receiver_t>>
          tag_invoke(get_env_t, const __t& self) noexcept {
          return self.operation_state_.make_env();
        }
      };
    };

    template <class CvrefSenderId, class InnerReceiverId, class OuterReceiverId>
    struct operation_state_ {
      using sender_t = __cvref_t<CvrefSenderId>;
      using inner_receiver_t = stdexec::__t<InnerReceiverId>;
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;
      using env_t = make_stream_env_t<env_of_t<outer_receiver_t>>;
      using variant_t = variant_storage_t<sender_t, env_t>;
      using temp_storage_t = typename inner_receiver_t::temporary_storage_type;

      using base_t = //
        operation_state_base_t< OuterReceiverId, temp_storage_t>;

      using task_t = continuation_task_t<inner_receiver_t, variant_t>;
      using stream_enqueue_receiver_t =
        stdexec::__t<stream_enqueue_receiver<stdexec::__id<env_t>, variant_t>>;
      using intermediate_receiver =
        __if_c<stream_sender<sender_t>, inner_receiver_t, stream_enqueue_receiver_t>;
      using inner_op_state_t = connect_result_t<sender_t, intermediate_receiver>;

      struct __t : base_t {
        using __id = operation_state_;

        friend void tag_invoke(start_t, __t& op) noexcept {
          op.started_.test_and_set(::cuda::std::memory_order::relaxed);

          if (op.status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            op.propagate_completion_signal(stdexec::set_error, std::move(op.status_));
            return;
          }

          if constexpr (stream_receiver<inner_receiver_t>) {
            if constexpr (!same_as<void, temp_storage_t>) {
              // Allocate managed temporary storage
              cudaError_t status = cudaSuccess;
              op.temp_storage_ = host_allocate<optional<temp_storage_t>>(
                status, op.context_state_.managed_resource_);
              if (status != cudaSuccess) {
                op.propagate_completion_signal(stdexec::set_error, (cudaError_t) status);
                return;
              }
            }
          }

          start(op.inner_op_);
        }

        template <__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t>
        __t(
          sender_t&& sender,
          OutR&& out_receiver,
          ReceiverProvider receiver_provider,
          context_state_t context_state)
          : base_t((outer_receiver_t&&) out_receiver, context_state, false)
          , inner_op_{
              connect((sender_t&&) sender, receiver_provider(static_cast<base_t&>(*this)))} {
        }

        template <__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
        __t(
          sender_t&& sender,
          OutR&& out_receiver,
          ReceiverProvider receiver_provider,
          context_state_t context_state)
          : base_t((outer_receiver_t&&) out_receiver, context_state, true)
          , storage_(host_allocate<variant_t>(this->status_, context_state.pinned_resource_))
          , task_(host_allocate<task_t>(
                    this->status_,
                    context_state.pinned_resource_,
                    receiver_provider(*this),
                    storage_.get(),
                    this->get_stream(),
                    context_state.pinned_resource_)
                    .release())
          , inner_op_{connect(
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
        }

        STDEXEC_IMMOVABLE(__t);

        host_ptr<variant_t> storage_;
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
        __cvref_id<CvrefSender>,
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
    concept stream_completing_sender = //
      sender<S> &&                     //
      requires(const S& sndr) {
        {
          get_completion_scheduler<set_value_t>(get_env(sndr)).context_state_
        } -> __decays_to<context_state_t>;
      };

    template <class R>
    concept receiver_with_stream_env = //
      receiver<R> &&                   //
      requires(const R& rcvr) {
        { get_scheduler(get_env(rcvr)).context_state_ } -> __decays_to<context_state_t>;
      };

    template <class InnerReceiverProvider, class OuterReceiver>
    using inner_receiver_t = //
      __call_result_t< InnerReceiverProvider, operation_state_base_t<stdexec::__id<OuterReceiver>>&>;

    template <class CvrefSender, class InnerReceiver, class OuterReceiver>
    using stream_op_state_t = //
      operation_state_t<
        __cvref_id<CvrefSender>,
        stdexec::__id<InnerReceiver>,
        stdexec::__id<OuterReceiver>>;

    template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
    stream_op_state_t<Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver>
      stream_op_state(
        Sender&& sndr,
        OuterReceiver&& out_receiver,
        ReceiverProvider receiver_provider) {
      auto sch = get_completion_scheduler<set_value_t>(get_env(sndr));
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
