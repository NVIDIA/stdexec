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

// clang-format Language: Cpp

#pragma once

#include "../../stdexec/execution.hpp"

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <memory_resource>
#include <optional>
#include <stack>
#include <type_traits>

#include "../detail/config.cuh"
#include "../detail/cuda_atomic.cuh" // IWYU pragma: keep
#include "../detail/queue.cuh"
#include "../detail/throw_on_cuda_error.cuh"
#include "../detail/variant.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec {
  enum class stream_priority {
    high,
    normal,
    low
  };

  enum class device_type {
    host,
    device
  };

#if defined(__clang__) && defined(__CUDA__) && !defined(STDEXEC_CLANG_TIDY_INVOKED)
  __host__ inline auto get_device_type() noexcept -> device_type {
    return device_type::host;
  }

  __device__ inline auto get_device_type() noexcept -> device_type {
    return device_type::device;
  }
#else
  __host__ __device__ inline auto get_device_type() noexcept -> device_type {
    NV_IF_TARGET(NV_IS_HOST, (return device_type::host;), (return device_type::device;));
  }
#endif

  inline STDEXEC_ATTRIBUTE(host, device) auto is_on_gpu() noexcept -> bool {
    return get_device_type() == device_type::device;
  }

  namespace _strm {
    // Used by stream_domain to late-customize senders for execution
    // on the stream_scheduler.
    template <class Tag, class Env>
    struct transform_sender_for;

    template <class Tag>
    struct apply_sender_for;
  } // namespace _strm
} // namespace nvexec

namespace nvexec {
  struct stream_context;

  // The stream_domain is how the stream scheduler customizes the sender algorithms. All of the
  // algorithms use the current scheduler's domain to transform senders before starting them.
  struct stream_domain : STDEXEC::default_domain {
    template <STDEXEC::sender_expr Sender, class Tag = STDEXEC::tag_of_t<Sender>, class Env>
      requires STDEXEC::__applicable<_strm::transform_sender_for<Tag, Env>, Sender>
    static auto transform_sender(STDEXEC::set_value_t, Sender&& sndr, const Env& env) {
      return STDEXEC::__apply(
        _strm::transform_sender_for<Tag, Env>{env}, static_cast<Sender&&>(sndr));
    }

    template <class Tag, STDEXEC::sender Sender, class... Args>
      requires STDEXEC::__callable<_strm::apply_sender_for<Tag>, Sender, Args...>
    static auto apply_sender(Tag, Sender&& sndr, Args&&... args) {
      return _strm::apply_sender_for<Tag>{}(
        static_cast<Sender&&>(sndr), static_cast<Args&&>(args)...);
    }
  };

  namespace _strm {

#if STDEXEC_HAS_BUILTIN(__is_reference)
    template <class... Ts>
    concept trivially_copyable = ((STDEXEC_IS_TRIVIALLY_COPYABLE(Ts) || __is_reference(Ts)) && ...);
#else
    template <class... Ts>
    concept trivially_copyable =
      ((STDEXEC_IS_TRIVIALLY_COPYABLE(Ts) || std::is_reference_v<Ts>) && ...);
#endif

    inline auto get_stream_priority(stream_priority priority) -> std::pair<int, cudaError_t> {
      int least{};
      int greatest{};

      if (cudaError_t status = STDEXEC_LOG_CUDA_API(
            cudaDeviceGetStreamPriorityRange(&least, &greatest));
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

    class stream_pool_t {
      std::stack<cudaStream_t> streams_;
      std::mutex mtx_;

     public:
      stream_pool_t() = default;
      stream_pool_t(const stream_pool_t&) = delete;
      auto operator=(const stream_pool_t&) -> stream_pool_t& = delete;

      auto borrow_stream(stream_priority priority) -> std::pair<cudaStream_t, cudaError_t> {
        std::lock_guard<std::mutex> lock(mtx_);

        if (streams_.empty()) {
          cudaStream_t stream{};
          cudaError_t status{cudaSuccess};

          if (priority == stream_priority::normal) {
            status = STDEXEC_LOG_CUDA_API(cudaStreamCreate(&stream));
          } else {
            int cuda_priority{};
            std::tie(cuda_priority, status) = get_stream_priority(priority);

            if (status != cudaSuccess) {
              return std::make_pair(cudaStream_t{}, status);
            }

            status = STDEXEC_LOG_CUDA_API(
              cudaStreamCreateWithPriority(&stream, cudaStreamDefault, cuda_priority));
          }

          return std::make_pair(stream, status);
        }

        cudaStream_t stream = streams_.top();
        streams_.pop();
        return std::make_pair(stream, cudaSuccess);
      }

      void return_stream(cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        streams_.push(stream);
      }

      ~stream_pool_t() {
        while (!streams_.empty()) {
          cudaStream_t stream = streams_.top();
          streams_.pop();
          cudaStreamDestroy(stream);
        }
      }
    };

    class stream_pools_t {
      std::array<stream_pool_t, 3> pools_;

      auto get(stream_priority priority) -> stream_pool_t& {
        return pools_[static_cast<int>(priority)];
      }

     public:
      auto borrow_stream(stream_priority priority) -> std::pair<cudaStream_t, cudaError_t> {
        return get(priority).borrow_stream(priority);
      }

      void return_stream(cudaStream_t stream, stream_priority priority) {
        get(priority).return_stream(stream);
      }
    };

    struct context {
      std::pmr::memory_resource* pinned_resource_{nullptr};
      std::pmr::memory_resource* managed_resource_{nullptr};
      stream_pools_t* stream_pools_;
      queue::task_hub* hub_{nullptr};
      stream_priority priority_;

      context(
        std::pmr::memory_resource* pinned_resource,
        std::pmr::memory_resource* managed_resource,
        stream_pools_t* stream_pools,
        queue::task_hub* hub,
        stream_priority priority = stream_priority::normal) noexcept
        : pinned_resource_(pinned_resource)
        , managed_resource_(managed_resource)
        , stream_pools_(stream_pools)
        , hub_(hub)
        , priority_(priority) {
      }

      auto borrow_stream() -> std::pair<cudaStream_t, cudaError_t> {
        return stream_pools_->borrow_stream(priority_);
      }

      void return_stream(cudaStream_t stream) {
        stream_pools_->return_stream(stream, priority_);
      }
    };

    struct stream_scheduler;
    struct multi_gpu_stream_scheduler;

    template <class Sender, class Shape, class Fn>
    struct multi_gpu_bulk_sender;

    template <class Scheduler, class Env>
    concept gpu_stream_scheduler =
      scheduler<Scheduler>
      && __std::derived_from<
        __result_of<get_completion_domain<set_value_t>, Scheduler, Env>,
        stream_domain
      >
      && requires(Scheduler sched) {
           { sched.ctx_ } -> __decays_to<context>;
         };

    struct stream_sender_base {
      using sender_concept = STDEXEC::sender_t;
    };

    struct stream_receiver_base {
      using receiver_concept = STDEXEC::receiver_t;
      static constexpr std::size_t memory_allocation_size() noexcept {
        return 0;
      }
    };

    struct stream_env_base {
      cudaStream_t stream_;
    };

    template <class Type>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void destructor_kernel(Type* obj) {
      obj->~Type();
    }

    struct stream_provider {
      cudaError_t status_{cudaSuccess};
      std::optional<cudaStream_t> own_stream_{};
      context context_;

      std::mutex custodian_;
      std::vector<std::function<void()>> cemetery_;

      stream_provider(bool borrows_stream, context context)
        : context_(context) {
        if (!borrows_stream) {
          std::tie(own_stream_, status_) = context_.borrow_stream();
        }
      }

      stream_provider(context context)
        : stream_provider(false, context) {
      }

      void bury(std::function<void()> rite) {
        std::lock_guard lock(custodian_);
        cemetery_.emplace_back(rite);
      }

      ~stream_provider() {
        if (own_stream_) {
          cudaStream_t stream = *own_stream_;

          if (!cemetery_.empty()) {
            for (auto& f: cemetery_) {
              f();
            }
            cemetery_.clear();
          }

          context_.return_stream(stream);
          own_stream_.reset();
        }

        STDEXEC_ASSERT(cemetery_.empty());
      }
    };

    struct get_stream_provider_t : STDEXEC::__query<get_stream_provider_t> {
      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto query(STDEXEC::forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    struct set_noop {
      template <class... Ts>
      STDEXEC_ATTRIBUTE(host, device)
      void operator()(Ts&&...) const noexcept {
        // TODO TRAP
        std::printf("ERROR: use of empty variant.");
      }
    };

    template <class... Ts>
    // using _nullable_variant_t = STDEXEC::__variant<::cuda::std::tuple<set_noop>, Ts...>;
    using _nullable_variant_t = variant_t<::cuda::std::tuple<set_noop>, Ts...>;

    template <class... Ts>
    using decayed_tuple_t = ::cuda::std::tuple<__decay_t<Ts>...>;

    template <class Sender, class Env>
    using variant_storage_t = __for_each_completion_signature_t<
      __completion_signatures_of_t<Sender, Env>,
      decayed_tuple_t,
      __munique<__q<_nullable_variant_t>>::__f
    >;

    inline constexpr get_stream_provider_t get_stream_provider{};

    struct get_stream_t {
      template <class Env>
        requires __callable<get_stream_provider_t, const Env&>
      auto operator()(const Env& env) const noexcept -> cudaStream_t {
        return get_stream_provider(env)->own_stream_.value();
      }

      STDEXEC_ATTRIBUTE(host, device) auto operator()() const noexcept {
        return STDEXEC::read_env(*this);
      }

      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto query(STDEXEC::forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    template <class Sender>
    struct stream_sender_attrs {
      template <__forwarding_query Query>
        requires __queryable_with<env_of_t<Sender>, Query>
      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(Query) const noexcept(__nothrow_queryable_with<env_of_t<Sender>, Query>)
        -> __query_result_t<env_of_t<Sender>, Query> {
        return STDEXEC::__query<Query>()(STDEXEC::get_env(*child_));
      }

      const Sender* child_{};
    };

    template <class BaseEnv>
    auto make_stream_env(BaseEnv&& base_env, stream_provider* stream_provider) noexcept {
      return __env::__join(
        prop{get_stream_provider, stream_provider}, static_cast<BaseEnv&&>(base_env));
    }

    template <class BaseEnv>
      requires __callable<get_stream_provider_t, const BaseEnv&>
    auto make_stream_env(BaseEnv&& base_env, stream_provider*) noexcept -> __decay_t<BaseEnv> {
      return static_cast<BaseEnv&&>(base_env);
    }

    template <class BaseEnv>
    using stream_env_t = decltype(_strm::make_stream_env(
      __declval<BaseEnv>(),
      static_cast<stream_provider*>(nullptr)));

    template <class BaseEnv>
    auto make_terminal_stream_env(BaseEnv&& base_env, stream_provider* stream_provider) noexcept {
      return __env::__join(
        prop{get_stream_provider, stream_provider}, static_cast<BaseEnv&&>(base_env));
    }
    template <class BaseEnv>
    using terminal_stream_env_t = decltype(_strm::make_terminal_stream_env(
      __declval<BaseEnv>(),
      static_cast<stream_provider*>(nullptr)));

    template <class BaseEnv>
    using make_stream_env_t = stream_env_t<BaseEnv>;

    template <class BaseEnv>
    using make_terminal_stream_env_t = terminal_stream_env_t<BaseEnv>;

    template <class Sender, class E>
    concept stream_sender = sender_in<Sender, E>
                         && STDEXEC_IS_BASE_OF(
                              stream_sender_base,
                              STDEXEC_REMOVE_REFERENCE(
                                STDEXEC::transform_sender_result_t<Sender, E>));

    template <class Receiver>
    concept stream_receiver =
      receiver<Receiver>
      && STDEXEC_IS_BASE_OF(stream_receiver_base, STDEXEC_REMOVE_REFERENCE(Receiver));

    struct stream_opstate_base { };

    template <class Env, class Variant>
    struct stream_enqueue_receiver {
      Env* env_;
      Variant* variant_;
      queue::task_base* task_;
      queue::producer producer_;

     public:
      using receiver_concept = STDEXEC::receiver_t;

      template <class... Args>
      STDEXEC_ATTRIBUTE(host, device)
      void set_value(Args&&... args) noexcept {
        variant_->template emplace<decayed_tuple_t<set_value_t, Args...>>(
          set_value_t(), static_cast<Args&&>(args)...);
        producer_(task_);
      }

      STDEXEC_ATTRIBUTE(host, device) void set_stopped() noexcept {
        variant_->template emplace<decayed_tuple_t<set_stopped_t>>(set_stopped_t());
        producer_(task_);
      }

      template <class Error>
      STDEXEC_ATTRIBUTE(host, device)
      void set_error(Error&& err) noexcept {
        if constexpr (__decays_to<Error, std::exception_ptr>) {
          // What is `exception_ptr` but death pending
          variant_->template emplace<decayed_tuple_t<set_error_t, cudaError_t>>(
            STDEXEC::set_error, cudaErrorUnknown);
        } else {
          variant_->template emplace<decayed_tuple_t<set_error_t, Error>>(
            set_error_t(), static_cast<Error&&>(err));
        }
        producer_(task_);
      }

      auto get_env() const noexcept -> const Env& {
        return *env_;
      }

      stream_enqueue_receiver(
        Env* env,
        Variant* variant,
        queue::task_base* task,
        queue::producer producer)
        : env_(env)
        , variant_(variant)
        , task_(task)
        , producer_(producer) {
      }
    };

    template <class Receiver, class... Args, class Tag>
    STDEXEC_ATTRIBUTE(launch_bounds(1))
    __global__ void continuation_kernel(Receiver rcvr, Tag, Args... args) {
      static_assert(trivially_copyable<Receiver, Tag, Args...>);
      Tag()(::cuda::std::move(rcvr), static_cast<Args&&>(args)...);
    }

    template <class Receiver, class Variant>
    struct continuation_task : queue::task_base {
      Receiver rcvr_;
      Variant* variant_;
      cudaStream_t stream_{};
      std::pmr::memory_resource* pinned_resource_{};
      cudaError_t status_{cudaSuccess};

      continuation_task(
        Receiver rcvr,
        Variant* variant,
        cudaStream_t stream,
        std::pmr::memory_resource* pinned_resource) noexcept
        : rcvr_{rcvr}
        , variant_{variant}
        , stream_{stream}
        , pinned_resource_(pinned_resource) {
        this->execute_ = [](task_base* task) noexcept {
          continuation_task& self = *static_cast<continuation_task*>(task);

          nvexec::visit(
            [&self](auto& tpl) noexcept {
              ::cuda::std::apply(
                [&self]<class Tag, class... Args>(Tag, Args&... args) noexcept {
                  Tag()(std::move(self.rcvr_), std::move(args)...);
                },
                tpl);
            },
            *self.variant_);
        };

        this->free_ = [](task_base* task) noexcept {
          continuation_task& self = *static_cast<continuation_task*>(task);
          STDEXEC_ASSERT_CUDA_API(cudaFreeAsync(self.atom_next_, self.stream_));
          self.pinned_resource_
            ->deallocate(task, sizeof(continuation_task), std::alignment_of_v<continuation_task>);
        };

        this->next_ = nullptr;

        constexpr std::size_t ptr_size = sizeof(this->atom_next_);
        status_ = STDEXEC_LOG_CUDA_API(cudaMallocAsync(&this->atom_next_, ptr_size, stream_));

        if (status_ == cudaSuccess) {
          status_ = STDEXEC_LOG_CUDA_API(cudaMemsetAsync(this->atom_next_, 0, ptr_size, stream_));
        }
      }
    };

    template <class Env>
    constexpr auto borrows_stream_h() -> bool {
      return __callable<get_stream_provider_t, const Env&>;
    }

    template <class OuterReceiver>
    struct opstate_base : stream_opstate_base {
      using operation_state_concept = STDEXEC::operation_state_t;
      using outer_env_t = env_of_t<OuterReceiver>;
      using env_t = make_stream_env_t<outer_env_t>;

      static constexpr bool borrows_stream = borrows_stream_h<outer_env_t>();

      opstate_base(OuterReceiver rcvr, context ctx)
        : ctx_(ctx)
        , rcvr_(rcvr)
        , stream_provider_(borrows_stream, ctx) {
      }

      [[nodiscard]]
      auto get_stream_provider() const -> stream_provider* {
        stream_provider* provider{};

        if constexpr (borrows_stream) {
          const outer_env_t& env = get_env(rcvr_);
          provider = ::nvexec::_strm::get_stream_provider(env);
        } else {
          provider = &const_cast<stream_provider&>(stream_provider_);
        }

        return provider;
      }

      [[nodiscard]]
      auto get_stream() const -> cudaStream_t {
        return get_stream_provider()->own_stream_.value();
      }

      template <class Type>
      void defer_temp_storage_destruction(Type* ptr) {
        STDEXEC_ASSERT(ptr == this->temp_storage_);

        if constexpr (!std::is_trivially_destructible_v<Type>) {
          temp_storage_ = nullptr; // defer deallocation to the stream provider
          stream_provider* stream_provider = get_stream_provider();
          std::pmr::memory_resource* managed_resource = ctx_.managed_resource_;

          // Stream is destroyed when the last object is buried, so it's safe to use it here
          cudaStream_t stream = stream_provider->own_stream_.value();
          stream_provider->bury([ptr, stream, managed_resource] {
            std::int32_t device_id = cudaInvalidDeviceId;

            cudaMemRangeGetAttribute(
              &device_id, 4, cudaMemRangeAttributeLastPrefetchLocation, ptr, sizeof(Type));

            if (cudaCpuDeviceId == device_id) {
              ptr->~Type();
            } else {
              destructor_kernel<<<1, 1, 0, stream>>>(ptr);

              // TODO Bury all the memory associated with the stream provider and then
              //      deallocate the memory
              cudaStreamSynchronize(stream);
            }

            managed_resource->deallocate(ptr, sizeof(Type));
          });
        }
      }

      auto make_env() const noexcept -> env_t {
        return make_stream_env(get_env(rcvr_), get_stream_provider());
      }

      template <__decays_to<cudaError_t> Error>
      void propagate_completion_signal(set_error_t, Error&& status) noexcept {
        if constexpr (stream_receiver<OuterReceiver>) {
          STDEXEC::set_error(static_cast<OuterReceiver&&>(rcvr_), cudaError_t(status));
        } else {
          // pass a cudaError_t by value:
          continuation_kernel<OuterReceiver, Error>
            <<<1, 1, 0, get_stream()>>>(static_cast<OuterReceiver&&>(rcvr_), set_error_t(), status);
        }
      }

      template <class Tag, class... Args>
      void propagate_completion_signal(Tag, Args&&... args) noexcept {
        if constexpr (stream_receiver<OuterReceiver>) {
          Tag()(static_cast<OuterReceiver&&>(rcvr_), static_cast<Args&&>(args)...);
        } else {
          continuation_kernel<OuterReceiver, Args&&...> // by reference
            <<<1, 1, 0, get_stream()>>>(
              static_cast<OuterReceiver&&>(rcvr_), Tag(), static_cast<Args&&>(args)...);
        }
      }

      context ctx_;
      void* temp_storage_{nullptr};
      OuterReceiver rcvr_;
      stream_provider stream_provider_;
    };

    template <class OuterReceiver>
    struct propagate_receiver : stream_receiver_base {
      opstate_base<OuterReceiver>& opstate_;

      template <class... Args>
      void set_value(Args&&... args) noexcept {
        opstate_.propagate_completion_signal(set_value_t(), static_cast<Args&&>(args)...);
      }

      template <class Error>
      void set_error(Error&& __err) noexcept {
        opstate_.propagate_completion_signal(set_error_t(), static_cast<Error&&>(__err));
      }

      void set_stopped() noexcept {
        opstate_.propagate_completion_signal(set_stopped_t());
      }

      auto get_env() const noexcept -> decltype(auto) {
        return opstate_.make_env();
      }
    };

    template <class CvSender, class InnerReceiver, class OuterReceiver>
    struct opstate : opstate_base<OuterReceiver> {
      using typename opstate_base<OuterReceiver>::env_t;
      using variant_t = variant_storage_t<CvSender, env_t>;

      using base_t = opstate_base<OuterReceiver>;

      using task_t = continuation_task<InnerReceiver, variant_t>;
      using stream_enqueue_receiver_t = stream_enqueue_receiver<env_t, variant_t>;
      using intermediate_receiver_t =
        __if_c<stream_sender<CvSender, env_t>, InnerReceiver, stream_enqueue_receiver_t>;
      using inner_opstate_t = connect_result_t<CvSender, intermediate_receiver_t>;

      void start() & noexcept {
        started_.test_and_set(::cuda::std::memory_order::relaxed);

        if (this->stream_provider_.status_ != cudaSuccess) {
          // Couldn't allocate memory for opstate state, complete with error
          this->propagate_completion_signal(
            STDEXEC::set_error, std::move(this->stream_provider_.status_));
          return;
        }

        if constexpr (stream_receiver<InnerReceiver>) {
          if (InnerReceiver::memory_allocation_size()) {
            STDEXEC_TRY {
              this->temp_storage_ = this->ctx_.managed_resource_
                                      ->allocate(InnerReceiver::memory_allocation_size());
            }
            STDEXEC_CATCH_ALL {
              this->propagate_completion_signal(STDEXEC::set_error, cudaErrorMemoryAllocation);
              return;
            }
          }
        }

        STDEXEC::start(inner_op_);
      }

      template <class ReceiverProvider>
        requires stream_sender<CvSender, env_t>
      opstate(
        CvSender&& sender,
        OuterReceiver out_receiver,
        ReceiverProvider receiver_provider,
        context ctx)
        : base_t(static_cast<OuterReceiver&&>(out_receiver), ctx)
        , inner_op_{connect(
            static_cast<CvSender&&>(sender),
            receiver_provider(static_cast<base_t&>(*this)))} {
      }

      template <class ReceiverProvider>
      opstate(
        CvSender&& sender,
        OuterReceiver out_receiver,
        ReceiverProvider receiver_provider,
        context ctx)
        : base_t(static_cast<OuterReceiver&&>(out_receiver), ctx)
        , storage_(host_allocate<variant_t>(this->stream_provider_.status_, ctx.pinned_resource_))
        , task_(
            host_allocate<task_t>(
              this->stream_provider_.status_,
              ctx.pinned_resource_,
              receiver_provider(*this),
              storage_.get(),
              this->get_stream(),
              ctx.pinned_resource_)
              .release())
        , env_(
            host_allocate<env_t>(
              this->stream_provider_.status_,
              ctx.pinned_resource_,
              this->make_env()))
        , inner_op_{connect(
            static_cast<CvSender&&>(sender),
            stream_enqueue_receiver_t{env_.get(), storage_.get(), task_, ctx.hub_->producer()})} {
        if (this->stream_provider_.status_ == cudaSuccess) {
          this->stream_provider_.status_ = task_->status_;
        }
      }

      ~opstate() {
        if (!started_.test(::cuda::memory_order_relaxed)) {
          if (task_) {
            task_->free_(task_);
          }
        }

        if (this->temp_storage_) {
          this->ctx_.managed_resource_
            ->deallocate(this->temp_storage_, InnerReceiver::memory_allocation_size());
          this->temp_storage_ = nullptr;
        }
      }

      STDEXEC_IMMOVABLE(opstate);

      host_ptr_t<variant_t> storage_;
      task_t* task_{};
      ::cuda::std::atomic_flag started_{};
      host_ptr_t<__decay_t<env_t>> env_{};
      inner_opstate_t inner_op_;
    };

    template <class CvSender, class OuterReceiver>
      requires stream_receiver<OuterReceiver>
    using exit_opstate_t =
      _strm::opstate<CvSender, propagate_receiver<OuterReceiver>, OuterReceiver>;

    template <class Sender, class OuterReceiver>
    auto exit_opstate(Sender&& sndr, OuterReceiver rcvr, context ctx) noexcept
      -> exit_opstate_t<Sender, OuterReceiver> {
      return exit_opstate_t<Sender, OuterReceiver>(
        static_cast<Sender&&>(sndr),
        static_cast<OuterReceiver&&>(rcvr),
        [](opstate_base<OuterReceiver>& op) -> propagate_receiver<OuterReceiver> {
          return propagate_receiver<OuterReceiver>{{}, op};
        },
        ctx);
    }

    template <class Sender, class E>
    concept stream_completing_sender =
      sender<Sender>
      && gpu_stream_scheduler<
        __result_of<get_completion_scheduler<set_value_t>, env_of_t<Sender>, E>,
        E
      >;

    template <class InnerReceiverProvider, class OuterReceiver>
    using inner_receiver_t = __call_result_t<InnerReceiverProvider, opstate_base<OuterReceiver>&>;

    template <class CvSender, class InnerReceiver, class OuterReceiver>
    using stream_opstate_t = _strm::opstate<CvSender, InnerReceiver, OuterReceiver>;

    template <class Sender, class OuterReceiver, class ReceiverProvider>
      requires stream_completing_sender<Sender, env_of_t<OuterReceiver>>
    auto stream_opstate(
      Sender&& sndr,
      OuterReceiver&& out_receiver,
      ReceiverProvider receiver_provider)
      -> stream_opstate_t<Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver> {
      auto sch = get_completion_scheduler<set_value_t>(get_env(sndr), get_env(out_receiver));
      context ctx = sch.ctx_;

      return stream_opstate_t<
        Sender,
        inner_receiver_t<ReceiverProvider, OuterReceiver>,
        OuterReceiver
      >(static_cast<Sender&&>(sndr),
        static_cast<OuterReceiver&&>(out_receiver),
        receiver_provider,
        ctx);
    }

    template <class Sender, class OuterReceiver, class ReceiverProvider>
    auto stream_opstate(
      Sender&& sndr,
      OuterReceiver&& out_receiver,
      ReceiverProvider receiver_provider,
      context ctx)
      -> stream_opstate_t<Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver> {
      return stream_opstate_t<
        Sender,
        inner_receiver_t<ReceiverProvider, OuterReceiver>,
        OuterReceiver
      >(static_cast<Sender&&>(sndr),
        static_cast<OuterReceiver&&>(out_receiver),
        receiver_provider,
        ctx);
    }
  } // namespace _strm

  inline constexpr _strm::get_stream_t get_stream{};

#if CUDART_VERSION >= 13'00'0
  __host__ inline cudaError_t cudaMemPrefetchAsync(
    const void* dev_ptr,
    size_t count,
    int dst_device,
    cudaStream_t stream = 0) {
    return ::cudaMemPrefetchAsync(
      dev_ptr, count, {.type = cudaMemLocationTypeDevice, .id = dst_device}, 0, stream);
  }
#endif

  template <class Ty>
  inline constexpr std::size_t _sizeof_v = sizeof(Ty);

  template <>
  inline constexpr std::size_t _sizeof_v<void> = 0;

  struct maxsize {
    template <class... Sizes>
    using __f = STDEXEC::__msize_t<STDEXEC::__umax({std::size_t(0), Sizes::value...})>;
  };

  struct result_size_for {
    template <class Fun, class... Args>
    using __f = STDEXEC::__msize_t<_sizeof_v<STDEXEC::__call_result_t<Fun, Args...>>>;
  };
} // namespace nvexec

STDEXEC_PRAGMA_POP()
