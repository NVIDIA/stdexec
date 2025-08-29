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

#include <cuda/std/type_traits>
#include <cuda/std/tuple>

#include <optional>
#include <type_traits>
#include <stack>
#include <memory_resource>

#include "../detail/config.cuh"
#include "../detail/cuda_atomic.cuh" // IWYU pragma: keep
#include "../detail/throw_on_cuda_error.cuh"
#include "../detail/queue.cuh"
#include "../detail/variant.cuh"

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_EDG(cuda_compile)

namespace nvexec {
  using stdexec::operator""_mstr;

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
    template <class Tag, class... Env>
    struct transform_sender_for;

    template <class Tag>
    struct apply_sender_for;
  } // namespace _strm
} // namespace nvexec

namespace nvexec {
  struct stream_context;

  // The stream_domain is how the stream scheduler customizes the sender algorithms. All of the
  // algorithms use the current scheduler's domain to transform senders before starting them.
  struct stream_domain : stdexec::default_domain {
    template <stdexec::sender_expr Sender, class Tag = stdexec::tag_of_t<Sender>, class... Env>
      requires stdexec::__callable<
        stdexec::__sexpr_apply_t,
        Sender,
        _strm::transform_sender_for<Tag, Env...>
      >
    static auto transform_sender(Sender&& sndr, const Env&... env) {
      return stdexec::__sexpr_apply(
        static_cast<Sender&&>(sndr), _strm::transform_sender_for<Tag, Env...>{env...});
    }

    template <class Tag, stdexec::sender Sender, class... Args>
      requires stdexec::__callable<_strm::apply_sender_for<Tag>, Sender, Args...>
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

    struct context_state_t {
      std::pmr::memory_resource* pinned_resource_{nullptr};
      std::pmr::memory_resource* managed_resource_{nullptr};
      stream_pools_t* stream_pools_;
      queue::task_hub_t* hub_{nullptr};
      stream_priority priority_;

      context_state_t(
        std::pmr::memory_resource* pinned_resource,
        std::pmr::memory_resource* managed_resource,
        stream_pools_t* stream_pools,
        queue::task_hub_t* hub,
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
    struct multi_gpu_bulk_sender_t;

    template <class Scheduler>
    concept gpu_stream_scheduler = scheduler<Scheduler>
                                && derived_from<__domain_of_t<Scheduler>, stream_domain>
                                && requires(Scheduler sched) {
                                     { sched.context_state_ } -> __decays_to<context_state_t>;
                                   };

    struct stream_sender_base {
      using sender_concept = stdexec::sender_t;
    };

    struct stream_receiver_base {
      using receiver_concept = stdexec::receiver_t;
      static constexpr std::size_t memory_allocation_size = 0;
    };

    struct stream_env_base {
      cudaStream_t stream_;
    };

    template <class T>
    __launch_bounds__(1) __global__ void destructor_kernel(T* obj) {
      obj->~T();
    }

    struct stream_provider_t {
      cudaError_t status_{cudaSuccess};
      std::optional<cudaStream_t> own_stream_{};
      context_state_t context_;

      std::mutex custodian_;
      std::vector<std::function<void()>> cemetery_;

      stream_provider_t(bool borrows_stream, context_state_t context)
        : context_(context) {
        if (!borrows_stream) {
          std::tie(own_stream_, status_) = context_.borrow_stream();
        }
      }

      stream_provider_t(context_state_t context)
        : stream_provider_t(false, context) {
      }

      void bury(std::function<void()> rite) {
        std::lock_guard lock(custodian_);
        cemetery_.emplace_back(rite);
      }

      ~stream_provider_t() {
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

    struct get_stream_provider_t {
      template <class Env>
        requires tag_invocable<get_stream_provider_t, const Env&>
      auto operator()(const Env& env) const noexcept -> stream_provider_t* {
        return tag_invoke(get_stream_provider_t{}, env);
      }

      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto query(stdexec::forwarding_query_t) noexcept -> bool {
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
    using _nullable_variant_t = variant_t<::cuda::std::tuple<set_noop>, Ts...>;

    template <class... Ts>
    using decayed_tuple = ::cuda::std::tuple<__decay_t<Ts>...>;

    template <class _Sender, class _Env>
    using variant_storage_t = __for_each_completion_signature<
      __completion_signatures_of_t<_Sender, _Env>,
      decayed_tuple,
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
        return stdexec::read_env(*this);
      }

      STDEXEC_ATTRIBUTE(host, device)
      static constexpr auto query(stdexec::forwarding_query_t) noexcept -> bool {
        return true;
      }
    };

    template <class Sender>
    struct stream_sender_attrs {
      using __t = stream_sender_attrs;
      using __id = stream_sender_attrs;

      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(get_domain_override_t) const noexcept -> stream_domain {
        return {};
      }

      template <__forwarding_query Query>
        requires __env::__queryable<env_of_t<Sender>, Query>
      STDEXEC_ATTRIBUTE(nodiscard)
      constexpr auto query(Query) const
        noexcept(__env::__nothrow_queryable<env_of_t<Sender>, Query>)
          -> __env::__query_result_t<env_of_t<Sender>, Query> {
        return stdexec::get_env(*child_).query(Query{});
      }

      const Sender* child_{};
    };

    template <class BaseEnv>
    auto make_stream_env(BaseEnv&& base_env, stream_provider_t* stream_provider) noexcept {
      return __env::__join(
        prop{get_stream_provider, stream_provider}, static_cast<BaseEnv&&>(base_env));
    }

    template <class BaseEnv>
      requires __callable<get_stream_provider_t, const BaseEnv&>
    auto make_stream_env(BaseEnv&& base_env, stream_provider_t*) noexcept -> __decay_t<BaseEnv> {
      return static_cast<BaseEnv&&>(base_env);
    }

    template <class BaseEnv>
    using stream_env = decltype(_strm::make_stream_env(
      __declval<BaseEnv>(),
      static_cast<stream_provider_t*>(nullptr)));

    template <class BaseEnv>
    auto make_terminal_stream_env(BaseEnv&& base_env, stream_provider_t* stream_provider) noexcept {
      return __env::__join(
        prop{get_stream_provider, stream_provider}, static_cast<BaseEnv&&>(base_env));
    }
    template <class BaseEnv>
    using terminal_stream_env = decltype(_strm::make_terminal_stream_env(
      __declval<BaseEnv>(),
      static_cast<stream_provider_t*>(nullptr)));

    template <class BaseEnv>
    using make_stream_env_t = stream_env<BaseEnv>;

    template <class BaseEnv>
    using make_terminal_stream_env_t = terminal_stream_env<BaseEnv>;

    template <class S, class E>
    concept stream_sender = sender_in<S, E>
                         && STDEXEC_IS_BASE_OF(
                              stream_sender_base,
                              __decay_t<transform_sender_result_t<__late_domain_of_t<S, E>, S, E>>);

    template <class R>
    concept stream_receiver = receiver<R> && STDEXEC_IS_BASE_OF(stream_receiver_base, __decay_t<R>);

    struct stream_op_state_base { };

    template <class EnvId, class Variant>
    struct stream_enqueue_receiver {
      using Env = stdexec::__cvref_t<EnvId>;

      class __t {
        Env* env_;
        Variant* variant_;
        queue::task_base_t* task_;
        queue::producer_t producer_;

       public:
        using receiver_concept = stdexec::receiver_t;
        using __id = stream_enqueue_receiver;

        template <class... As>
        STDEXEC_ATTRIBUTE(host, device)
        void set_value(As&&... as) noexcept {
          variant_->template emplace<decayed_tuple<set_value_t, As...>>(
            set_value_t(), static_cast<As&&>(as)...);
          producer_(task_);
        }

        STDEXEC_ATTRIBUTE(host, device) void set_stopped() noexcept {
          variant_->template emplace<decayed_tuple<set_stopped_t>>(set_stopped_t());
          producer_(task_);
        }

        template <class Error>
        STDEXEC_ATTRIBUTE(host, device)
        void set_error(Error&& err) noexcept {
          if constexpr (__decays_to<Error, std::exception_ptr>) {
            // What is `exception_ptr` but death pending
            variant_->template emplace<decayed_tuple<set_error_t, cudaError_t>>(
              stdexec::set_error, cudaErrorUnknown);
          } else {
            variant_->template emplace<decayed_tuple<set_error_t, Error>>(
              set_error_t(), static_cast<Error&&>(err));
          }
          producer_(task_);
        }

        auto get_env() const noexcept -> const Env& {
          return *env_;
        }

        __t(Env* env, Variant* variant, queue::task_base_t* task, queue::producer_t producer)
          : env_(env)
          , variant_(variant)
          , task_(task)
          , producer_(producer) {
        }
      };
    };

    template <class Receiver, class... As, class Tag>
    __launch_bounds__(1) __global__ void continuation_kernel(Receiver rcvr, Tag, As... as) {
      static_assert(trivially_copyable<Receiver, Tag, As...>);
      Tag()(::cuda::std::move(rcvr), static_cast<As&&>(as)...);
    }

    template <class Receiver, class Variant>
    struct continuation_task_t : queue::task_base_t {
      Receiver rcvr_;
      Variant* variant_;
      cudaStream_t stream_{};
      std::pmr::memory_resource* pinned_resource_{};
      cudaError_t status_{cudaSuccess};

      continuation_task_t(
        Receiver rcvr,
        Variant* variant,
        cudaStream_t stream,
        std::pmr::memory_resource* pinned_resource) noexcept
        : rcvr_{rcvr}
        , variant_{variant}
        , stream_{stream}
        , pinned_resource_(pinned_resource) {
        this->execute_ = [](task_base_t* t) noexcept {
          continuation_task_t& self = *static_cast<continuation_task_t*>(t);

          visit(
            [&self](auto& tpl) noexcept {
              ::cuda::std::apply(
                [&self]<class Tag, class... As>(Tag, As&... as) noexcept {
                  Tag()(std::move(self.rcvr_), std::move(as)...);
                },
                tpl);
            },
            *self.variant_);
        };

        this->free_ = [](task_base_t* t) noexcept {
          continuation_task_t& self = *static_cast<continuation_task_t*>(t);
          STDEXEC_ASSERT_CUDA_API(cudaFreeAsync(self.atom_next_, self.stream_));
          self.pinned_resource_
            ->deallocate(t, sizeof(continuation_task_t), std::alignment_of_v<continuation_task_t>);
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

    template <class OuterReceiverId>
    struct operation_state_base_ {
      using outer_receiver_t = stdexec::__t<OuterReceiverId>;
      using outer_env_t = env_of_t<outer_receiver_t>;
      static constexpr bool borrows_stream = borrows_stream_h<outer_env_t>();

      struct __t : stream_op_state_base {
        using __id = operation_state_base_;
        using env_t = make_stream_env_t<outer_env_t>;

        context_state_t context_state_;
        void* temp_storage_{nullptr};
        outer_receiver_t rcvr_;
        stream_provider_t stream_provider_;

        __t(outer_receiver_t rcvr, context_state_t context_state)
          : context_state_(context_state)
          , rcvr_(rcvr)
          , stream_provider_(borrows_stream, context_state) {
        }

        [[nodiscard]]
        auto get_stream_provider() const -> stream_provider_t* {
          stream_provider_t* stream_provider{};

          if constexpr (borrows_stream) {
            const outer_env_t& env = get_env(rcvr_);
            stream_provider = ::nvexec::_strm::get_stream_provider(env);
          } else {
            stream_provider = &const_cast<stream_provider_t&>(stream_provider_);
          }

          return stream_provider;
        }

        [[nodiscard]]
        auto get_stream() const -> cudaStream_t {
          return get_stream_provider()->own_stream_.value();
        }

        template <class T>
        void defer_temp_storage_destruction(T* ptr) {
          STDEXEC_ASSERT(ptr == this->temp_storage_);

          if constexpr (!std::is_trivially_destructible_v<T>) {
            temp_storage_ = nullptr; // defer deallocation to the stream provider
            stream_provider_t* stream_provider = get_stream_provider();
            std::pmr::memory_resource* managed_resource = context_state_.managed_resource_;

            // Stream is destroyed when the last object is buried, so it's safe to use it here
            cudaStream_t stream = stream_provider->own_stream_.value();
            stream_provider->bury([ptr, stream, managed_resource] {
              std::int32_t device_id = cudaInvalidDeviceId;

              cudaMemRangeGetAttribute(
                &device_id, 4, cudaMemRangeAttributeLastPrefetchLocation, ptr, sizeof(T));

              if (cudaCpuDeviceId == device_id) {
                ptr->~T();
              } else {
                destructor_kernel<<<1, 1, 0, stream>>>(ptr);

                // TODO Bury all the memory associated with the stream provider and then
                //      deallocate the memory
                cudaStreamSynchronize(stream);
              }

              managed_resource->deallocate(ptr, sizeof(T));
            });
          }
        }

        auto make_env() const noexcept -> env_t {
          return make_stream_env(get_env(rcvr_), get_stream_provider());
        }

        template <__decays_to<cudaError_t> Error>
        void propagate_completion_signal(set_error_t, Error&& status) noexcept {
          if constexpr (stream_receiver<outer_receiver_t>) {
            stdexec::set_error(static_cast<outer_receiver_t&&>(rcvr_), cudaError_t(status));
          } else {
            // pass a cudaError_t by value:
            continuation_kernel<outer_receiver_t, Error><<<1, 1, 0, get_stream()>>>(
              static_cast<outer_receiver_t&&>(rcvr_), set_error_t(), status);
          }
        }

        template <class Tag, class... As>
        void propagate_completion_signal(Tag, As&&... as) noexcept {
          if constexpr (stream_receiver<outer_receiver_t>) {
            Tag()(static_cast<outer_receiver_t&&>(rcvr_), static_cast<As&&>(as)...);
          } else {
            continuation_kernel<outer_receiver_t, As&&...> // by reference
              <<<1, 1, 0, get_stream()>>>(
                static_cast<outer_receiver_t&&>(rcvr_), Tag(), static_cast<As&&>(as)...);
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

        template <class... _Args>
        void set_value(_Args&&... __args) noexcept {
          operation_state_
            .propagate_completion_signal(set_value_t(), static_cast<_Args&&>(__args)...);
        }

        template <class _Error>
        void set_error(_Error&& __err) noexcept {
          operation_state_.propagate_completion_signal(set_error_t(), static_cast<_Error&&>(__err));
        }

        void set_stopped() noexcept {
          operation_state_.propagate_completion_signal(set_stopped_t());
        }

        auto get_env() const noexcept -> decltype(auto) {
          return operation_state_.make_env();
        }
      };
    };

    template <class CvrefSenderId, class InnerReceiverId, class OuterReceiverId>
    struct operation_state_ {
      struct __t : operation_state_base_t<OuterReceiverId> {
        using __id = operation_state_;
        using sender_t = __cvref_t<CvrefSenderId>;
        using inner_receiver_t = stdexec::__t<InnerReceiverId>;
        using outer_receiver_t = stdexec::__t<OuterReceiverId>;
        using typename operation_state_base_t<OuterReceiverId>::env_t;
        using variant_t = variant_storage_t<sender_t, env_t>;

        using base_t = operation_state_base_t<OuterReceiverId>;

        using task_t = continuation_task_t<inner_receiver_t, variant_t>;
        using stream_enqueue_receiver_t =
          stdexec::__t<stream_enqueue_receiver<stdexec::__cvref_id<env_t>, variant_t>>;
        using intermediate_receiver =
          __if_c<stream_sender<sender_t, env_t>, inner_receiver_t, stream_enqueue_receiver_t>;
        using inner_op_state_t = connect_result_t<sender_t, intermediate_receiver>;

        void start() & noexcept {
          started_.test_and_set(::cuda::std::memory_order::relaxed);

          if (this->stream_provider_.status_ != cudaSuccess) {
            // Couldn't allocate memory for operation state, complete with error
            this->propagate_completion_signal(
              stdexec::set_error, std::move(this->stream_provider_.status_));
            return;
          }

          if constexpr (stream_receiver<inner_receiver_t>) {
            if (inner_receiver_t::memory_allocation_size) {
              STDEXEC_TRY {
                this->temp_storage_ = this->context_state_.managed_resource_
                                        ->allocate(inner_receiver_t::memory_allocation_size);
              }
              STDEXEC_CATCH_ALL {
                this->propagate_completion_signal(stdexec::set_error, cudaErrorMemoryAllocation);
                return;
              }
            }
          }

          stdexec::start(inner_op_);
        }

        template <__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
          requires stream_sender<sender_t, env_t>
        __t(
          sender_t&& sender,
          OutR&& out_receiver,
          ReceiverProvider receiver_provider,
          context_state_t context_state)
          : base_t(static_cast<outer_receiver_t&&>(out_receiver), context_state)
          , inner_op_{connect(
              static_cast<sender_t&&>(sender),
              receiver_provider(static_cast<base_t&>(*this)))} {
        }

        template <__decays_to<outer_receiver_t> OutR, class ReceiverProvider>
        __t(
          sender_t&& sender,
          OutR&& out_receiver,
          ReceiverProvider receiver_provider,
          context_state_t context_state)
          : base_t(static_cast<outer_receiver_t&&>(out_receiver), context_state)
          , storage_(
              make_host<variant_t>(this->stream_provider_.status_, context_state.pinned_resource_))
          , task_(
              make_host<task_t>(
                this->stream_provider_.status_,
                context_state.pinned_resource_,
                receiver_provider(*this),
                storage_.get(),
                this->get_stream(),
                context_state.pinned_resource_)
                .release())
          , env_(
              make_host<env_t>(
                this->stream_provider_.status_,
                context_state.pinned_resource_,
                this->make_env()))
          , inner_op_{connect(
              static_cast<sender_t&&>(sender),
              stream_enqueue_receiver_t{
                env_.get(),
                storage_.get(),
                task_,
                context_state.hub_->producer()})} {
          if (this->stream_provider_.status_ == cudaSuccess) {
            this->stream_provider_.status_ = task_->status_;
          }
        }

        ~__t() {
          if (!started_.test(::cuda::memory_order_relaxed)) {
            if (task_) {
              task_->free_(task_);
            }
          }

          if (this->temp_storage_) {
            this->context_state_.managed_resource_
              ->deallocate(this->temp_storage_, inner_receiver_t::memory_allocation_size);
            this->temp_storage_ = nullptr;
          }
        }

        STDEXEC_IMMOVABLE(__t);

        host_ptr<variant_t> storage_;
        task_t* task_{};
        ::cuda::std::atomic_flag started_{};
        host_ptr<__decay_t<env_t>> env_{};
        inner_op_state_t inner_op_;
      };
    };

    template <class CvrefSenderId, class InnerReceiverId, class OuterReceiverId>
    using operation_state_t =
      stdexec::__t<operation_state_<CvrefSenderId, InnerReceiverId, OuterReceiverId>>;

    template <class CvrefSender, class OuterReceiver>
      requires stream_receiver<OuterReceiver>
    using exit_operation_state_t = operation_state_t<
      __cvref_id<CvrefSender>,
      stdexec::__id<stdexec::__t<propagate_receiver_t<stdexec::__id<OuterReceiver>>>>,
      stdexec::__id<OuterReceiver>
    >;

    template <class Sender, class OuterReceiver>
    auto exit_op_state(Sender&& sndr, OuterReceiver rcvr, context_state_t context_state) noexcept
      -> exit_operation_state_t<Sender, OuterReceiver> {
      using ReceiverId = stdexec::__id<OuterReceiver>;
      return exit_operation_state_t<Sender, OuterReceiver>(
        static_cast<Sender&&>(sndr),
        static_cast<OuterReceiver&&>(rcvr),
        [](operation_state_base_t<ReceiverId>& op)
          -> stdexec::__t<propagate_receiver_t<ReceiverId>> {
          return stdexec::__t<propagate_receiver_t<ReceiverId>>{{}, op};
        },
        context_state);
    }

    template <class S>
    concept stream_completing_sender =
      sender<S>
      && gpu_stream_scheduler<__result_of<get_completion_scheduler<set_value_t>, env_of_t<S>>>;

    template <class R>
    concept receiver_with_stream_env = receiver<R> && requires(const R& rcvr) {
      { get_scheduler(get_env(rcvr)).context_state_ } -> __decays_to<context_state_t>;
    };

    template <class InnerReceiverProvider, class OuterReceiver>
    using inner_receiver_t =
      __call_result_t<InnerReceiverProvider, operation_state_base_t<stdexec::__id<OuterReceiver>>&>;

    template <class CvrefSender, class InnerReceiver, class OuterReceiver>
    using stream_op_state_t = operation_state_t<
      __cvref_id<CvrefSender>,
      stdexec::__id<InnerReceiver>,
      stdexec::__id<OuterReceiver>
    >;

    template <stream_completing_sender Sender, class OuterReceiver, class ReceiverProvider>
    auto stream_op_state(
      Sender&& sndr,
      OuterReceiver&& out_receiver,
      ReceiverProvider receiver_provider)
      -> stream_op_state_t<Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver> {
      auto sch = get_completion_scheduler<set_value_t>(get_env(sndr));
      context_state_t context_state = sch.context_state_;

      return stream_op_state_t<
        Sender,
        inner_receiver_t<ReceiverProvider, OuterReceiver>,
        OuterReceiver
      >(static_cast<Sender&&>(sndr),
        static_cast<OuterReceiver&&>(out_receiver),
        receiver_provider,
        context_state);
    }

    template <class Sender, class OuterReceiver, class ReceiverProvider>
    auto stream_op_state(
      Sender&& sndr,
      OuterReceiver&& out_receiver,
      ReceiverProvider receiver_provider,
      context_state_t context_state)
      -> stream_op_state_t<Sender, inner_receiver_t<ReceiverProvider, OuterReceiver>, OuterReceiver> {
      return stream_op_state_t<
        Sender,
        inner_receiver_t<ReceiverProvider, OuterReceiver>,
        OuterReceiver
      >(static_cast<Sender&&>(sndr),
        static_cast<OuterReceiver&&>(out_receiver),
        receiver_provider,
        context_state);
    }
  } // namespace _strm

  inline constexpr _strm::get_stream_t get_stream{};

#if CUDART_VERSION >= 13'00'0
  __host__ inline cudaError_t cudaMemPrefetchAsync(const void *devPtr,
                                                   size_t count, int dstDevice,
                                                   cudaStream_t stream = 0) {
    return ::cudaMemPrefetchAsync(
        devPtr, count, {.type = cudaMemLocationTypeDevice, .id = dstDevice}, 0,
        stream);
  }
#endif

} // namespace nvexec

STDEXEC_PRAGMA_POP()
