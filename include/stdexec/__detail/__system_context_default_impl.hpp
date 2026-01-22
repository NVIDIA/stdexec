/*
 * Copyright (c) 2023 Lee Howes, Lucian Radu Teodorescu
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

#include "__atomic.hpp"
#include "__system_context_replaceability_api.hpp"

#if STDEXEC_ENABLE_LIBDISPATCH
#  include "../../exec/libdispatch_queue.hpp" // IWYU pragma: keep
#elif STDEXEC_ENABLE_IO_URING
#  include "../../exec/linux/io_uring_context.hpp" // IWYU pragma: keep
#elif STDEXEC_ENABLE_WINDOWS_THREAD_POOL
#  include "../../exec/windows/windows_thread_pool.hpp" // IWYU pragma: keep
#else
#  include "../../exec/static_thread_pool.hpp" // IWYU pragma: keep
#endif

namespace STDEXEC::__system_context_default_impl {
  using system_context_replaceability::__parallel_scheduler_backend_factory;

  /// Receiver that calls the callback when the operation completes.
  template <class _Sender>
  struct __operation;

  /*
  Storage needed for a backend operation-state:

  schedule:
  - __recv::__r_ (receiver*) -- 8
  - __recv::__op_ (__operation*) -- 8
  - __operation::__inner_op_ (STDEXEC::connect_result_t<_Sender, __recv<_Sender>>) -- 56 (when connected with an empty receiver)
  - __operation::__on_heap_ (bool) -- optimized away
  ---------------------
  Total: 72; extra 16 bytes compared to internal operation state.

  extra for bulk:
  - __recv::__r_ (receiver*) -- 8
  - __recv::__op_ (__operation*) -- 8
  - __operation::__inner_op_ (STDEXEC::connect_result_t<_Sender, __recv<_Sender>>) -- 128 (when connected with an empty receiver & fun)
  - __operation::__on_heap_ (bool) -- optimized away
  - __bulk_unchunked_functor::__r_ (bulk_item_receiver*) - 8
  ---------------------
  Total: 152; extra 24 bytes compared to internal operation state.

  Using libdispatch backend, the operation sizes are 48 (down from 80) and 128 (down from 160).

  [*] sizes taken on an Apple M2 Pro arm64 arch. They may differ on other architectures, or with different implementations.
  */

  template <class _Sender>
  struct __recv {
    using receiver_concept = STDEXEC::receiver_t;

    //! The operation state on the frontend.
    STDEXEC::system_context_replaceability::receiver_proxy* __r_;

    //! The parent operation state that we will destroy when we complete.
    __operation<_Sender>* __op_;

    void set_value() noexcept {
      auto __op = __op_;
      auto __rcvr = __r_;
      __op->__destruct(); // destroys the operation, including `this`.
      __rcvr->set_value();
      // Note: when calling a completion signal, the parent operation might complete, making the
      // static storage passed to this operation invalid. Thus, we need to ensure that we are not
      // using the operation state after the completion signal.
    }

    void set_error(std::exception_ptr __ptr) noexcept {
      auto __op = __op_;
      auto __rcvr = __r_;
      __op->__destruct(); // destroys the operation, including `this`.
      __rcvr->set_error(std::move(__ptr));
    }

    void set_stopped() noexcept {
      auto __op = __op_;
      auto __rcvr = __r_;
      __op->__destruct(); // destroys the operation, including `this`.
      __rcvr->set_stopped();
    }

    [[nodiscard]]
    auto get_env() const noexcept -> decltype(auto) {
      auto __o = __r_->try_query<STDEXEC::inplace_stop_token>(STDEXEC::get_stop_token);
      STDEXEC::inplace_stop_token __st = __o ? *__o : STDEXEC::inplace_stop_token{};
      return STDEXEC::prop{STDEXEC::get_stop_token, __st};
    }
  };

  /// Ensure that `__storage` is aligned to `__alignment`. Shrinks the storage, if needed, to match desired alignment.
  inline auto __ensure_alignment(std::span<std::byte> __storage, size_t __alignment) noexcept
    -> std::span<std::byte> {
    auto __pn = reinterpret_cast<uintptr_t>(__storage.data());
    if (__pn % __alignment == 0) {
      return __storage;
    } else if (__storage.size() < __alignment) {
      return {};
    } else {
      auto __new_pn = (__pn + __alignment - 1) & ~(__alignment - 1);
      return {
        reinterpret_cast<std::byte*>(__new_pn),
        static_cast<size_t>(__storage.size() - (__new_pn - __pn))};
    }
  }

  template <typename _Sender>
  struct __operation {
    /// The inner operation state, that results out of connecting the underlying sender with the receiver.
    STDEXEC::connect_result_t<_Sender, __recv<_Sender>> __inner_op_;
    /// True if the operation is on the heap, false if it is in the preallocated space.
    bool __on_heap_;

    /// Try to construct the operation in the preallocated memory if it fits, otherwise allocate a new operation.
    static auto __construct_maybe_alloc(
      std::span<std::byte> __storage,
      STDEXEC::system_context_replaceability::receiver_proxy* __completion,
      _Sender __sndr) -> __operation* {
      __storage = __ensure_alignment(__storage, alignof(__operation));
      if (__storage.data() == nullptr || __storage.size() < sizeof(__operation)) {
        return new __operation(std::move(__sndr), __completion, true);
      } else {
        return new (__storage.data()) __operation(std::move(__sndr), __completion, false);
      }
    }

    //! Starts the operation that will schedule work on the system scheduler.
    void start() & noexcept {
      STDEXEC::start(__inner_op_);
    }

    /// Destructs the operation; frees any allocated memory.
    void __destruct() {
      if (__on_heap_) {
        delete this;
      } else {
        std::destroy_at(this);
      }
    }

   private:
    __operation(
      _Sender __sndr,
      STDEXEC::system_context_replaceability::receiver_proxy* __completion,
      bool __on_heap)
      : __inner_op_(STDEXEC::connect(std::move(__sndr), __recv<_Sender>{__completion, this}))
      , __on_heap_(__on_heap) {
    }
  };

  template <typename _T>
  concept __has_available_paralellism = requires(_T __pool) {
    { __pool.available_parallelism() } -> std::integral;
  };

  template <typename _BaseSchedulerContext>
  struct __generic_impl : STDEXEC::system_context_replaceability::parallel_scheduler_backend {
    __generic_impl()
      : __pool_scheduler_(__pool_.get_scheduler()) {
      // If the pool exposes the available parallelism, use it to determine the chunk size.
      if constexpr (__has_available_paralellism<_BaseSchedulerContext>) {
        __available_parallelism_ = static_cast<size_t>(__pool_.available_parallelism());
      } else {
        __available_parallelism_ = std::thread::hardware_concurrency();
      }
    }
   private:
    using __pool_scheduler_t = decltype(std::declval<_BaseSchedulerContext>().get_scheduler());

    //! The underlying thread pool.
    _BaseSchedulerContext __pool_;
    //! The scheduler to use for starting work in our pool.
    __pool_scheduler_t __pool_scheduler_;
    //! The available parallelism of the pool, used to determine the chunk size.
    //! Use a value of 0 to disable chunking.
    size_t __available_parallelism_{};

    //! Helper class that maps from a chunk index to the start and end of the chunk.
    struct __chunker {
      size_t __chunk_size_;
      size_t __max_size_;

      [[nodiscard]]
      size_t __begin(size_t __chunk_index) const noexcept {
        return __chunk_index * __chunk_size_;
      }

      [[nodiscard]]
      size_t __end(size_t __chunk_index) const noexcept {
        return (std::min) (__begin(__chunk_index + 1), __max_size_);
      }
    };

    //! Functor called by the `bulk_chunked` operation; sends a `execute` signal to the frontend.
    struct __bulk_chunked_functor {
      STDEXEC::system_context_replaceability::bulk_item_receiver_proxy* __r_;
      __chunker __chunker_;

      void operator()(size_t const __idx) const noexcept {
        __r_->execute(__chunker_.__begin(__idx), __chunker_.__end(__idx));
      }
    };

    //! Functor called by the `bulk_unchunked` operation; sends a `execute` signal to the frontend.
    struct __bulk_unchunked_functor {
      STDEXEC::system_context_replaceability::bulk_item_receiver_proxy* __r_;

      void operator()(size_t const __idx) const noexcept {
        __r_->execute(__idx, __idx + 1);
      }
    };

    using __schedule_operation_t =
      __operation<decltype(STDEXEC::schedule(std::declval<__pool_scheduler_t>()))>;

    using __schedule_bulk_chunked_operation_t = __operation<decltype(STDEXEC::bulk(
      STDEXEC::schedule(std::declval<__pool_scheduler_t>()),
      STDEXEC::par,
      std::declval<size_t>(),
      std::declval<__bulk_chunked_functor>()))>;

    using __schedule_bulk_unchunked_operation_t = __operation<decltype(STDEXEC::bulk(
      STDEXEC::schedule(std::declval<__pool_scheduler_t>()),
      STDEXEC::par,
      std::declval<size_t>(),
      std::declval<__bulk_unchunked_functor>()))>;

   public:
    void schedule(
      STDEXEC::system_context_replaceability::receiver_proxy& __rcvr,
      std::span<std::byte> __storage) noexcept override {
      STDEXEC_TRY {
        auto __sndr = STDEXEC::schedule(__pool_scheduler_);
        auto __os =
          __schedule_operation_t::__construct_maybe_alloc(__storage, &__rcvr, std::move(__sndr));
        __os->start();
      }
      STDEXEC_CATCH_ALL {
        __rcvr.set_error(std::current_exception());
      }
    }

    void schedule_bulk_chunked(
      size_t __size,
      STDEXEC::system_context_replaceability::bulk_item_receiver_proxy& __rcvr,
      std::span<std::byte> __storage) noexcept override {
      STDEXEC_TRY {
        // Determine the chunking size based on the ratio between the given size and the number of workers in our pool.
        // Aim at having 2 chunks per worker.
        size_t __chunk_size = (__available_parallelism_ > 0
                               && __size > 3ul * __available_parallelism_)
                              ? __size / __available_parallelism_ / 2ul
                              : 1ul;
        size_t __num_chunks = (__size + __chunk_size - 1) / __chunk_size;

        auto __sndr = STDEXEC::bulk(
          STDEXEC::schedule(__pool_scheduler_),
          STDEXEC::par,
          __num_chunks,
          __bulk_chunked_functor{
            &__rcvr, __chunker{__chunk_size, __size}
        });
        auto __os = __schedule_bulk_chunked_operation_t::__construct_maybe_alloc(
          __storage, &__rcvr, std::move(__sndr));
        __os->start();
      }
      STDEXEC_CATCH_ALL {
        __rcvr.set_error(std::current_exception());
      }
    }

    void schedule_bulk_unchunked(
      size_t __size,
      STDEXEC::system_context_replaceability::bulk_item_receiver_proxy& __rcvr,
      std::span<std::byte> __storage) noexcept override {
      STDEXEC_TRY {
        auto __sndr = STDEXEC::bulk(
          STDEXEC::schedule(__pool_scheduler_),
          STDEXEC::par,
          __size,
          __bulk_unchunked_functor{&__rcvr});
        auto __os = __schedule_bulk_unchunked_operation_t::__construct_maybe_alloc(
          __storage, &__rcvr, std::move(__sndr));
        __os->start();
      }
      STDEXEC_CATCH_ALL {
        __rcvr.set_error(std::current_exception());
      }
    }
  };

  /// Keeps track of the backends for the system context interfaces.
  template <typename _Interface, typename _Impl>
  struct __instance_data {
    // work around for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=119652
    constexpr __instance_data() noexcept // NOLINT(modernize-use-equals-default)
    {
    }

    /// Gets the current instance; if there is no instance, uses the current factory to create one.
    auto __get_current_instance() -> std::shared_ptr<_Interface> {
      // If we have a valid instance, return it.
      __acquire_instance_lock();
      auto __rcvr = __instance_;
      __release_instance_lock();
      if (__rcvr) {
        return __rcvr;
      }

      // Otherwise, create a new instance using the factory.
      // Note: we are lazy-loading the instance to avoid creating it if it is not needed.
      auto __new_instance = __factory_.load(STDEXEC::__std::memory_order_relaxed)();

      // Store the newly created instance.
      __acquire_instance_lock();
      __instance_ = __new_instance;
      __release_instance_lock();
      return __new_instance;
    }

    /// Set `__new_factory` as the new factory for `_Interface` and return the old one.
    auto __set_backend_factory(__parallel_scheduler_backend_factory __new_factory)
      -> __parallel_scheduler_backend_factory {
      // Replace the factory, keeping track of the old one.
      auto __old_factory = __factory_.exchange(__new_factory);
      // Create a new instance with the new factory.
      auto __new_instance = __new_factory();
      // Replace the current instance with the new one.
      __acquire_instance_lock();
      auto __old_instance = std::exchange(__instance_, __new_instance);
      __release_instance_lock();
      // Make sure to delete the old instance after releasing the lock.
      __old_instance.reset();
      return __old_factory;
    }

   private:
    STDEXEC::__std::atomic<bool> __instance_locked_{false};
    std::shared_ptr<_Interface> __instance_{nullptr};
    STDEXEC::__std::atomic<__parallel_scheduler_backend_factory> __factory_{__default_factory};

    /// The default factory returns an instance of `_Impl`.
    static auto __default_factory() -> std::shared_ptr<_Interface> {
      return std::make_shared<_Impl>();
    }

    void __acquire_instance_lock() {
      while (__instance_locked_.exchange(true, STDEXEC::__std::memory_order_acquire)) {
        // Spin until we acquire the lock.
      }
    }

    void __release_instance_lock() {
      __instance_locked_.store(false, STDEXEC::__std::memory_order_release);
    }
  };

#if STDEXEC_ENABLE_LIBDISPATCH
  using __parallel_scheduler_backend_impl = __generic_impl<exec::libdispatch_queue>;
#elif STDEXEC_ENABLE_IO_URING
  using __parallel_scheduler_backend_impl = __generic_impl<exec::io_uring_context>;
#elif STDEXEC_ENABLE_WINDOWS_THREAD_POOL
  using __parallel_scheduler_backend_impl = __generic_impl<exec::windows_thread_pool>;
#else
  using __parallel_scheduler_backend_impl = __generic_impl<exec::static_thread_pool>;
#endif

  /// The singleton to hold the `parallel_scheduler_backend` instance.
  inline constinit __instance_data<
    STDEXEC::system_context_replaceability::parallel_scheduler_backend,
    __parallel_scheduler_backend_impl
  >
    __parallel_scheduler_backend_singleton{};

} // namespace STDEXEC::__system_context_default_impl
