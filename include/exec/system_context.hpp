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

#include <utility>

#include "../stdexec/execution.hpp"
#include "__detail/__system_context_replaceability_api.hpp"

#ifndef STDEXEC_SYSTEM_CONTEXT_SCHEDULE_OP_SIZE
#  define STDEXEC_SYSTEM_CONTEXT_SCHEDULE_OP_SIZE 72
#endif
#ifndef STDEXEC_SYSTEM_CONTEXT_SCHEDULE_OP_ALIGN
#  define STDEXEC_SYSTEM_CONTEXT_SCHEDULE_OP_ALIGN 8
#endif
#ifndef STDEXEC_SYSTEM_CONTEXT_BULK_SCHEDULE_OP_SIZE
#  define STDEXEC_SYSTEM_CONTEXT_BULK_SCHEDULE_OP_SIZE 152
#endif
#ifndef STDEXEC_SYSTEM_CONTEXT_BULK_SCHEDULE_OP_ALIGN
#  define STDEXEC_SYSTEM_CONTEXT_BULK_SCHEDULE_OP_ALIGN 8
#endif

// TODO: make these configurable by providing policy to the system context

namespace exec {
  namespace __detail {
    using namespace stdexec::tags;

    /// Allows a frontend receiver of type `_Rcvr` to be passed to the backend.
    template <class _Rcvr>
    struct __receiver_adapter : system_context_replaceability::receiver {
      explicit __receiver_adapter(_Rcvr&& __rcvr)
        : __rcvr_{std::forward<_Rcvr>(__rcvr)} {
      }

      auto __query_env(__uuid __id, void* __dest) noexcept -> bool override {
        using system_context_replaceability::__runtime_property_helper;
        using __StopToken = decltype(stdexec::get_stop_token(stdexec::get_env(__rcvr_)));
        if constexpr (std::is_same_v<stdexec::inplace_stop_token, __StopToken>) {
          if (__id == __runtime_property_helper<stdexec::inplace_stop_token>::__property_identifier) {
            *static_cast<stdexec::inplace_stop_token*>(__dest) = stdexec::get_stop_token(
              stdexec::get_env(__rcvr_));
            return true;
          }
        }
        return false;
      }

      void set_value() noexcept override {
        stdexec::set_value(std::forward<_Rcvr>(__rcvr_));
      }

      void set_error(std::exception_ptr __ex) noexcept override {
        stdexec::set_error(std::forward<_Rcvr>(__rcvr_), std::move(__ex));
      }

      void set_stopped() noexcept override {
        stdexec::set_stopped(std::forward<_Rcvr>(__rcvr_));
      }

      STDEXEC_ATTRIBUTE(no_unique_address)
      _Rcvr __rcvr_;
    };

    /// The type large enough to store the data produced by a sender.
    template <class _Sender>
    using __sender_data_t = decltype(stdexec::sync_wait(std::declval<_Sender>()).value());

  } // namespace __detail

  class parallel_scheduler;
  class __parallel_sender;
  template <bool, stdexec::sender _S, std::integral _Size, class _Fn, bool>
  class __parallel_bulk_sender;

  /// Returns a scheduler that can add work to the underlying execution context.
  auto get_parallel_scheduler() -> parallel_scheduler;

  /// Concept that matches `bulk_chunked` and `bulk_unchunked` senders.
  template <class _Sender>
  concept __bulk_chunked_or_unchunked =
    stdexec::sender_expr_for<_Sender, stdexec::bulk_chunked_t>
    || stdexec::sender_expr_for<_Sender, stdexec::bulk_unchunked_t>;

  /// The execution domain of the parallel_scheduler, used for the purposes of customizing
  /// sender algorithms such as `bulk_chunked` and `bulk_unchunked`.
  struct __parallel_scheduler_domain : stdexec::default_domain {
    template <__bulk_chunked_or_unchunked _Sender>
    auto transform_sender(_Sender&& __sndr) const noexcept;
    template <__bulk_chunked_or_unchunked _Sender, class _Env>
    auto transform_sender(_Sender&& __sndr, const _Env& __env) const noexcept;
  };

  namespace __detail {
    using __backend_ptr =
      std::shared_ptr<system_context_replaceability::parallel_scheduler_backend>;

    template <class T>
    auto __make_parallel_scheduler_from(T, __backend_ptr) noexcept;

    /// Describes the environment of this sender.
    struct __parallel_scheduler_env {
      /// Returns the system scheduler as the completion scheduler for `set_value_t`.
      template <stdexec::__one_of<stdexec::set_value_t> _Tag>
      [[nodiscard]]
      auto query(stdexec::get_completion_scheduler_t<_Tag>) const noexcept {
        return __detail::__make_parallel_scheduler_from(_Tag(), __scheduler_);
      }

      /// The underlying implementation of the scheduler we are using.
      __backend_ptr __scheduler_;
    };

    template <size_t _Size, size_t _Align>
    struct __aligned_storage {
      alignas(_Align) unsigned char __data_[_Size];

      auto __as_storage() noexcept -> std::span<std::byte> {
        return {reinterpret_cast<std::byte*>(__data_), _Size};
      }

      template <class _T>
      auto __as() noexcept -> _T& {
        static_assert(alignof(_T) <= _Align);
        return *reinterpret_cast<_T*>(__data_);
      }

      auto __as_ptr() noexcept -> void* {
        return __data_;
      }
    };

    /*
    Storage needed for a frontend operation-state:

    schedule:
    - __receiver_adapter::__vtable -- 8
    - __receiver_adapter::__rcvr_ (Rcvr) -- assuming 0
    - __system_op::__preallocated_ (__preallocated) -- 72
    ---------------------
    Total: 80; extra 8 bytes compared to backend needs.

    for bulk:
    - __bulk_state_base::__fun_ (_Fn) -- 0 (assuming empty function)
    - __bulk_state_base::__rcvr_ (_Rcvr) -- 0 (assuming empty receiver)
    - __forward_args_receiver::__vtable -- 8
    - __forward_args_receiver::__arguments_data_ (array of bytes) -- 8 (depending on previous sender)
    - __bulk_state_base::__prepare_storage_for_backend (fun ptr) -- 8
    - __bulk_state_base::__size_ (_Size) -- 4
    - __bulk_state::__preallocated_ (__preallocated_) -- 152
      - __previous_operation_state_ (__inner_op_state) -- 104
        - __bulk_intermediate_receiver::__state_ (__state_&) -- 8
        - __bulk_intermediate_receiver::__scheduler_ (parallel_scheduler*) -- 8
    ---------------------
    Total: 176; extra 24 bytes compared to backend needs.

    [*] sizes taken on an Apple M2 Pro arm64 arch. They may differ on other architectures, or with different implementations.
    */

    /// The operation state used to execute the work described by this sender.
    template <class _S, class _Rcvr>
    struct __system_op {
      /// Constructs `this` from `__rcvr` and `__scheduler_impl`.
      __system_op(_Rcvr&& __rcvr, __backend_ptr __scheduler_impl)
        : __rcvr_{std::forward<_Rcvr>(__rcvr)} {
        // Before the operation starts, we store the scheduelr implementation in __preallocated_.
        // After the operation starts, we don't need this pointer anymore, and the storage can be used by the backend
        auto* __p = &__preallocated_.__as<__backend_ptr>();
        std::construct_at(__p, std::move(__scheduler_impl));
      }

      ~__system_op() = default;

      __system_op(const __system_op&) = delete;
      __system_op(__system_op&&) = delete;
      auto operator=(const __system_op&) -> __system_op& = delete;
      auto operator=(__system_op&&) -> __system_op& = delete;

      /// Starts the work stored in `this`.
      void start() & noexcept {
        auto st = stdexec::get_stop_token(stdexec::get_env(__rcvr_.__rcvr_));
        if (st.stop_requested()) {
          stdexec::set_stopped(__rcvr_);
          return;
        }
        auto& __scheduler_impl = __preallocated_.__as<__backend_ptr>();
        auto __impl = std::move(__scheduler_impl);
        std::destroy_at(&__scheduler_impl);
        __impl->schedule(__preallocated_.__as_storage(), __rcvr_);
      }

      /// Object that receives completion from the work described by the sender.
      __receiver_adapter<_Rcvr> __rcvr_;

      /// Preallocated space for storing the operation state on the implementation size.
      /// We also store here the backend interface for the scheduler before we actually start the operation.
      __aligned_storage<
        STDEXEC_SYSTEM_CONTEXT_SCHEDULE_OP_SIZE,
        STDEXEC_SYSTEM_CONTEXT_SCHEDULE_OP_ALIGN
      >
        __preallocated_;
    };
  } // namespace __detail

  /// The sender used to schedule new work in the system context.
  class __parallel_sender {
   public:
    /// Marks this type as being a sender; not to spec.
    using sender_concept = stdexec::sender_t;
    /// Declares the completion signals sent by `this`.
    using completion_signatures = stdexec::completion_signatures<
      stdexec::set_value_t(),
      stdexec::set_stopped_t(),
      stdexec::set_error_t(std::exception_ptr)
    >;

    /// Implementation detail. Constructs the sender to wrap `__impl`.
    explicit __parallel_sender(__detail::__backend_ptr __impl)
      : __scheduler_{std::move(__impl)} {
    }

    /// Gets the environment of this sender.
    [[nodiscard]]
    auto get_env() const noexcept -> __detail::__parallel_scheduler_env {
      return {__scheduler_};
    }

    /// Value completion happens on the parallel scheduler.
    [[nodiscard]]
    auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept
      -> parallel_scheduler;

    /// Connects `__self` to `__rcvr`, returning the operation state containing the work to be done.
    template <stdexec::receiver _Rcvr>
    auto connect(_Rcvr __rcvr) && noexcept(stdexec::__nothrow_move_constructible<_Rcvr>)
      -> __detail::__system_op<__parallel_sender, _Rcvr> {
      return {std::move(__rcvr), std::move(__scheduler_)};
    }

    template <stdexec::receiver _Rcvr>
    auto connect(_Rcvr __rcvr) & noexcept(stdexec::__nothrow_move_constructible<_Rcvr>)
      -> __detail::__system_op<__parallel_sender, _Rcvr> {
      return {std::move(__rcvr), __scheduler_};
    }

   private:
    /// The underlying implementation of the system scheduler.
    __detail::__backend_ptr __scheduler_;
  };

  /// A scheduler that can add work to the system context.
  class parallel_scheduler {
   public:
    parallel_scheduler() = delete;

    /// Returns `true` iff `*this` refers to the same scheduler as the argument.
    auto operator==(const parallel_scheduler&) const noexcept -> bool = default;

    /// Implementation detail. Constructs the scheduler to wrap `__impl`.
    explicit parallel_scheduler(__detail::__backend_ptr&& __impl)
      : __impl_(__impl) {
    }

    /// Returns the forward progress guarantee of `this`.
    [[nodiscard]]
    auto query(stdexec::get_forward_progress_guarantee_t) const noexcept
      -> stdexec::forward_progress_guarantee;

    /// Returns the execution domain of `this`.
    [[nodiscard]]
    auto query(stdexec::get_domain_t) const noexcept -> __parallel_scheduler_domain {
      return {};
    }

    /// Schedules new work, returning the sender that signals the start of the work.
    [[nodiscard]]
    auto schedule() const noexcept -> __parallel_sender {
      return __parallel_sender{__impl_};
    }

   private:
    template <bool, stdexec::sender, std::integral, class, bool>
    friend class __parallel_bulk_sender;

    /// The underlying implementation of the scheduler.
    __detail::__backend_ptr __impl_;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // bulk

  namespace __detail {
    template <class T>
    auto __make_parallel_scheduler_from(T, __backend_ptr __impl) noexcept {
      return parallel_scheduler{std::move(__impl)};
    }

    /// Helper that knows how to store the values sent by `_Previous` and pass them to bulk item calls or to the completion signal.
    /// This represents the base class that abstracts the storage of the values sent by the previous sender.
    /// Derived class will properly implement the receiver methods.
    template <class _Previous>
    struct __forward_args_receiver : system_context_replaceability::bulk_item_receiver {
      using __storage_t = __detail::__sender_data_t<_Previous>;

      /// Storage for the arguments received from the previous sender.
      alignas(__storage_t) unsigned char __arguments_data_[sizeof(__storage_t)];
    };

    /// Derived class that properly forwards the arguments received from `_Previous` to the receiver methods.
    /// Uses the storage defined in the base class. No extra data is added here.
    template <class _Previous, class _BulkState, class... _As>
    struct __typed_forward_args_receiver : __forward_args_receiver<_Previous> {
      using __base_t = __forward_args_receiver<_Previous>;
      using __rcvr_t = typename _BulkState::__rcvr_t;

      /// Stores `__as` in the base class storage, with the right types.
      explicit __typed_forward_args_receiver(_As&&... __as) {
        static_assert(sizeof(std::tuple<_As...>) <= sizeof(__base_t::__arguments_data_));
        new (__base_t::__arguments_data_)
          std::tuple<stdexec::__decay_t<_As>...>{std::move(__as)...};
      }

      auto __query_env(__uuid __id, void* __dest) noexcept -> bool override {
        auto __state = reinterpret_cast<_BulkState*>(this);
        using system_context_replaceability::__runtime_property_helper;
        using __StopToken = decltype(stdexec::get_stop_token(stdexec::get_env(__state->__rcvr_)));
        if constexpr (std::is_same_v<stdexec::inplace_stop_token, __StopToken>) {
          if (__id == __runtime_property_helper<stdexec::inplace_stop_token>::__property_identifier) {
            *static_cast<stdexec::inplace_stop_token*>(__dest) = stdexec::get_stop_token(
              stdexec::get_env(__state->__rcvr_));
            return true;
          }
        }
        return false;
      }

      /// Calls `set_value()` on the final receiver of the bulk operation, using the values from the previous sender.
      void set_value() noexcept override {
        auto __state = reinterpret_cast<_BulkState*>(this);
        std::apply(
          [&](auto&&... __args) {
            stdexec::set_value(
              std::forward<__rcvr_t>(__state->__rcvr_), std::forward<_As>(__args)...);
          },
          *reinterpret_cast<std::tuple<_As...>*>(__base_t::__arguments_data_));
      }

      /// Calls `set_error()` on the final receiver of the bulk operation, passing `__ex`.
      void set_error(std::exception_ptr __ex) noexcept override {
        auto __state = reinterpret_cast<_BulkState*>(this);
        stdexec::set_error(std::forward<__rcvr_t>(__state->__rcvr_), std::move(__ex));
      }

      /// Calls `set_stopped()` on the final receiver of the bulk operation.
      void set_stopped() noexcept override {
        auto __state = reinterpret_cast<_BulkState*>(this);
        stdexec::set_stopped(std::forward<__rcvr_t>(__state->__rcvr_));
      }

      /// Calls the bulk functor passing `__index` and the values from the previous sender.
      void execute(uint32_t __begin, uint32_t __end) noexcept override {
        auto __state = reinterpret_cast<_BulkState*>(this);
        if constexpr (_BulkState::__is_unchunked) {
          (void) __end; // not used
          // If we are not parallelizing, we need to run all the iterations sequentially.
          uint32_t __increments = 1;
          if constexpr (!_BulkState::__parallelize) {
            __increments = static_cast<uint32_t>(__state->__size_);
          }
          for (uint32_t __i = __begin; __i < __begin + __increments; __i++) {
            std::apply(
              [&](auto&&... __args) { __state->__fun_(__i, __args...); },
              *reinterpret_cast<std::tuple<_As...>*>(__base_t::__arguments_data_));
          }
        } else {
          // If we are not parallelizing, we need to pass the entire range to the functor.
          if constexpr (!_BulkState::__parallelize) {
            __begin = 0;
            __end = static_cast<uint32_t>(__state->__size_);
          }
          std::apply(
            [&](auto&&... __args) { __state->__fun_(__begin, __end, __args...); },
            *reinterpret_cast<std::tuple<_As...>*>(__base_t::__arguments_data_));
        }
      }
    };

    /// The state needed to execute the bulk sender created from system context, minus the preallocates space.
    /// The preallocated space is obtained by calling the `__prepare_storage_for_backend` function pointer.
    template <
      stdexec::sender _Previous,
      std::integral _Size,
      class _Fn,
      class _Rcvr,
      bool _IsUnchunked,
      bool _Parallelize
    >
    struct __bulk_state_base {
      using __rcvr_t = _Rcvr;
      using __forward_args_helper_t = __forward_args_receiver<_Previous>;
      static constexpr bool __is_unchunked = _IsUnchunked;
      static constexpr bool __parallelize = _Parallelize;

      /// Storage for the arguments and the helper needed to pass the arguments from the previous bulk sender to the bulk functor and receiver.
      /// Needs to be the first member, to easier the convertion between `__forward_args_helper_` and `this`.
      alignas(__forward_args_helper_t) unsigned char __forward_args_helper_[sizeof(
        __forward_args_helper_t)]{};

      /// The function to be executed to perform the bulk work.
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Fn __fun_;
      /// The receiver object that receives completion from the work described by the sender.
      STDEXEC_ATTRIBUTE(no_unique_address)
      _Rcvr __rcvr_;

      /// Function that prepares the preallocated storage for calling the backend.
      std::span<std::byte> (*__prepare_storage_for_backend)(__bulk_state_base*){nullptr};
      /// The size of the bulk operation.
      _Size __size_;

      __bulk_state_base(_Fn&& __fun, _Rcvr&& __rcvr, _Size __size)
        : __fun_{std::move(__fun)}
        , __rcvr_{std::move(__rcvr)}
        , __size_{__size} {
      }
    };

    /// Receiver that is used in "bulk" to connect to the input sender of the bulk operation.
    template <class _BulkState, stdexec::sender _Previous>
    struct __bulk_intermediate_receiver {
      /// Declare that this is a `receiver`.
      using receiver_concept = stdexec::receiver_t;

      /// Object that holds the relevant data for the entire bulk operation.
      _BulkState& __state_;
      /// The underlying implementation of the scheduler we are using.
      __backend_ptr __scheduler_{nullptr};

      template <class... _As>
      void set_value(_As&&... __as) noexcept {
        auto st = stdexec::get_stop_token(stdexec::get_env(__state_.__rcvr_));
        if (st.stop_requested()) {
          stdexec::set_stopped(__state_.__rcvr_);
          return;
        }

        // Store the input data in the shared state.
        using __typed_forward_args_receiver_t =
          __typed_forward_args_receiver<_Previous, _BulkState, _As...>;
        auto __r = new (&__state_.__forward_args_helper_)
          __typed_forward_args_receiver_t(std::forward<_As>(__as)...);

        auto __scheduler = __scheduler_;
        auto __size = static_cast<uint32_t>(__state_.__size_);

        auto __storage = __state_.__prepare_storage_for_backend(&__state_);
        // This might destroy the `this` object.

        // Schedule the bulk work on the system scheduler.
        // This will invoke `execute` on our receiver multiple times, and then a completion signal (e.g., `set_value`).
        if constexpr (_BulkState::__is_unchunked) {
          __scheduler
            ->schedule_bulk_unchunked(_BulkState::__parallelize ? __size : 1, __storage, *__r);
        } else {
          __scheduler
            ->schedule_bulk_chunked(_BulkState::__parallelize ? __size : 1, __storage, *__r);
        }
      }

      /// Invoked when the previous sender completes with "stopped" to stop the entire work.
      void set_stopped() noexcept {
        stdexec::set_stopped(std::move(__state_.__rcvr_));
      }

      /// Invoked when the previous sender completes with error to forward the error to the connected receiver.
      template <typename __E>
      void set_error(__E __e) noexcept {
        stdexec::set_error(std::move(__state_.__rcvr_), std::move(__e));
      }

      /// Gets the environment of this receiver; returns the environment of the connected receiver.
      [[nodiscard]]
      auto get_env() const noexcept -> decltype(auto) {
        return stdexec::get_env(__state_.__rcvr_);
      }
    };

    /// The operation state object for the system bulk sender.
    template <
      bool _IsUnchunked,
      stdexec::sender _Previous,
      std::integral _Size,
      class _Fn,
      class _Rcvr,
      bool _Parallelize
    >
    struct __system_bulk_op
      : __bulk_state_base<_Previous, _Size, _Fn, _Rcvr, _IsUnchunked, _Parallelize> {

      /// The type that holds the state of the bulk operation.
      using __bulk_state_base_t =
        __bulk_state_base<_Previous, _Size, _Fn, _Rcvr, _IsUnchunked, _Parallelize>;

      /// The type of the receiver that will be connected to the previous sender.
      using __intermediate_receiver_t =
        __bulk_intermediate_receiver<__bulk_state_base_t, _Previous>;

      /// The type of inner operation state, which is the result of connecting the previous sender to the bulk intermediate receiver.
      using __inner_op_state = stdexec::connect_result_t<_Previous, __intermediate_receiver_t>;

      static constexpr size_t _PreallocatedSize =
        std::max(size_t(STDEXEC_SYSTEM_CONTEXT_BULK_SCHEDULE_OP_SIZE), sizeof(__inner_op_state));
      static constexpr size_t _PreallocatedAlign =
        std::max(size_t(STDEXEC_SYSTEM_CONTEXT_BULK_SCHEDULE_OP_ALIGN), alignof(__inner_op_state));

      /// Preallocated space for storing the inner operation state, and then storage space for the backend call.
      __aligned_storage<_PreallocatedSize, _PreallocatedAlign> __preallocated_;

      /// Destroys the inner operation state object, and returns the preallocated storage for it to be used by the backend.
      static auto
        __prepare_storage_for_backend_impl(__bulk_state_base_t* __base) -> std::span<std::byte> {
        auto* __self = static_cast<__system_bulk_op*>(__base);
        // We don't need anymore the storage for the previous operation state.
        __self->__preallocated_.template __as<__inner_op_state>().~__inner_op_state();
        // Reuse the preallocated storage for the backend.
        return __self->__preallocated_.__as_storage();
      }

      /// Constructs `this` from `__snd` and `__rcvr`, using the object returned by `__initFunc` to start the operation.
      ///
      /// Using a functor to initialize the operation state allows the use of `this` to get the
      /// underlying implementation object.
      ///
      /// `_Snd` is a `__parallel_bulk_sender`.
      template <class _Snd, class _InitF>
      __system_bulk_op(_Snd&& __snd, _Rcvr&& __rcvr, _InitF&& __initFunc)
        : __bulk_state_base_t{std::move(__snd.__fun_), std::move(__rcvr), __snd.__size_} {
        // Write the function that prepares the storage for the backend.
        __bulk_state_base_t::__prepare_storage_for_backend =
          &__system_bulk_op::__prepare_storage_for_backend_impl;

        // Start using the preallocated buffer to store the inner operation state.
        new (__preallocated_.__as_ptr()) __inner_op_state(__initFunc(*this));
      }

      __system_bulk_op(const __system_bulk_op&) = delete;
      __system_bulk_op(__system_bulk_op&&) = delete;
      auto operator=(const __system_bulk_op&) -> __system_bulk_op& = delete;
      auto operator=(__system_bulk_op&&) -> __system_bulk_op& = delete;

      /// Starts the work stored in `*this`.
      void start() & noexcept {
        // Start previous operation state.
        // Bulk operation will be started when the previous sender completes.
        stdexec::start(__preallocated_.template __as<__inner_op_state>());
      }
    };
  } // namespace __detail

  /// The sender used to schedule bulk work in the system context.
  template <
    bool _IsUnchunked,
    stdexec::sender _Previous,
    std::integral _Size,
    class _Fn,
    bool _Parallelize
  >
  class __parallel_bulk_sender {
    /// Meta-function that returns the completion signatures of `this`.
    template <class _Self, class... _Env>
    using __completions_t = stdexec::transform_completion_signatures<
      stdexec::__completion_signatures_of_t<stdexec::__copy_cvref_t<_Self, _Previous>, _Env...>,
      stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr)>
    >;

    template <bool, stdexec::sender, std::integral, class, class, bool>
    friend struct __detail::__system_bulk_op;

   public:
    /// Marks this type as being a sender
    using sender_concept = stdexec::sender_t;

    /// Constructs `this`.
    __parallel_bulk_sender(
      parallel_scheduler __sched,
      _Previous __previous,
      _Size __size,
      _Fn&& __fun)
      : __scheduler_{__sched.__impl_}
      , __previous_{std::move(__previous)}
      , __size_{std::move(__size)}
      , __fun_{std::move(__fun)} {
    }

    /// Gets the environment of this sender.
    [[nodiscard]]
    auto get_env() const noexcept -> __detail::__parallel_scheduler_env {
      return {__scheduler_};
    }

    /// Connects `__self` to `__rcvr`, returning the operation state containing the work to be done.
    template <stdexec::receiver _Rcvr>
    auto connect(_Rcvr __rcvr) && noexcept(stdexec::__nothrow_move_constructible<_Rcvr>)
      -> __detail::__system_bulk_op<_IsUnchunked, _Previous, _Size, _Fn, _Rcvr, _Parallelize> {
      using __res_t =
        __detail::__system_bulk_op<_IsUnchunked, _Previous, _Size, _Fn, _Rcvr, _Parallelize>;
      using __receiver_t = typename __res_t::__intermediate_receiver_t;
      return {std::move(*this), std::move(__rcvr), [this](auto& __op) {
                // Connect bulk input receiver with the previous operation and store in the operating state.
                return stdexec::connect(
                  std::move(this->__previous_), __receiver_t{__op, std::move(this->__scheduler_)});
              }};
    }

    /// Gets the completion signatures for this sender.
    template <stdexec::__decays_to<__parallel_bulk_sender> _Self, class... _Env>
    static auto get_completion_signatures(_Self&&, _Env&&...) -> __completions_t<_Self, _Env...> {
      return {};
    }

   private:
    /// The underlying implementation of the scheduler we are using.
    __detail::__backend_ptr __scheduler_{nullptr};
    /// The previous sender, the one that produces the input value for the bulk function.
    _Previous __previous_;
    /// The size of the bulk operation.
    _Size __size_;
    /// The function to be executed to perform the bulk work.
    STDEXEC_ATTRIBUTE(no_unique_address)
    _Fn __fun_;
  };

  inline auto get_parallel_scheduler() -> parallel_scheduler {
    auto __impl = system_context_replaceability::query_parallel_scheduler_backend();
    if (!__impl) {
      STDEXEC_THROW(std::runtime_error{"No system context implementation found"});
    }
    return parallel_scheduler{std::move(__impl)};
  }

  [[deprecated("get_system_scheduler has been renamed get_parallel_scheduler")]]
  inline auto get_system_scheduler() -> parallel_scheduler {
    return get_parallel_scheduler();
  }

  inline auto __parallel_sender::query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>)
    const noexcept -> parallel_scheduler {
    return __detail::__make_parallel_scheduler_from(stdexec::set_value_t{}, __scheduler_);
  }

  inline auto parallel_scheduler::query(stdexec::get_forward_progress_guarantee_t) const noexcept
    -> stdexec::forward_progress_guarantee {
    return stdexec::forward_progress_guarantee::parallel;
  }

  struct __transform_parallel_bulk_sender {
    template <class _Data, class _Previous>
    auto
      operator()(stdexec::bulk_chunked_t, _Data&& __data, _Previous&& __previous) const noexcept {
      auto [__pol, __shape, __fn] = static_cast<_Data&&>(__data);
      using __policy_t = std::remove_cvref_t<decltype(__pol.__get())>;
      constexpr bool __parallelize = std::same_as<__policy_t, stdexec::parallel_policy>
                                  || std::same_as<__policy_t, stdexec::parallel_unsequenced_policy>;
      return __parallel_bulk_sender<
        false,
        _Previous,
        decltype(__shape),
        decltype(__fn),
        __parallelize
      >{__sched_, static_cast<_Previous&&>(__previous), __shape, std::move(__fn)};
    }

    template <class _Data, class _Previous>
    auto
      operator()(stdexec::bulk_unchunked_t, _Data&& __data, _Previous&& __previous) const noexcept {
      auto [__pol, __shape, __fn] = static_cast<_Data&&>(__data);
      using __policy_t = std::remove_cvref_t<decltype(__pol.__get())>;
      constexpr bool __parallelize = std::same_as<__policy_t, stdexec::parallel_policy>
                                  || std::same_as<__policy_t, stdexec::parallel_unsequenced_policy>;
      return __parallel_bulk_sender<
        true,
        _Previous,
        decltype(__shape),
        decltype(__fn),
        __parallelize
      >{__sched_, static_cast<_Previous&&>(__previous), __shape, std::move(__fn)};
    }

    parallel_scheduler __sched_;
  };

  template <class _Sender>
  struct __not_a_sender {
    using sender_concept = stdexec::sender_t;
  };

  template <__bulk_chunked_or_unchunked _Sender>
  auto __parallel_scheduler_domain::transform_sender(_Sender&& __sndr) const noexcept {
    if constexpr (stdexec::__completes_on<_Sender, parallel_scheduler>) {
      auto __sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(
        stdexec::get_env(__sndr));
      return stdexec::__sexpr_apply(
        static_cast<_Sender&&>(__sndr), __transform_parallel_bulk_sender{__sched});
    } else {
      static_assert(
        stdexec::__completes_on<_Sender, parallel_scheduler>,
        "No parallel_scheduler instance can be found in the sender's "
        "attributes on which to schedule bulk work.");
      return __not_a_sender<stdexec::__name_of<_Sender>>();
    }
  }

  template <__bulk_chunked_or_unchunked _Sender, class _Env>
  auto __parallel_scheduler_domain::transform_sender(_Sender&& __sndr, const _Env& __env)
    const noexcept {
    if constexpr (stdexec::__starts_on<_Sender, parallel_scheduler, _Env>) {
      auto __sched = stdexec::get_scheduler(__env);
      return stdexec::__sexpr_apply(
        static_cast<_Sender&&>(__sndr), __transform_parallel_bulk_sender{__sched});
    } else {
      static_assert(
        stdexec::__starts_on<_Sender, parallel_scheduler, _Env>,
        "No parallel_scheduler instance can be found in the receiver's "
        "environment on which to schedule bulk work.");
      return __not_a_sender<stdexec::__name_of<_Sender>>();
    }
  }
} // namespace exec

#if defined(STDEXEC_SYSTEM_CONTEXT_HEADER_ONLY)
#  define STDEXEC_SYSTEM_CONTEXT_INLINE inline
#  include "__detail/__system_context_default_impl_entry.hpp"
#endif
