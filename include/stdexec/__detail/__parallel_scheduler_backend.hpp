/*
 * Copyright (c) 2025 Lucian Radu Teodorescu, Lewis Baker
 * Copyright (c) 2026 NVIDIA Corporation
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

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
// #include "any_allocator.cuh"
#include "../functional.hpp" // IWYU pragma: keep for __with_default
#include "../stop_token.hpp"
#include "__queries.hpp"
#include "__typeinfo.hpp"

#include <exception>
#include <optional>
#include <span>

STDEXEC_PRAGMA_PUSH()
STDEXEC_PRAGMA_IGNORE_MSVC(4702) // warning C4702: unreachable code

namespace STDEXEC {
  template <class _Ty>
  class any_allocator : public std::allocator<_Ty> {
   public:
    template <class _OtherTy>
    struct rebind {
      using other = any_allocator<_OtherTy>;
    };

    template <__not_same_as<any_allocator> _Alloc>
    any_allocator(const _Alloc&) noexcept {
    }
  };

  template <class _Alloc>
  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
    any_allocator(_Alloc) -> any_allocator<typename _Alloc::value_type>;

  STDEXEC_HOST_DEVICE_DEDUCTION_GUIDE
  any_allocator(std::allocator<void>) -> any_allocator<std::byte>;

  class task_scheduler;

  // namespace __detail {
  //   struct __env_proxy : __immovable {
  //     [[nodiscard]]
  //     virtual auto query(const get_stop_token_t&) const noexcept -> inplace_stop_token = 0;
  //     [[nodiscard]]
  //     virtual auto query(const get_allocator_t&) const noexcept -> any_allocator<std::byte> = 0;
  //     [[nodiscard]]
  //     virtual auto query(const get_scheduler_t&) const noexcept -> task_scheduler = 0;
  //   };
  // } // namespace __detail

  namespace system_context_replaceability {
    /// Interface for completing a sender operation. Backend will call frontend though
    /// this interface for completing the `schedule` and `schedule_bulk` operations.
    class receiver_proxy { //: __detail::__env_proxy {
     public:
      virtual ~receiver_proxy() = 0;

      virtual void set_value() noexcept = 0;
      virtual void set_error(std::exception_ptr&&) noexcept = 0;
      virtual void set_stopped() noexcept = 0;

      // // NOT TO SPEC:
      // [[nodiscard]]
      // auto get_env() const noexcept -> const __detail::__env_proxy& {
      //   return *this;
      // }

      /// Query the receiver for a property of type `_P`.
      template <class _P, __class _Query>
      auto try_query(_Query) const noexcept -> std::optional<_P> {
        std::optional<_P> __p;
        __query_env(__mtypeid<_Query>, __mtypeid<_P>, &__p);
        return __p;
      }

     protected:
      virtual void __query_env(__type_index, __type_index, void*) const noexcept = 0;
    };

    inline receiver_proxy::~receiver_proxy() = default;

    struct bulk_item_receiver_proxy : receiver_proxy {
      virtual void execute(size_t, size_t) noexcept = 0;
    };

    /// Interface for the parallel scheduler backend.
    struct parallel_scheduler_backend {
      virtual ~parallel_scheduler_backend() = 0;

      /// Schedule work on parallel scheduler, calling `__r` when done and using `__s` for preallocated
      /// memory.
      virtual void schedule(receiver_proxy&, std::span<std::byte>) noexcept = 0;

      /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for different
      /// subranges of [0, __n), and using `__s` for preallocated memory.
      virtual void
        schedule_bulk_chunked(size_t, bulk_item_receiver_proxy&, std::span<std::byte>) noexcept = 0;

      /// Schedule bulk work of size `__n` on parallel scheduler, calling `__r` for each item, and
      /// using `__s` for preallocated memory.
      virtual void schedule_bulk_unchunked(
        size_t,
        bulk_item_receiver_proxy&,
        std::span<std::byte>) noexcept = 0;
    };

    inline parallel_scheduler_backend::~parallel_scheduler_backend() = default;
  } // namespace system_context_replaceability

  namespace __detail {
    // Partially implements the _RcvrProxy interface (either receiver_proxy or
    // bulk_item_receiver_proxy) in terms of a concrete receiver type _Rcvr.
    template <class _Rcvr, class _RcvrProxy>
    struct __receiver_proxy_base : _RcvrProxy {
     public:
      using receiver_concept = receiver_t;

      explicit __receiver_proxy_base(_Rcvr rcvr) noexcept
        : __rcvr_(static_cast<_Rcvr&&>(rcvr)) {
      }

      void set_error(std::exception_ptr&& eptr) noexcept final {
        STDEXEC::set_error(std::move(__rcvr_), std::move(eptr));
      }

      void set_stopped() noexcept final {
        STDEXEC::set_stopped(std::move(__rcvr_));
      }

     protected:
      void __query_env(__type_index __query_id, __type_index __value, void* __dest)
        const noexcept final {
        if (__query_id == __mtypeid<get_stop_token_t>) {
          __query(get_stop_token, __value, __dest);
        } else if (__query_id == __mtypeid<get_allocator_t>) {
          __query(get_allocator, __value, __dest);
        }
      }

     private:
      void __query(get_stop_token_t, __type_index __value_type, void* __dest) const noexcept {
        using __stop_token_t = stop_token_of_t<env_of_t<_Rcvr>>;
        if constexpr (std::is_same_v<inplace_stop_token, __stop_token_t>) {
          if (__value_type == __mtypeid<inplace_stop_token>) {
            using __dest_t = std::optional<inplace_stop_token>;
            *static_cast<__dest_t*>(__dest) = STDEXEC::get_stop_token(STDEXEC::get_env(__rcvr_));
          }
        }
      }

      void __query(get_allocator_t, __type_index __value_type, void* __dest) const noexcept {
        if (__value_type == __mtypeid<any_allocator<std::byte>>) {
          using __dest_t = std::optional<any_allocator<std::byte>>;
          *static_cast<__dest_t*>(__dest) = any_allocator{
            __with_default(get_allocator, std::allocator<std::byte>())(STDEXEC::get_env(__rcvr_))};
        }
      }

      //   [[nodiscard]]
      //   auto query(const get_stop_token_t&) const noexcept -> inplace_stop_token final {
      //     if constexpr (__callable<const get_stop_token_t&, env_of_t<_Rcvr>>) {
      //       if constexpr (__same_as<stop_token_of_t<env_of_t<_Rcvr>>, inplace_stop_token>) {
      //         return get_stop_token(get_env(__rcvr_));
      //       }
      //     }
      //     return inplace_stop_token{}; // MSVC thinks this is unreachable. :-?
      //   }

      //   [[nodiscard]]
      //   auto query(const get_allocator_t&) const noexcept -> any_allocator<std::byte> final {
      //     return any_allocator{
      //       __with_default(get_allocator, std::allocator<std::byte>())(get_env(__rcvr_))};
      //   }

      //   // defined in task_scheduler.cuh:
      //   [[nodiscard]]
      //   auto query(const get_scheduler_t& __query) const noexcept -> task_scheduler final;

     public:
      _Rcvr __rcvr_;
    };

    template <class _Rcvr>
    struct __receiver_proxy
      : __receiver_proxy_base<_Rcvr, system_context_replaceability::receiver_proxy> {
      using __receiver_proxy_base<
        _Rcvr,
        system_context_replaceability::receiver_proxy
      >::__receiver_proxy_base;

      void set_value() noexcept final {
        STDEXEC::set_value(std::move(this->__rcvr_));
      }
    };

    // A receiver type that forwards its completion operations to a _RcvrProxy member held by
    // reference (where _RcvrProxy is one of receiver_proxy or bulk_item_receiver_proxy). It
    // is also responsible to destroying and, if necessary, deallocating the operation state.
    template <class _RcvrProxy>
    struct __proxy_receiver {
      using receiver_concept = receiver_t;
      using __delete_fn_t = void(void*) noexcept;

      void set_value() noexcept {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_); // NB: destroys *this
        __proxy.set_value();
      }

      void set_error(std::exception_ptr eptr) noexcept {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_); // NB: destroys *this
        __proxy.set_error(std::move(eptr));
      }

      void set_stopped() noexcept {
        auto& __proxy = __rcvr_proxy_;
        __delete_fn_(__opstate_storage_); // NB: destroys *this
        __proxy.set_stopped();
      }

      [[nodiscard]]
      auto get_env() const noexcept -> env_of_t<_RcvrProxy> {
        return STDEXEC::get_env(__rcvr_proxy_);
      }

      _RcvrProxy& __rcvr_proxy_;
      void* __opstate_storage_;
      __delete_fn_t* __delete_fn_;
    };
  } // namespace __detail
} // namespace STDEXEC

STDEXEC_PRAGMA_POP()
